"""
AMLNet - imbalance-handling pipeline components.
Implements the two strategies described in the research report, in the exact
order specified by the project data pipeline:
  Strategy 1 - SMOTENC
      scaling  ->  SMOTENC  ->  one-hot encoding  ->  model
      (NO one-hot encoding before SMOTENC; encoding happens afterwards)
  Strategy 2 - Class weighting
      scaling + one-hot encoding  ->  model(class_weight=...)
Why SMOTENC and not SMOTE
-------------------------
AMLNet contains two categorical features (`type`, `category`). Plain SMOTE
interpolates linearly between neighbours, which would produce meaningless
fractional category values. SMOTENC (Chawla et al., 2002) handles nominal
features by assigning the majority category among the k nearest neighbours,
so categories stay discrete and valid.
Memory note
-----------
A ColumnTransformer that scales numerics and passes string categoricals
through returns an `object` dtype array. At 763,120 x 21 that exhausts memory
before SMOTENC even starts (verified: OOM-killed on a 2 GB machine).
`ScaleAndCodeCategoricals` avoids this by scaling the numeric columns and
representing categoricals as integer *codes* inside a float32 frame. This is
semantically identical for SMOTENC - it is told exactly which column indices
are categorical, and it assigns whole-number codes by neighbour majority
vote - but it uses ~70 MB instead of several GB. Codes are expanded back into
one-hot columns after resampling by `OneHotAfterResampling`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
class ScaleAndCodeCategoricals(BaseEstimator, TransformerMixin):
    """
    Scale numeric features; encode categoricals as integer codes.
    Output is a float32 numpy array with numeric columns first, then
    categorical code columns, so the categorical indices handed to SMOTENC are
    simply the last `len(categorical_features)` positions.
    """
    def __init__(self, numeric_features, categorical_features):
        # sklearn clone() requires __init__ to store params unmodified.
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
    @property
    def categorical_indices_(self):
        n = len(list(self.numeric_features))
        return list(range(n, n + len(list(self.categorical_features))))
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        num = list(self.numeric_features)
        cat = list(self.categorical_features)
        self.scaler_ = StandardScaler().fit(X[num])
        self.categories_ = {
            col: pd.Index(pd.Series(X[col]).astype(str).unique()).sort_values()
            for col in cat
        }
        self.feature_names_in_ = num + cat
        return self
    def transform(self, X):
        X = pd.DataFrame(X)
        num = self.scaler_.transform(
            X[list(self.numeric_features)]).astype(np.float32)
        cats = []
        for col in list(self.categorical_features):
            codes = pd.Categorical(
                pd.Series(X[col]).astype(str),
                categories=self.categories_[col],
            ).codes.astype(np.float32)
            # Unseen categories become -1; map them to 0 so downstream one-hot
            # treats them as the first level rather than crashing.
            codes[codes < 0] = 0
            cats.append(codes.reshape(-1, 1))
        if cats:
            return np.hstack([num] + cats).astype(np.float32)

        return num
class OneHotAfterResampling(BaseEstimator, TransformerMixin):
    """
    Expand the integer code columns produced by `ScaleAndCodeCategoricals`
    into one-hot columns. Applied AFTER SMOTENC, as the report specifies.
    """
    def __init__(self, categorical_indices, categories):
        # Stored unmodified so sklearn clone() works.
        self.categorical_indices = categorical_indices
        self.categories = categories  # dict: feature name -> list of levels
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        cat_idx = list(self.categorical_indices)
        self.numeric_indices_ = [
            i for i in range(self.n_features_in_) if i not in cat_idx
        ]
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        blocks = [X[:, self.numeric_indices_]]
        for pos, idx in enumerate(list(self.categorical_indices)):
            name = list(self.categories.keys())[pos]
            n_levels = len(self.categories[name])
            codes = np.rint(X[:, idx]).astype(int)
            codes = np.clip(codes, 0, n_levels - 1)
            onehot = np.zeros((X.shape[0], n_levels), dtype=np.float32)
            onehot[np.arange(X.shape[0]), codes] = 1.0
            blocks.append(onehot)
        return np.hstack(blocks).astype(np.float32)
    def get_feature_names_out(self, input_features=None):
        names = []
        num_names = [f"num_{i}" for i in self.numeric_indices_]
        if input_features is not None:
            num_names = [input_features[i] for i in self.numeric_indices_]
        names.extend(num_names)
        for name, levels in self.categories.items():
            names.extend([f"{name}_{lvl}" for lvl in levels])
        return np.array(names, dtype=object)
def build_smotenc_pipeline(estimator, numeric_features, categorical_features,
                           X_reference, sampling_strategy, random_state=42,
                           k_neighbors=5):
    """
    scaling -> SMOTENC -> one-hot -> model
    `X_reference` supplies the categorical levels so the one-hot step knows
    how many columns each categorical expands into.
    """
    from imblearn.over_sampling import SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
    coder = ScaleAndCodeCategoricals(numeric_features, categorical_features)
    coder.fit(X_reference)
    cat_idx = coder.categorical_indices_
    categories = {c: list(coder.categories_[c]) for c in categorical_features}
    return ImbPipeline([
        ("scale_and_code", ScaleAndCodeCategoricals(
            numeric_features, categorical_features)),
        ("smotenc", SMOTENC(categorical_features=cat_idx,
                            sampling_strategy=sampling_strategy,
                            k_neighbors=k_neighbors,
                            random_state=random_state)),
        ("onehot_after_smotenc", OneHotAfterResampling(cat_idx, categories)),
        ("model", estimator),
    ])
def smotenc_feature_names(numeric_features, categorical_features, X_reference):
    """Final feature names emitted by a SMOTENC pipeline (for explainability)."""
    coder = ScaleAndCodeCategoricals(numeric_features, categorical_features)
    coder.fit(X_reference)
    names = list(numeric_features)
    for col in categorical_features:
        names.extend([f"{col}_{lvl}" for lvl in coder.categories_[col]])
    return names
def build_class_weight_pipeline(estimator, numeric_features,
                                categorical_features, scale_numeric=True):
    """
    scaling + one-hot -> model(class_weight=...)
    Preprocessing is fitted inside the pipeline, so it only ever sees
    training folds.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:                                       # sklearn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", encoder, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    return Pipeline([("preprocessor", pre), ("model", estimator)])
