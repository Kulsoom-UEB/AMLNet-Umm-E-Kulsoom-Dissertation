"""
Step 08: Explainability - SHAP, feature importance, permutation importance.
CRISP-DM: Evaluation.
Explains whichever model step 06 selected. Handles both pipeline shapes
(class-weighting ColumnTransformer and SMOTENC scale->resample->one-hot)
without hardcoding either, so the explanations stay correct no matter which
model wins.
One-hot columns are aggregated back to their original feature groups so the
report can discuss "transaction category" rather than 11 dummy columns.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
import amlnet_resampling as R  # noqa: E402
PERMUTATION_SAMPLE = 30000
SHAP_SAMPLE = 1500
N_LOCAL_EXAMPLES = 5
def resolve_feature_names(pipeline, X_ref, numeric_features, categorical_features):
    """Final feature names, whichever pipeline architecture is in use."""
    steps = dict(pipeline.named_steps)
    if "preprocessor" in steps:                      # class-weighting pipeline
        return list(steps["preprocessor"].get_feature_names_out())
    if "onehot_after_smotenc" in steps:              # SMOTENC pipeline
        return R.smotenc_feature_names(numeric_features, categorical_features, X_ref)
    raise ValueError("Unrecognised pipeline architecture.")
def transform_features(pipeline, X):
    """Apply every step except the final estimator."""
    steps = list(pipeline.named_steps.items())
    Xt = X
    for name, step in steps[:-1]:
        if name == "smotenc":       # resampler: no-op at inference time
            continue
        Xt = step.transform(Xt)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return np.asarray(Xt, dtype=np.float32)
def group_of(encoded_name, categorical_features):
    """Map an encoded column name back to its original feature group."""
    name = str(encoded_name).split("__", 1)[-1]
    for cat in categorical_features:
        if name.startswith(cat + "_"):
            return cat
    return name
def stratified_sample(X, y, n, seed=C.RANDOM_STATE):
    """Sample n rows, keeping every positive case."""
    pos = y.index[y == 1]
    neg = y.index[y == 0]
    rng = np.random.default_rng(seed)
    keep = rng.choice(neg, size=min(max(n - len(pos), 1), len(neg)), replace=False)
    idx = np.concatenate([pos.values, keep])
    rng.shuffle(idx)
    return X.loc[idx], y.loc[idx]
def main() -> None:
    C.set_seeds()
    C.ensure_dirs()
    config = json.loads(C.FINAL_CONFIG_PATH.read_text())
    threshold = float(config["selected_threshold"])
    model_name = config["selected_model"]
    pipeline = joblib.load(C.FINAL_MODEL_PATH)
    model = pipeline.named_steps["model"]
    X_test = C.read_parquet_or_csv(C.SPLIT_PATHS["X_test"])

    y_test = C.read_parquet_or_csv(C.SPLIT_PATHS["y_test"])[C.TARGET_COLUMN]
    X_train = C.read_parquet_or_csv(C.SPLIT_PATHS["X_train"])
    categorical_features = [c for c in C.CATEGORICAL_FEATURES if c in X_test.columns]
    numeric_features = [c for c in X_test.columns if c not in categorical_features]
    feature_names = resolve_feature_names(pipeline, X_train,
                                          numeric_features, categorical_features)
    groups = [group_of(f, categorical_features) for f in feature_names]
    C.log_step(f"Explaining {model_name}: {len(feature_names)} encoded features "
               f"-> {len(set(groups))} groups")
    # ---------------------------------------------------- 1. Built-in importance
    if hasattr(model, "feature_importances_"):
        raw, kind = model.feature_importances_, "Gini / gain importance"
    elif hasattr(model, "coef_"):
        raw, kind = np.abs(np.ravel(model.coef_)), "Absolute logistic coefficient"
    else:
        raw, kind = np.zeros(len(feature_names)), "unavailable"
    builtin = pd.DataFrame({"encoded_feature": feature_names,
                            "feature_group": groups,
                            "importance": raw}).sort_values(
                                "importance", ascending=False)
    builtin["importance_type"] = kind
    C.save_table(builtin, "explainability_builtin_importance_encoded")
    bg = (builtin.groupby("feature_group")["importance"].sum()
          .sort_values(ascending=False).reset_index())
    bg["importance_percent"] = np.round(100 * bg["importance"] / bg["importance"].sum(), 4)
    C.save_table(bg, "explainability_builtin_importance_grouped")
    print(f"\nBuilt-in importance ({kind}) by group:")
    print(bg.head(10).to_string(index=False))
    # ---------------------------------------------------- 2. Permutation importance
    C.log_step("Computing permutation importance (PR-AUC)...")
    Xp, yp = stratified_sample(X_test, y_test, PERMUTATION_SAMPLE)
    perm = permutation_importance(pipeline, Xp, yp, scoring="average_precision",
                                  n_repeats=5, random_state=C.RANDOM_STATE, n_jobs=1)
    perm_df = pd.DataFrame({
        "feature": Xp.columns,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    perm_df["rank"] = range(1, len(perm_df) + 1)
    C.save_table(perm_df, "explainability_permutation_importance")
    print("\nPermutation importance (drop in PR-AUC when shuffled):")
    print(perm_df.head(10).to_string(index=False))
    # ---------------------------------------------------- 3. SHAP
    C.log_step("Computing SHAP values...")
    import shap
    Xs, ys = stratified_sample(X_test, y_test, SHAP_SAMPLE, seed=7)
    Xs_enc = transform_features(pipeline, Xs)
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        out = explainer.shap_values(Xs_enc, check_additivity=False)
        if isinstance(out, list):
            shap_pos = np.asarray(out[1] if len(out) > 1 else out[0])
        else:
            arr = np.asarray(out)
            shap_pos = arr[:, :, 1] if arr.ndim == 3 else arr
        base = explainer.expected_value
        base_value = float(np.ravel(base)[-1] if np.ndim(base) else base)
    else:
        bg_sample = shap.sample(Xs_enc, 100, random_state=C.RANDOM_STATE)
        explainer = shap.LinearExplainer(model, bg_sample)
        shap_pos = np.asarray(explainer.shap_values(Xs_enc))
        base_value = float(np.ravel(explainer.expected_value)[0])
    shap_enc = pd.DataFrame({
        "encoded_feature": feature_names, "feature_group": groups,
        "mean_abs_shap": np.abs(shap_pos).mean(axis=0),
        "mean_shap": shap_pos.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    C.save_table(shap_enc, "explainability_shap_importance_encoded")
    sg = (shap_enc.groupby("feature_group")["mean_abs_shap"].sum()
          .sort_values(ascending=False).reset_index())
    sg["contribution_percent"] = np.round(
        100 * sg["mean_abs_shap"] / sg["mean_abs_shap"].sum(), 4)
    sg["rank"] = range(1, len(sg) + 1)
    C.save_table(sg, "explainability_shap_importance_grouped")
    print("\nGlobal SHAP importance by group:")
    print(sg.head(10).to_string(index=False))
    # ---------------------------------------------------- 4. Method agreement
    agree = (bg[["feature_group", "importance_percent"]]
             .rename(columns={"importance_percent": "builtin_percent"})
             .merge(sg[["feature_group", "contribution_percent"]]
                    .rename(columns={"contribution_percent": "shap_percent"}),
                    on="feature_group", how="outer")
             .merge(perm_df[["feature", "permutation_importance_mean"]]
                    .rename(columns={"feature": "feature_group"}),
                    on="feature_group", how="outer"))
    for c, col in [("builtin_rank", "builtin_percent"), ("shap_rank", "shap_percent"),
                   ("permutation_rank", "permutation_importance_mean")]:

        agree[c] = agree[col].rank(ascending=False)
    C.save_table(agree.sort_values("shap_rank"), "explainability_method_agreement")
    # ---------------------------------------------------- 5. Local explanations
    proba = pipeline.predict_proba(Xs)[:, 1]
    pred = (proba >= threshold).astype(int)
    outcome = np.select(
        [(ys.values == 1) & (pred == 1), (ys.values == 0) & (pred == 1),
         (ys.values == 1) & (pred == 0)],
        ["True positive", "False positive", "False negative"], "True negative")
    rows = []
    for case in ["True positive", "False positive", "False negative"]:
        pos = np.where(outcome == case)[0]
        if not len(pos):
            continue
        for i in pos[np.argsort(-proba[pos])][:N_LOCAL_EXAMPLES]:
            contrib = pd.DataFrame({"feature_group": groups,
                                    "shap_value": shap_pos[i]})
            g = contrib.groupby("feature_group")["shap_value"].sum().reset_index()
            g["abs_value"] = g["shap_value"].abs()
            for _, r in g.nlargest(5, "abs_value").iterrows():
                rows.append({
                    "case_type": case, "row_index": int(Xs.index[i]),
                    "risk_score": round(float(proba[i]), 6),
                    "true_label": int(ys.values[i]), "predicted_label": int(pred[i]),
                    "base_value": round(base_value, 6),
                    "feature_group": r["feature_group"],
                    "shap_contribution": round(float(r["shap_value"]), 6),
                    "direction": "Increases risk" if r["shap_value"] > 0 else "Decreases risk",
                    "observed_value": str(Xs.iloc[i].get(r["feature_group"], "n/a")),
                })
    local_df = pd.DataFrame(rows)
    C.save_table(local_df, "explainability_local_shap_examples")
    # ---------------------------------------------------- 6. Figures
    top = sg.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(top["feature_group"], top["mean_abs_shap"], color="#1f4e79")
    ax.set_xlabel("Mean |SHAP value| (summed over encoded columns)")
    ax.set_title("Global SHAP feature-group importance")
    ax.grid(alpha=.3, axis="x")
    C.save_figure(fig, "fig09_shap_global_importance"); plt.close(fig)
    topb = bg.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(topb["feature_group"], topb["importance_percent"], color="#2e7d32")
    ax.set_xlabel(f"{kind} (%)")
    ax.set_title("Built-in feature-group importance")
    ax.grid(alpha=.3, axis="x")
    C.save_figure(fig, "fig10_builtin_importance"); plt.close(fig)
    topp = perm_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(topp["feature"], topp["permutation_importance_mean"],
            xerr=topp["permutation_importance_std"], color="#c00000")
    ax.set_xlabel("Mean decrease in PR-AUC when shuffled")
    ax.set_title("Permutation importance (test set)")
    ax.grid(alpha=.3, axis="x")
    C.save_figure(fig, "fig11_permutation_importance"); plt.close(fig)
    tp = local_df[local_df["case_type"] == "True positive"]
    if not tp.empty:
        case = tp[tp["row_index"] == tp.iloc[0]["row_index"]].sort_values("shap_contribution")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        cols = ["#c00000" if v > 0 else "#1f4e79" for v in case["shap_contribution"]]
        ax.barh(case["feature_group"], case["shap_contribution"], color=cols)
        ax.axvline(0, color="black", lw=.8)
        ax.set_xlabel("SHAP contribution to laundering risk score")
        ax.set_title(f"Local explanation - flagged transaction "
                     f"(risk {case.iloc[0]['risk_score']:.3f})")
        ax.grid(alpha=.3, axis="x")
        C.save_figure(fig, "fig12_local_shap_example"); plt.close(fig)
    # ---------------------------------------------------- 7. Caveats
    C.save_table(pd.DataFrame([
        {"caveat": "Synthetic data",
         "detail": "AMLNet is generated; learned patterns reflect generator rules "
                   "as much as real laundering behaviour."},
        {"caveat": "Category/type dominance",
         "detail": "Transaction category and type dominate importance, indicating "
                   "strong generator-side signal rather than subtle behavioural detection."},
        {"caveat": "Correlated features",
         "detail": "Engineered balance and amount features are correlated, so "
                   "importance is shared and single attributions should not be read alone."},
        {"caveat": "Explanations are not evidence",
         "detail": "SHAP shows what the model responded to, not that laundering occurred."},
        {"caveat": "Non-temporal split",
         "detail": "Stratified random split does not replicate chronological or "
                   "account-disjoint deployment conditions."},
    ]), "explainability_caveats")
    C.log_step("Step 08 complete.")
if __name__ == "__main__":
    main()
