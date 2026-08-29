"""
AMLNet project - shared configuration and helper utilities.
Project title:
    Explainable Machine Learning for Suspicious Transaction Risk Detection
    in Anti-Money Laundering
Dataset:
    AMLNet (Huda, S. et al., 2025), Zenodo record 16736515
    https://doi.org/10.5281/zenodo.16736515
    File: AMLNet_August 2025.csv  -> stored as data/raw/AMLNet.csv
This module centralises paths, constants, leakage-control column groups,
memory-efficient loading and the evaluation helpers that are reused by
notebooks 00 to 10. Keeping them in one place avoids the copy-paste drift
that existed in the earlier version of the code.
"""
from __future__ import annotations
import json
import os
import platform
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Fix all seeds that affect this project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the AMLNet_Project root whether the code is executed from
    the project root, from notebooks/ or from src/.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if candidate.name == "AMLNet_Project":
            return candidate
        if (candidate / "data" / "raw").exists() and (candidate / "src").exists():
            return candidate
    # Fall back to the parent of this file's directory (src/ -> project root)
    return Path(__file__).resolve().parent.parent
PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORT_EVIDENCE_DIR = OUTPUTS_DIR / "report_evidence"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
REQUIRED_DIRS = [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    NOTEBOOKS_DIR,
    SRC_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    REPORT_EVIDENCE_DIR,
    DASHBOARD_DIR,
]
def ensure_dirs() -> None:

    for folder in REQUIRED_DIRS:
        folder.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------------------------
# Dataset constants
# --------------------------------------------------------------------------
DATASET_FILENAME = "AMLNet.csv"
RAW_DATA_PATH = DATA_RAW_DIR / DATASET_FILENAME
DATASET_DOI = "10.5281/zenodo.16736515"
DATASET_MD5 = "7668fc7d74c787e07546ce85c6f790b9"
TARGET_COLUMN = "isMoneyLaundering"
EXPECTED_COLUMNS = [
    "step", "type", "amount", "category", "nameOrig", "nameDest",
    "oldbalanceOrg", "newbalanceOrig", "isFraud", "isMoneyLaundering",
    "laundering_typology", "metadata", "fraud_probability",
    "hour", "day_of_week", "day_of_month", "month",
]
# `metadata` is a very large embedded JSON/py-dict string column (~600 MB of the
# 691 MB file). It is leakage-prone AND memory-hostile, so it is never loaded
# into the modelling pipeline. Notebook 01 samples it separately for auditing.
HEAVY_COLUMNS = ["metadata"]
LEAKAGE_OR_PROXY_COLUMNS = [
    "laundering_typology",   # typology label -> directly reveals the target
    "metadata",              # embedded generator risk scores / typology traces
    "fraud_probability",     # generated risk-score-like field
    "isFraud",               # related generated label
]
IDENTIFIER_COLUMNS = ["nameOrig", "nameDest"]
# `step` is deliberately NOT modelled.
# Justification: "The `step` column was removed because it represents the
# simulation time index rather than a meaningful transaction attribute for
# suspicious transaction risk prediction." It is still audited in step 01
# (temporal coverage) but never enters the model.
SAFE_CANDIDATE_FEATURES = [
    "type", "amount", "category",
    "oldbalanceOrg", "newbalanceOrig",
    "hour", "day_of_week", "day_of_month", "month",
]
CATEGORICAL_FEATURES = ["type", "category"]
ENGINEERED_FEATURES = [
    "balance_change_orig",
    "balance_change_minus_amount",
    "amount_to_oldbalance_ratio",
    "amount_to_newbalance_ratio",
    "balance_change_to_oldbalance_ratio",
    "is_zero_oldbalanceOrg",
    "is_zero_newbalanceOrig",
    "log_amount",
    "log_oldbalanceOrg",
    "log_newbalanceOrig",
    "log_abs_balance_change",
]
# Memory-efficient dtypes (the machine running this only needs ~250 MB for the
# full 1.09 M row modelling frame instead of ~1.4 GB).
READ_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "category": "category",
    "nameOrig": "string",
    "nameDest": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "isFraud": "int8",
    "isMoneyLaundering": "int8",
    "laundering_typology": "category",
    "fraud_probability": "float32",
    "hour": "int8",
    "day_of_week": "int8",
    "day_of_month": "int8",
    "month": "int8",
}
# --------------------------------------------------------------------------
# Processed data / artefact paths
# --------------------------------------------------------------------------
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_amlnet.parquet"
ENGINEERED_DATA_PATH = DATA_PROCESSED_DIR / "engineered_amlnet.parquet"
SPLIT_PATHS = {
    "X_train": DATA_PROCESSED_DIR / "X_train.parquet",
    "X_val": DATA_PROCESSED_DIR / "X_val.parquet",
    "X_test": DATA_PROCESSED_DIR / "X_test.parquet",
    "y_train": DATA_PROCESSED_DIR / "y_train.parquet",
    "y_val": DATA_PROCESSED_DIR / "y_val.parquet",
    "y_test": DATA_PROCESSED_DIR / "y_test.parquet",
}
FINAL_MODEL_PATH = MODELS_DIR / "final_model_pipeline.joblib"
FINAL_CONFIG_PATH = MODELS_DIR / "final_model_config.json"
# --------------------------------------------------------------------------

# Loading helpers
# --------------------------------------------------------------------------
def load_raw(usecols: list[str] | None = None,
             nrows: int | None = None,
             include_metadata: bool = False) -> pd.DataFrame:
    """
    Load the raw AMLNet CSV with memory-safe dtypes.
    The `metadata` column is excluded by default because it holds a very large
    Python-dict string per row and is a leakage-prone field.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"AMLNet dataset not found at: {RAW_DATA_PATH}\n"
            "Download 'AMLNet_August 2025.csv' from "
            "https://zenodo.org/records/16736515 , rename it to AMLNet.csv "
            "and place it in data/raw/."
        )
    if usecols is None:
        usecols = [c for c in EXPECTED_COLUMNS
                   if include_metadata or c not in HEAVY_COLUMNS]
    dtypes = {k: v for k, v in READ_DTYPES.items() if k in usecols}
    return pd.read_csv(
        RAW_DATA_PATH,
        usecols=usecols,
        dtype=dtypes,
        nrows=nrows,
        low_memory=False,
    )
def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Save an evidence table to outputs/tables and return its path."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, index=index)
    return path
def save_figure(fig, name: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure to outputs/figures and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
def read_parquet_or_csv(path: Path) -> pd.DataFrame:
    """Read a processed artefact, falling back to a .csv twin if needed."""
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    csv_twin = path.with_suffix(".csv")
    if csv_twin.exists():
        return pd.read_csv(csv_twin, low_memory=False)
    raise FileNotFoundError(
        f"Required processed file not found: {path}\n"
        "Please run the earlier notebooks/scripts in order."
    )
def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a processed artefact as parquet (falls back to csv)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=True)
        return path
    except Exception:                                    # pragma: no cover
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=True)
        return csv_path
# --------------------------------------------------------------------------
# Feature engineering (single definition, reused by notebook 03 and dashboard)
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the safe engineered transaction-level features.
    Only `amount`, `oldbalanceOrg` and `newbalanceOrig` are used, so no
    leakage-prone, typology, risk-score or identifier field can enter the
    model through the engineered features.
    """
    out = df.copy()
    amount = out["amount"].astype("float32")
    old_bal = out["oldbalanceOrg"].astype("float32")
    new_bal = out["newbalanceOrig"].astype("float32")
    out["balance_change_orig"] = (old_bal - new_bal).astype("float32")
    out["balance_change_minus_amount"] = (
        out["balance_change_orig"] - amount
    ).astype("float32")
    out["amount_to_oldbalance_ratio"] = np.where(
        old_bal > 0, amount / np.where(old_bal > 0, old_bal, 1), 0
    ).astype("float32")

    out["amount_to_newbalance_ratio"] = np.where(
        new_bal > 0, amount / np.where(new_bal > 0, new_bal, 1), 0
    ).astype("float32")
    out["balance_change_to_oldbalance_ratio"] = np.where(
        old_bal > 0,
        out["balance_change_orig"] / np.where(old_bal > 0, old_bal, 1),
        0,
    ).astype("float32")
    out["is_zero_oldbalanceOrg"] = (old_bal == 0).astype("int8")
    out["is_zero_newbalanceOrig"] = (new_bal == 0).astype("int8")
    out["log_amount"] = np.log1p(amount.clip(lower=0)).astype("float32")
    out["log_oldbalanceOrg"] = np.log1p(old_bal.clip(lower=0)).astype("float32")
    out["log_newbalanceOrig"] = np.log1p(new_bal.clip(lower=0)).astype("float32")
    out["log_abs_balance_change"] = np.log1p(
        out["balance_change_orig"].abs()
    ).astype("float32")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
# --------------------------------------------------------------------------
# Modelling helpers
# --------------------------------------------------------------------------
def make_onehot_encoder(sparse: bool = True):
    """OneHotEncoder that works across scikit-learn versions."""
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:                                    # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)
def build_preprocessor(numeric_features: list[str],
                       categorical_features: list[str],
                       scale_numeric: bool = True,
                       sparse: bool = True):
    """
    ColumnTransformer used inside every model pipeline.
    Fitting happens inside the pipeline, so it is always fitted on training
    data only - this is a core leakage-control requirement of the project.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ("encoder", make_onehot_encoder(sparse=sparse)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse else 0.0,
    )
def evaluate_predictions(y_true, y_proba, threshold: float = 0.5,
                         model_name: str = "model",
                         split_name: str = "validation") -> dict:
    """
    Imbalance-aware evaluation record for one model at one threshold.
    Accuracy is reported for completeness only; PR-AUC, recall, precision and
    the confusion-matrix counts are the metrics used for decisions because the
    positive class is ~0.16 % of the data.
    """
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos = int(y_true.sum())
    n_alerts = int(y_pred.sum())
    return {
        "model_name": model_name,
        "split": split_name,
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),

        "pr_auc_average_precision": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba) if n_pos else np.nan,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "positives_in_split": n_pos,
        "predicted_positive_alerts": n_alerts,
        "alert_rate_percent": round(100 * n_alerts / len(y_true), 6),
    }
def environment_summary() -> pd.DataFrame:
    """Package/environment table for the reproducibility appendix."""
    import importlib.metadata as md
    packages = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn",
                "shap", "xgboost", "lightgbm", "imbalanced-learn", "joblib",
                "pyarrow", "streamlit"]
    rows = [{"item": "python_version", "value": sys.version.split()[0]},
            {"item": "platform", "value": platform.platform()},
            {"item": "random_state", "value": str(RANDOM_STATE)}]
    for pkg in packages:
        try:
            rows.append({"item": pkg, "value": md.version(pkg)})
        except Exception:
            rows.append({"item": pkg, "value": "not installed"})
    return pd.DataFrame(rows)
def log_step(message: str) -> None:
    print(f"[AMLNet] {message}", flush=True)
__all__ = [n for n in dir() if not n.startswith("_")]
