"""
Notebook/Step 03: Feature Engineering.
CRISP-DM: Data Preparation.
Creates safe engineered features from amount and balance fields only.
No feature is derived from typology, metadata, risk-score, fraud-label or
identifier columns.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
FEATURE_DOC = [
    ("balance_change_orig", "oldbalanceOrg, newbalanceOrig", "Numeric",
     "Captures originator balance movement after the transaction."),
    ("balance_change_minus_amount", "oldbalanceOrg, newbalanceOrig, amount", "Numeric",
     "Checks whether observed balance movement aligns with the transaction amount."),
    ("amount_to_oldbalance_ratio", "amount, oldbalanceOrg", "Numeric",
     "Measures transaction amount relative to the starting balance."),
    ("amount_to_newbalance_ratio", "amount, newbalanceOrig", "Numeric",
     "Measures transaction amount relative to the ending balance."),
    ("balance_change_to_oldbalance_ratio", "oldbalanceOrg, newbalanceOrig", "Numeric",
     "Measures balance movement relative to the starting balance."),
    ("is_zero_oldbalanceOrg", "oldbalanceOrg", "Binary indicator",
     "Flags transactions where the originator starts with zero balance."),
    ("is_zero_newbalanceOrig", "newbalanceOrig", "Binary indicator",
     "Flags transactions where the originator ends with zero balance."),
    ("log_amount", "amount", "Numeric log-transformed",
     "Reduces skewness in transaction amount."),
    ("log_oldbalanceOrg", "oldbalanceOrg", "Numeric log-transformed",
     "Reduces skewness in old originator balance."),
    ("log_newbalanceOrig", "newbalanceOrig", "Numeric log-transformed",
     "Reduces skewness in new originator balance."),
    ("log_abs_balance_change", "oldbalanceOrg, newbalanceOrig", "Numeric log-transformed",
     "Reduces skewness in absolute balance movement."),
]
def main() -> None:
    C.set_seeds()
    C.ensure_dirs()
    df_cleaned = C.read_parquet_or_csv(C.CLEANED_DATA_PATH)
    C.log_step(f"Loaded cleaned dataset: {df_cleaned.shape[0]:,} rows")
    # ---- Input safety check ------------------------------------------------
    forbidden = [c for c in (C.LEAKAGE_OR_PROXY_COLUMNS + C.IDENTIFIER_COLUMNS)
                 if c in df_cleaned.columns]
    if forbidden:
        raise ValueError(f"Forbidden columns present in input: {forbidden}")
    missing = [c for c in C.SAFE_CANDIDATE_FEATURES + [C.TARGET_COLUMN]
               if c not in df_cleaned.columns]
    if missing:
        raise ValueError(f"Missing expected cleaned columns: {missing}")
    C.save_table(pd.DataFrame([{
        "check": "Feature-engineering input safety check",
        "forbidden_columns_found": "none",
        "status": "PASSED",
    }]), "feature_engineering_input_safety_check")
    # ---- Engineer ----------------------------------------------------------
    df_eng = C.engineer_features(df_cleaned)
    # ---- Validate ----------------------------------------------------------
    miss = df_eng[C.ENGINEERED_FEATURES].isna().sum()
    missing_summary = pd.DataFrame({
        "engineered_feature": miss.index,
        "missing_count": miss.values,
        "missing_percentage": np.round(100 * miss.values / len(df_eng), 6),
    })
    inf_summary = pd.DataFrame([{
        "engineered_feature": c,
        "infinite_count": int(np.isinf(df_eng[c].astype("float64")).sum()),
    } for c in C.ENGINEERED_FEATURES])
    counts = df_eng[C.TARGET_COLUMN].value_counts().sort_index()
    target_after = pd.DataFrame({
        "target_value": counts.index.astype(int),
        "count": counts.values,
        "percentage": np.round(100 * counts.values / len(df_eng), 6),
    })

    validation = pd.DataFrame([{
        "rows_before_engineering": int(len(df_cleaned)),
        "rows_after_engineering": int(len(df_eng)),
        "columns_before_engineering": int(df_cleaned.shape[1]),
        "columns_after_engineering": int(df_eng.shape[1]),
        "engineered_features_added": len(C.ENGINEERED_FEATURES),
        "total_missing_in_engineered": int(miss.sum()),
        "total_infinite_in_engineered": int(inf_summary["infinite_count"].sum()),
    }])
    doc = pd.DataFrame(FEATURE_DOC, columns=["engineered_feature", "source_columns",
                                             "feature_type", "reason"])
    doc["leakage_safe"] = "Yes"
    saved = C.write_parquet(df_eng.reset_index(drop=True), C.ENGINEERED_DATA_PATH)
    C.save_table(doc, "feature_engineering_summary")
    C.save_table(validation, "feature_engineering_validation_summary")
    C.save_table(missing_summary, "engineered_feature_missing_summary")
    C.save_table(inf_summary, "engineered_feature_infinite_summary")
    C.save_table(target_after, "target_distribution_after_feature_engineering")
    C.log_step(f"Engineered dataset: {df_eng.shape[0]:,} rows x "
               f"{df_eng.shape[1]} cols -> {saved}")
    C.log_step("Step 03 complete.")
if __name__ == "__main__":
    main()
