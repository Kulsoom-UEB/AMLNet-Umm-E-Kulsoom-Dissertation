"""
Notebook/Step 02: Data Preparation, Cleaning and Leakage Control.
CRISP-DM: Data Preparation.
Applies the leakage-control decisions, removes identifier and label-like
fields, handles missing values and duplicates, and writes a defensible
modelling dataset. No model is trained here.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
FEATURE_DECISIONS = [
    ("isMoneyLaundering", "Target", "No",
     "Binary AML target label to be predicted."),
    ("laundering_typology", "Exclude", "No",
     "Typology label (structuring/layering/integration) directly reveals the target."),
    ("metadata", "Exclude", "No",
     "Embedded generator dictionary containing risk_score, layering and structuring traces."),
    ("fraud_probability", "Exclude", "No",
     "Generated risk-score-like field produced by the data generator."),
    ("isFraud", "Exclude", "No",
     "Related generated fraud label; acts as a proxy for the AML target."),
    ("nameOrig", "Exclude", "No",
     "High-cardinality originator identifier; encourages account memorisation."),
    ("nameDest", "Exclude", "No",
     "High-cardinality destination identifier; encourages account memorisation."),
    ("step", "Exclude", "No",
     "The `step` column was removed because it represents the simulation time "
     "index rather than a meaningful transaction attribute for suspicious "
     "transaction risk prediction."),
    ("type", "Keep", "Yes", "Payment method; behavioural categorical feature."),
    ("amount", "Keep", "Yes", "Transaction amount; core behavioural feature."),
    ("category", "Keep", "Yes", "Transaction category; behavioural categorical feature."),
    ("oldbalanceOrg", "Keep", "Yes", "Originator balance before the transaction."),
    ("newbalanceOrig", "Keep", "Yes", "Originator balance after the transaction."),
    ("hour", "Keep", "Yes", "Hour of transaction; temporal behaviour."),
    ("day_of_week", "Keep", "Yes", "Day of week; temporal behaviour."),
    ("day_of_month", "Keep", "Yes", "Day of month; temporal behaviour."),
    ("month", "Keep", "Yes", "Month; temporal behaviour."),
]
def main() -> None:
    C.set_seeds()
    C.ensure_dirs()
    C.log_step("Loading raw AMLNet for preparation...")
    df_raw = C.load_raw()
    rows_raw = len(df_raw)
    # ---- Validate schema ---------------------------------------------------
    header = pd.read_csv(C.RAW_DATA_PATH, nrows=0).columns.tolist()
    missing = [c for c in C.EXPECTED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Expected columns missing: {missing}")
    if C.TARGET_COLUMN not in df_raw.columns:
        raise ValueError(f"Target column '{C.TARGET_COLUMN}' not found.")
    decisions = pd.DataFrame(FEATURE_DECISIONS,
                             columns=["column_name", "decision",
                                      "included_in_model", "reason"])
    C.save_table(decisions, "feature_decision_table")
    group_summary = pd.DataFrame({
        "group": (["Target"]
                  + ["Leakage/proxy - exclude"] * len(C.LEAKAGE_OR_PROXY_COLUMNS)
                  + ["Identifier - exclude"] * len(C.IDENTIFIER_COLUMNS)
                  + ["Safe candidate feature"] * len(C.SAFE_CANDIDATE_FEATURES)),
        "column_name": ([C.TARGET_COLUMN] + C.LEAKAGE_OR_PROXY_COLUMNS
                        + C.IDENTIFIER_COLUMNS + C.SAFE_CANDIDATE_FEATURES),
    })
    C.save_table(group_summary, "initial_column_group_summary")
    # ---- Apply leakage control --------------------------------------------
    keep_columns = C.SAFE_CANDIDATE_FEATURES + [C.TARGET_COLUMN]
    df = df_raw[keep_columns].copy()
    del df_raw
    # ---- Missing target ----------------------------------------------------
    before = len(df)
    df = df.dropna(subset=[C.TARGET_COLUMN])
    removed_missing_target = before - len(df)
    df[C.TARGET_COLUMN] = df[C.TARGET_COLUMN].astype("int8")
    # ---- Missing feature values -------------------------------------------

    missing_before = df[C.SAFE_CANDIDATE_FEATURES].isna().sum()
    # Numeric NaNs are left in place and imputed inside the model pipeline
    # (median imputer fitted on training data only) to avoid leakage.
    missing_report = pd.DataFrame({
        "column": missing_before.index,
        "missing_count": missing_before.values,
        "handling": ["Median imputation inside training-fitted pipeline"
                     if pd.api.types.is_numeric_dtype(df[c]) else
                     "OneHotEncoder(handle_unknown='ignore') inside pipeline"
                     for c in missing_before.index],
    })
    C.save_table(missing_report, "missing_value_handling_summary")
    # ---- Duplicates --------------------------------------------------------
    before_dup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_dup - len(df)
    # ---- Save --------------------------------------------------------------
    saved = C.write_parquet(df.reset_index(drop=True), C.CLEANED_DATA_PATH)
    counts = df[C.TARGET_COLUMN].value_counts().sort_index()
    cleaning_summary = pd.DataFrame([{
        "rows_in_raw_dataset": rows_raw,
        "rows_removed_missing_target": int(removed_missing_target),
        "duplicate_rows_removed": int(duplicates_removed),
        "rows_in_cleaned_dataset": int(len(df)),
        "columns_in_cleaned_dataset": int(df.shape[1]),
        "columns_excluded_for_leakage": len(C.LEAKAGE_OR_PROXY_COLUMNS),
        "columns_excluded_as_identifiers": len(C.IDENTIFIER_COLUMNS),
        "normal_transactions": int(counts.get(0, 0)),
        "laundering_transactions": int(counts.get(1, 0)),
        "positive_rate_percent": round(100 * counts.get(1, 0) / len(df), 6),
        "cleaned_dataset_path": str(saved),
    }])
    C.save_table(cleaning_summary, "data_cleaning_summary")
    # ---- Leakage assertion -------------------------------------------------
    forbidden_present = [c for c in
                         (C.LEAKAGE_OR_PROXY_COLUMNS + C.IDENTIFIER_COLUMNS)
                         if c in df.columns]
    if forbidden_present:
        raise AssertionError(f"Leakage columns survived cleaning: {forbidden_present}")
    C.save_table(pd.DataFrame([{
        "check": "No leakage/proxy/identifier column present in cleaned dataset",
        "status": "PASSED",
        "columns_checked": ", ".join(C.LEAKAGE_OR_PROXY_COLUMNS + C.IDENTIFIER_COLUMNS),
    }]), "leakage_control_assertion")
    C.log_step(f"Cleaned dataset: {len(df):,} rows x {df.shape[1]} cols -> {saved}")
    C.log_step(f"Duplicates removed: {duplicates_removed:,}")
    C.log_step("Step 02 complete.")
if __name__ == "__main__":
    main()
