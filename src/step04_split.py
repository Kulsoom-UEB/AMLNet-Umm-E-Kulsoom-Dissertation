"""
Notebook/Step 04: Preprocessing Design and Train-Validation-Test Split.
CRISP-DM: Data Preparation.
Separates X and y, defines numeric/categorical groups and creates a stratified
70/15/15 split. Validation and test sets are never resampled or transformed
outside a training-fitted pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
def main() -> None:
    C.set_seeds()
    C.ensure_dirs()
    df = C.read_parquet_or_csv(C.ENGINEERED_DATA_PATH)
    C.log_step(f"Loaded engineered dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")
    y = df[C.TARGET_COLUMN].astype("int8")
    X = df.drop(columns=[C.TARGET_COLUMN])
    categorical_features = [c for c in C.CATEGORICAL_FEATURES if c in X.columns]
    numeric_features = [c for c in X.columns if c not in categorical_features]
    # Cast categoricals to plain object/str so downstream encoders and SMOTENC
    # behave consistently across pandas/sklearn versions.
    for c in categorical_features:
        X[c] = X[c].astype(str)
    C.save_table(pd.DataFrame({
        "feature_name": numeric_features + categorical_features,
        "feature_group": (["Numeric"] * len(numeric_features)
                          + ["Categorical"] * len(categorical_features)),
    }), "final_feature_group_table")
    # ---- Stratified 70 / 15 / 15 -------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=C.RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp,
        random_state=C.RANDOM_STATE)
    del X_temp, y_temp
    splits = {"train": (X_train, y_train), "validation": (X_val, y_val),
              "test": (X_test, y_test)}
    rows = []
    for name, (Xs, ys) in splits.items():
        pos = int(ys.sum())
        rows.append({
            "split": name,
            "rows": int(len(ys)),
            "share_of_dataset_percent": round(100 * len(ys) / len(y), 4),
            "normal_transactions": int(len(ys) - pos),
            "laundering_transactions": pos,
            "positive_rate_percent": round(100 * pos / len(ys), 6),
        })
    dist = pd.DataFrame(rows)
    C.save_table(dist, "train_validation_test_class_distribution")
    print(dist.to_string(index=False))
    # ---- Integrity checks ---------------------------------------------------
    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    integrity = pd.DataFrame([{
        "total_rows_recombined": len(train_idx) + len(val_idx) + len(test_idx),
        "total_rows_original": len(X),
        "row_count_matches": (len(train_idx) + len(val_idx) + len(test_idx)) == len(X),
        "train_val_overlap": len(train_idx & val_idx),
        "train_test_overlap": len(train_idx & test_idx),
        "val_test_overlap": len(val_idx & test_idx),
        "no_overlap_confirmed": not (train_idx & val_idx or train_idx & test_idx
                                     or val_idx & test_idx),
        "all_splits_contain_positives": all(r["laundering_transactions"] > 0 for r in rows),
        "stratification_max_positive_rate_deviation_pp": round(
            float(np.max(np.abs(dist["positive_rate_percent"]
                                - 100 * y.mean()))), 6),
    }])
    C.save_table(integrity, "split_integrity_summary")
    if not bool(integrity.loc[0, "no_overlap_confirmed"]):
        raise AssertionError("Split overlap detected.")

    C.save_table(pd.DataFrame([
        {"design_item": "Numeric preprocessing",
         "decision": "SimpleImputer(median) + StandardScaler",
         "leakage_control": "Fitted inside pipeline on training folds only"},
        {"design_item": "Categorical preprocessing",
         "decision": "OneHotEncoder(handle_unknown='ignore')",
         "leakage_control": "Fitted inside pipeline on training folds only"},
        {"design_item": "Imbalance handling",
         "decision": "class_weight / scale_pos_weight, plus SMOTENC experiments",
         "leakage_control": "Resampling applied to training data only"},
        {"design_item": "Validation set use",
         "decision": "Model comparison and threshold selection",
         "leakage_control": "Never resampled, never used for fitting"},
        {"design_item": "Test set use",
         "decision": "Single final evaluation of the selected model",
         "leakage_control": "Untouched until Step 08"},
    ]), "preprocessing_design_summary")
    # ---- Persist ------------------------------------------------------------
    for key, obj in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        C.write_parquet(obj, C.SPLIT_PATHS[key])
    for key, obj in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        C.write_parquet(obj.to_frame(name=C.TARGET_COLUMN), C.SPLIT_PATHS[key])
    C.log_step("Splits saved to data/processed/. Step 04 complete.")
if __name__ == "__main__":
    main()
