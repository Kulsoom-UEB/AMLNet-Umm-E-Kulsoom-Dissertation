"""
Notebook/Step 01: Data Understanding and Quality Audit.
CRISP-DM: Data Understanding.
Audits shape, dtypes, missingness, duplicates, target balance, categorical
levels and temporal coverage of the raw AMLNet dataset. No cleaning and no
modelling happens here.
Memory note: the `metadata` column holds a large Python-dict string per row
(~600 MB of the 691 MB file). It is loaded only as a small sample for auditing
and is never brought into the modelling frame.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
def main() -> None:
    C.set_seeds()
    C.ensure_dirs()
    C.log_step("Loading raw AMLNet (metadata column excluded for memory safety)...")
    df = C.load_raw()
    C.log_step(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns "
               f"({df.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory)")
    # ---- 1. Shape and column inventory -----------------------------------
    header = pd.read_csv(C.RAW_DATA_PATH, nrows=0).columns.tolist()
    missing_expected = [c for c in C.EXPECTED_COLUMNS if c not in header]
    unexpected = [c for c in header if c not in C.EXPECTED_COLUMNS]
    if missing_expected:
        raise ValueError(f"Expected columns missing from the CSV: {missing_expected}")
    shape_summary = pd.DataFrame([{
        "rows": int(df.shape[0]),
        "columns_in_file": len(header),
        "columns_loaded_for_audit": int(df.shape[1]),
        "columns_excluded_for_memory": ", ".join(C.HEAVY_COLUMNS),
        "unexpected_columns": ", ".join(unexpected) if unexpected else "none",
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }])
    C.save_table(shape_summary, "dataset_shape_summary")
    # ---- 2. Column profile ------------------------------------------------
    profile = []
    for col in df.columns:
        s = df[col]
        rec = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_percent": round(100 * s.isna().mean(), 6),
            "unique_values": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            rec.update({
                "min": float(s.min()), "max": float(s.max()),
                "mean": float(s.mean()), "median": float(s.median()),
                "std": float(s.std()),
            })
        profile.append(rec)
    profile_df = pd.DataFrame(profile)
    C.save_table(profile_df, "column_profile_summary")
    # ---- 3. Duplicates ----------------------------------------------------
    behavioural_cols = [c for c in df.columns if c != "step"]
    dup_summary = pd.DataFrame([{
        "exact_duplicate_rows_all_loaded_columns": int(df.duplicated().sum()),
        "duplicate_rows_ignoring_step": int(df.duplicated(subset=behavioural_cols).sum()),
        "note": "metadata column excluded from duplicate check (not loaded).",
    }])
    C.save_table(dup_summary, "duplicate_check_summary")
    # ---- 4. Target balance -------------------------------------------------
    counts = df[C.TARGET_COLUMN].value_counts().sort_index()
    target_dist = pd.DataFrame({
        "target_value": counts.index.astype(int),
        "label": ["Normal (0)", "Money laundering (1)"][: len(counts)],
        "count": counts.values,

        "percentage": np.round(100 * counts.values / len(df), 6),
    })
    imbalance_ratio = counts.get(0, 0) / max(counts.get(1, 1), 1)
    target_dist["imbalance_ratio_negative_to_positive"] = round(imbalance_ratio, 2)
    C.save_table(target_dist, "target_class_distribution")
    C.log_step(f"Target balance: {counts.get(1, 0):,} positives "
               f"({100 * counts.get(1, 0) / len(df):.4f} %), "
               f"imbalance ratio 1:{imbalance_ratio:,.0f}")
    # ---- 5. Typology and label relationships (evidence for leakage control)
    typology = (df.groupby("laundering_typology", observed=True)[C.TARGET_COLUMN]
                  .agg(["count", "sum", "mean"])
                  .rename(columns={"count": "transactions",
                                   "sum": "laundering_positive",
                                   "mean": "positive_rate"})
                  .reset_index())
    C.save_table(typology, "laundering_typology_vs_target")
    crosstab = pd.crosstab(df["isFraud"], df[C.TARGET_COLUMN]).reset_index()
    C.save_table(crosstab, "isfraud_vs_target_crosstab")
    fp = df["fraud_probability"]
    fraud_prob = pd.DataFrame([{
        "missing_count": int(fp.isna().sum()),
        "missing_percent": round(100 * fp.isna().mean(), 4),
        "mean_when_target_0": float(fp[df[C.TARGET_COLUMN] == 0].mean()),
        "mean_when_target_1": float(fp[df[C.TARGET_COLUMN] == 1].mean()),
        "leakage_note": "Generated risk score, strongly separates classes -> excluded.",
    }])
    C.save_table(fraud_prob, "fraud_probability_leakage_evidence")
    # ---- 6. Categorical levels --------------------------------------------
    cat_rows = []
    for col in ["type", "category"]:
        vc = df[col].value_counts()
        pos = df.groupby(col, observed=True)[C.TARGET_COLUMN].mean()
        for level, n in vc.items():
            cat_rows.append({
                "column": col, "level": level, "count": int(n),
                "percent_of_rows": round(100 * n / len(df), 4),
                "laundering_positive_rate": round(float(pos.get(level, 0)), 6),
            })
    C.save_table(pd.DataFrame(cat_rows), "categorical_level_summary")
    # ---- 7. Identifier cardinality ----------------------------------------
    ident = pd.DataFrame([{
        "column": c,
        "unique_values": int(df[c].nunique()),
        "cardinality_ratio": round(df[c].nunique() / len(df), 6),
        "decision": "Exclude - high-cardinality identifier, memorisation risk",
    } for c in C.IDENTIFIER_COLUMNS])
    C.save_table(ident, "identifier_cardinality_summary")
    # ---- 8. Temporal coverage ---------------------------------------------
    temporal = pd.DataFrame([{
        "step_min": int(df["step"].min()), "step_max": int(df["step"].max()),
        "unique_steps": int(df["step"].nunique()),
        "hour_min": int(df["hour"].min()), "hour_max": int(df["hour"].max()),
        "months_present": ", ".join(map(str, sorted(df["month"].unique()))),
        "day_of_week_present": ", ".join(map(str, sorted(df["day_of_week"].unique()))),
    }])
    C.save_table(temporal, "temporal_coverage_summary")
    # ---- 9. Amount distribution by class ----------------------------------
    amt = (df.groupby(C.TARGET_COLUMN)["amount"]
             .describe(percentiles=[.25, .5, .75, .95, .99]).reset_index())
    C.save_table(amt, "amount_distribution_by_class")
    # ---- 10. Metadata sample audit ----------------------------------------
    meta_sample = pd.read_csv(C.RAW_DATA_PATH, usecols=["metadata"], nrows=200)
    meta_audit = pd.DataFrame([{
        "sampled_rows": len(meta_sample),
        "mean_characters_per_row": int(meta_sample["metadata"].str.len().mean()),
        "max_characters_per_row": int(meta_sample["metadata"].str.len().max()),
        "contains_risk_score_field": bool(
            meta_sample["metadata"].str.contains("risk_score").any()),
        "contains_layering_field": bool(
            meta_sample["metadata"].str.contains("layering").any()),
        "decision": "Exclude - embeds generator risk indicators and typology traces.",
    }])
    C.save_table(meta_audit, "metadata_column_audit")
    # ---- Figures -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Normal (0)", "Laundering (1)"], counts.values,
           color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("Number of transactions (log scale)")
    ax.set_title("AMLNet target class distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    C.save_figure(fig, "fig01_target_class_distribution")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, colour in [(0, "#4C72B0"), (1, "#C44E52")]:
        vals = np.log1p(df.loc[df[C.TARGET_COLUMN] == label, "amount"])
        ax.hist(vals, bins=60, alpha=0.6, density=True,

                label=f"class {label}", color=colour)
    ax.set_xlabel("log1p(amount)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount distribution by class")
    ax.legend()
    C.save_figure(fig, "fig02_amount_distribution_by_class")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    rate = df.groupby("category", observed=True)[C.TARGET_COLUMN].mean().sort_values()
    ax.barh(rate.index.astype(str), 100 * rate.values, color="#55A868")
    ax.set_xlabel("Laundering-positive rate (%)")
    ax.set_title("Laundering rate by transaction category")
    C.save_figure(fig, "fig03_laundering_rate_by_category")
    plt.close(fig)
    C.log_step("Step 01 complete: data understanding and quality audit written.")
if __name__ == "__main__":
    main()
