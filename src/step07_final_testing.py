"""
Step 07: Final Test-Set Evaluation.
CRISP-DM: Evaluation.
Loads the model and threshold frozen by step 06 and applies them to the
untouched test set exactly once. No tuning, no threshold changes, no model
choices are made here - doing any of those would invalidate the test result.
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
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, precision_recall_curve)
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
def main() -> None:
    C.set_seeds()
    C.ensure_dirs()
    config = json.loads(C.FINAL_CONFIG_PATH.read_text())
    threshold = float(config["selected_threshold"])
    model_name = config["selected_model"]
    pipeline = joblib.load(C.FINAL_MODEL_PATH)
    C.log_step(f"Final model (auto-selected in step 06): {model_name} @ t={threshold}")
    X_test = C.read_parquet_or_csv(C.SPLIT_PATHS["X_test"])
    y_test = C.read_parquet_or_csv(C.SPLIT_PATHS["y_test"])[C.TARGET_COLUMN]
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    rec = C.evaluate_predictions(y_test, proba, threshold, model_name, "test")
    rec["imbalance_strategy"] = config["imbalance_strategy"]
    rec["tuned_hyperparameters"] = config.get("tuned_hyperparameters", "{}")
    C.save_table(pd.DataFrame([rec]), "final_test_metrics")
    print(pd.DataFrame([rec]).T.to_string(header=False))
    # Default vs optimised threshold, on test
    default_rec = C.evaluate_predictions(y_test, proba, 0.50, model_name, "test")
    C.save_table(pd.DataFrame([
        {**default_rec, "threshold_source": "Default 0.50"},
        {**{k: v for k, v in rec.items() if k in default_rec},
         "threshold_source": f"Optimised {threshold}"},
    ]), "test_default_vs_optimised_threshold")
    # Validation vs test generalisation check
    sel = pd.read_csv(C.TABLES_DIR / "final_model_and_threshold_selection.csv")
    gap = pd.DataFrame([{
        "metric": m,
        "validation": float(sel.loc[0, f"validation_{k}"]),
        "test": float(rec[v]),
        "difference": round(float(rec[v]) - float(sel.loc[0, f"validation_{k}"]), 6),
    } for m, k, v in [
        ("Precision", "precision", "precision"),
        ("Recall", "recall", "recall"),
        ("F1-score", "f1", "f1_score"),
        ("PR-AUC", "pr_auc", "pr_auc_average_precision"),
    ]])
    gap["interpretation"] = np.where(
        gap["difference"].abs() < 0.05,
        "Consistent with validation - no evidence of overfitting",
        "Noticeable gap - interpret with caution")
    C.save_table(gap, "validation_versus_test_generalisation")
    print("\n" + gap.to_string(index=False))
    # Classification report + confusion matrix
    report = classification_report(
        y_test, y_pred, target_names=["Normal (0)", "Laundering (1)"],
        output_dict=True, zero_division=0)
    C.save_table(pd.DataFrame(report).T.reset_index()
                 .rename(columns={"index": "class"}),
                 "final_test_classification_report")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    C.save_table(pd.DataFrame(
        cm, index=["Actual normal", "Actual laundering"],

        columns=["Predicted normal", "Predicted laundering"]).reset_index()
        .rename(columns={"index": ""}), "final_test_confusion_matrix")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Laundering"]).plot(
        ax=ax, cmap="Blues", values_format=",d", colorbar=False)
    ax.set_title(f"Test confusion matrix\n{model_name.replace('_',' ')} @ t={threshold}",
                 fontsize=10)
    C.save_figure(fig, "fig07_test_confusion_matrix"); plt.close(fig)
    prec, recl, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recl, prec, color="#1f4e79", lw=2)
    ax.axhline(y_test.mean(), ls="--", color="grey",
               label=f"No-skill ({y_test.mean():.4f})")
    ax.scatter([rec["recall"]], [rec["precision"]], color="#c00000", zorder=5,
               label=f"Operating point (t={threshold})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Test precision-recall curve")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=.3)
    C.save_figure(fig, "fig08_test_pr_curve"); plt.close(fig)
    # Reviewer workload framing
    C.save_table(pd.DataFrame([{
        "test_transactions": int(len(y_test)),
        "actual_laundering_cases": int(y_test.sum()),
        "alerts_raised": int(y_pred.sum()),
        "alert_rate_percent": round(100 * y_pred.mean(), 4),
        "alerts_per_10k_transactions": round(10000 * y_pred.mean(), 2),
        "true_positives_confirmed": rec["true_positives"],
        "false_positives_to_review": rec["false_positives"],
        "missed_cases_false_negatives": rec["false_negatives"],
        "precision_of_alert_queue": round(rec["precision"], 6),
        "interpretation": "Each alert prioritises a transaction for human "
                          "review; it is not a determination of laundering.",
    }]), "final_test_reviewer_workload")
    # Predictions for dashboard / explainability
    preds = X_test.copy()
    preds["true_label"] = y_test.values
    preds["risk_score"] = proba
    preds["predicted_label"] = y_pred
    preds["outcome"] = np.select(
        [(y_test.values == 1) & (y_pred == 1), (y_test.values == 0) & (y_pred == 1),
         (y_test.values == 1) & (y_pred == 0)],
        ["True positive", "False positive", "False negative"], "True negative")
    preds.to_parquet(C.DATA_PROCESSED_DIR / "test_predictions.parquet")
    C.save_table(preds["outcome"].value_counts().reset_index()
                 .rename(columns={"index": "outcome", "count": "n"}),
                 "final_test_outcome_counts")
    C.log_step("Step 07 complete. Test set has now been used once.")
if __name__ == "__main__":
    main()
