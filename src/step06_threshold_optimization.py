"""
Step 06: Automatic Best-Model Selection and Threshold Optimisation.
CRISP-DM: Evaluation.
Nothing here is hardcoded. The best model is discovered from the validation
metrics table, and the operating threshold is discovered by sweeping the
validation probabilities. Both are then frozen to models/final_model_config.json
so that steps 07-08 cannot silently disagree about which model or threshold is
in use.
Selection rationale
-------------------
PR-AUC is the primary criterion because it is threshold-independent and
appropriate for a 0.16 % positive class. F1 at the optimised threshold breaks
ties, and reviewer alert volume breaks any remaining tie: in AML a model that
floods analysts with false positives is not operationally usable even when its
ranking quality is high.
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
from sklearn.metrics import precision_recall_curve, roc_curve
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
def sweep(y_true, proba, model_name):
    rows = [C.evaluate_predictions(y_true, proba, float(t), model_name, "validation")
            for t in THRESHOLD_GRID]
    df = pd.DataFrame(rows)
    df["alerts_per_10k_transactions"] = np.round(
        10000 * df["predicted_positive_alerts"] / len(y_true), 2)
    return df
def main() -> None:
    C.set_seeds()
    C.ensure_dirs()
    metrics = pd.read_csv(C.TABLES_DIR / "validation_metrics_all_models.csv")
    candidates = metrics[
        metrics["model_name"] != "Majority_Class_Benchmark"].copy()
    X_val = C.read_parquet_or_csv(C.SPLIT_PATHS["X_val"])
    y_val = C.read_parquet_or_csv(C.SPLIT_PATHS["y_val"])[C.TARGET_COLUMN]
    # ---- Optimise the threshold for EVERY candidate ------------------------
    all_sweeps, per_model_best = [], []
    for name in candidates["model_name"]:
        pipe = joblib.load(C.MODELS_DIR / f"{name}_pipeline.joblib")
        proba = pipe.predict_proba(X_val)[:, 1]
        s = sweep(y_val, proba, name)
        s["model_name"] = name
        all_sweeps.append(s)
        best = s.loc[s["f1_score"].idxmax()]
        strategy = candidates.loc[
            candidates["model_name"] == name, "imbalance_strategy"].iloc[0]
        per_model_best.append({
            "model_name": name,
            "imbalance_strategy": strategy,
            "optimised_threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1_score": float(best["f1_score"]),
            "pr_auc_average_precision": float(best["pr_auc_average_precision"]),
            "roc_auc": float(best["roc_auc"]),
            "true_positives": int(best["true_positives"]),
            "false_positives": int(best["false_positives"]),
            "false_negatives": int(best["false_negatives"]),
            "predicted_positive_alerts": int(best["predicted_positive_alerts"]),
        })
        C.log_step(f"{name}: optimal t={best['threshold']:.2f} "
                   f"F1={best['f1_score']:.4f} PR-AUC={best['pr_auc_average_precision']:.4f}")
    C.save_table(pd.concat(all_sweeps, ignore_index=True),
                 "threshold_sweep_all_models")

    best_df = pd.DataFrame(per_model_best)
    # ---- Automatic selection: PR-AUC, then F1, then fewest alerts ----------
    best_df["rank_pr_auc"] = best_df["pr_auc_average_precision"].rank(
        ascending=False, method="min")
    best_df["rank_f1"] = best_df["f1_score"].rank(ascending=False, method="min")
    best_df["rank_alerts"] = best_df["predicted_positive_alerts"].rank(
        ascending=True, method="min")
    best_df = best_df.sort_values(
        ["rank_pr_auc", "rank_f1", "rank_alerts"]).reset_index(drop=True)
    best_df["selected"] = [True] + [False] * (len(best_df) - 1)
    C.save_table(best_df, "model_selection_ranking")
    win = best_df.iloc[0]
    selected_model = str(win["model_name"])
    selected_threshold = float(win["optimised_threshold"])
    C.log_step(f"SELECTED (automatic): {selected_model} @ threshold {selected_threshold}")
    print("\nAutomatic model selection (validation, threshold-optimised):")
    print(best_df[["model_name", "imbalance_strategy", "optimised_threshold",
                   "precision", "recall", "f1_score",
                   "pr_auc_average_precision", "false_positives",
                   "selected"]].to_string(index=False))
    # ---- Evidence for the winner -------------------------------------------
    pipe = joblib.load(C.MODELS_DIR / f"{selected_model}_pipeline.joblib")
    proba = pipe.predict_proba(X_val)[:, 1]
    win_sweep = sweep(y_val, proba, selected_model)
    C.save_table(win_sweep, "selected_model_threshold_sweep")
    tuning = pd.read_csv(C.TABLES_DIR / "hyperparameter_tuning_results.csv")
    row = tuning[tuning["model_name"] == selected_model]
    best_params = row["best_params"].iloc[0] if len(row) else "{}"
    C.save_table(pd.DataFrame([{
        "selected_model": selected_model,
        "imbalance_strategy": win["imbalance_strategy"],
        "selection_rule": "Highest validation PR-AUC; ties broken by F1 then alert volume",
        "selection_automatic": True,
        "tuned_hyperparameters": best_params,
        "threshold_rule": "Maximise validation F1 over a 0.01-step grid",
        "selected_threshold": selected_threshold,
        "validation_precision": float(win["precision"]),
        "validation_recall": float(win["recall"]),
        "validation_f1": float(win["f1_score"]),
        "validation_pr_auc": float(win["pr_auc_average_precision"]),
        "validation_false_positives": int(win["false_positives"]),
        "validation_false_negatives": int(win["false_negatives"]),
        "test_set_used": False,
    }]), "final_model_and_threshold_selection")
    # ---- Figures -------------------------------------------------------------
    prec, rec, _ = precision_recall_curve(y_val, proba)
    fpr, tpr, _ = roc_curve(y_val, proba)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rec, prec, color="#1f4e79", lw=2, label=selected_model)
    ax.axhline(y_val.mean(), ls="--", color="grey",
               label=f"No-skill ({y_val.mean():.4f})")
    ax.scatter([win["recall"]], [win["precision"]], color="#c00000", zorder=5,
               label=f"Operating point (t={selected_threshold})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Validation precision-recall curve")
    ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=.3)
    C.save_figure(fig, "fig04_validation_pr_curve"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, color="#2e7d32", lw=2)
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Validation ROC curve"); ax.grid(alpha=.3)
    C.save_figure(fig, "fig05_validation_roc_curve"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(win_sweep["threshold"], win_sweep["precision"], label="Precision", lw=2)
    ax.plot(win_sweep["threshold"], win_sweep["recall"], label="Recall", lw=2)
    ax.plot(win_sweep["threshold"], win_sweep["f1_score"], label="F1-score", lw=2)
    ax.axvline(selected_threshold, ls="--", color="#c00000",
               label=f"Optimised t={selected_threshold}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold optimisation on validation data")
    ax.legend(); ax.grid(alpha=.3)
    C.save_figure(fig, "fig06_threshold_optimisation"); plt.close(fig)
    # Model comparison bar chart
    comp = best_df.sort_values("pr_auc_average_precision")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    yp = np.arange(len(comp)); h = 0.38
    ax.barh(yp - h/2, comp["pr_auc_average_precision"], h, label="PR-AUC", color="#1f4e79")
    ax.barh(yp + h/2, comp["f1_score"], h, label="F1-score", color="#8faadc")
    ax.set_yticks(yp)
    ax.set_yticklabels([m.replace("_", " ") for m in comp["model_name"]], fontsize=8)
    ax.set_xlabel("Validation score"); ax.set_xlim(0, 1.05)
    ax.set_title("Model comparison at optimised thresholds")
    ax.legend(loc="lower right"); ax.grid(alpha=.3, axis="x")
    C.save_figure(fig, "fig03_model_comparison"); plt.close(fig)
    # ---- Freeze --------------------------------------------------------------

    C.FINAL_CONFIG_PATH.write_text(json.dumps({
        "selected_model": selected_model,
        "imbalance_strategy": str(win["imbalance_strategy"]),
        "selected_threshold": selected_threshold,
        "tuned_hyperparameters": best_params,
        "model_path": str(C.MODELS_DIR / f"{selected_model}_pipeline.joblib"),
        "selection_automatic": True,
        "random_state": C.RANDOM_STATE,
        "dataset_doi": C.DATASET_DOI,
    }, indent=2))
    joblib.dump(pipe, C.FINAL_MODEL_PATH)
    C.log_step("Step 06 complete. Test set still untouched.")
if __name__ == "__main__":
    main()
