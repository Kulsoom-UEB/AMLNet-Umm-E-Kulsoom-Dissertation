"""
Step 05: Model Development with Hyperparameter Tuning.
CRISP-DM: Modelling.
Three algorithms (LightGBM removed at project scope):
    Logistic Regression - interpretable linear baseline
    Random Forest       - non-linear ensemble
    XGBoost             - gradient boosting
Two imbalance strategies, per the report's pipeline:
    Class weighting - scaling + one-hot BEFORE the model
    SMOTENC         - scaling -> SMOTENC -> one-hot AFTER resampling
Hyperparameters are tuned with RandomizedSearchCV using stratified CV on the
TRAINING set only, scored by average precision (PR-AUC) because the positive
class is ~0.16 % of the data. Validation data is never seen during tuning.
Nothing about the winning model is hardcoded: the best configuration per
algorithm is discovered by search, and the best overall model is selected in
step 07 from the metrics table.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
import amlnet_resampling as R  # noqa: E402
# Search budget. Kept modest so the whole pipeline runs in a sensible time on a
# laptop; raise N_ITER for a more exhaustive search.
N_ITER = 6
CV_FOLDS = 3
SMOTENC_SAMPLING_STRATEGY = 0.05   # minority : majority after resampling
TUNING_SUBSAMPLE = 250_000         # rows used for CV search (all positives kept)
def tuning_subsample(X, y, n, seed=C.RANDOM_STATE):
    """Subsample for CV search, keeping every laundering-positive case."""
    if len(y) <= n:
        return X, y
    pos = y.index[y == 1]
    neg = y.index[y == 0]
    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg, size=max(n - len(pos), 1), replace=False)
    idx = np.concatenate([pos.values, keep_neg])
    rng.shuffle(idx)
    return X.loc[idx], y.loc[idx]
def build_search_spaces(scale_pos_weight):
    """Estimator + hyperparameter grid for each algorithm."""
    from xgboost import XGBClassifier
    return {
        "Logistic_Regression": {
            "class_weight": LogisticRegression(
                class_weight="balanced", max_iter=3000, solver="lbfgs",
                n_jobs=-1, random_state=C.RANDOM_STATE),
            "smotenc": LogisticRegression(
                max_iter=3000, solver="lbfgs", n_jobs=-1,
                random_state=C.RANDOM_STATE),
            "grid": {"model__C": loguniform(1e-3, 1e2)},
        },
        "Random_Forest": {
            "class_weight": RandomForestClassifier(
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=C.RANDOM_STATE),
            "smotenc": RandomForestClassifier(
                n_jobs=-1, random_state=C.RANDOM_STATE),
            "grid": {
                "model__n_estimators": randint(100, 301),
                "model__max_depth": [None, 12, 20, 30],
                "model__min_samples_leaf": randint(1, 6),
                "model__max_features": ["sqrt", "log2"],

            },
        },
        "XGBoost": {
            "class_weight": XGBClassifier(
                scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE),
            "smotenc": XGBClassifier(
                eval_metric="aucpr", tree_method="hist", n_jobs=-1,
                random_state=C.RANDOM_STATE),
            "grid": {
                "model__n_estimators": randint(150, 401),
                "model__max_depth": randint(3, 9),
                "model__learning_rate": uniform(0.02, 0.18),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3),
            },
        },
    }
def main() -> None:
    C.set_seeds()
    C.ensure_dirs()
    X_train = C.read_parquet_or_csv(C.SPLIT_PATHS["X_train"])
    X_val = C.read_parquet_or_csv(C.SPLIT_PATHS["X_val"])
    y_train = C.read_parquet_or_csv(C.SPLIT_PATHS["y_train"])[C.TARGET_COLUMN]
    y_val = C.read_parquet_or_csv(C.SPLIT_PATHS["y_val"])[C.TARGET_COLUMN]
    categorical_features = [c for c in C.CATEGORICAL_FEATURES if c in X_train.columns]
    numeric_features = [c for c in X_train.columns if c not in categorical_features]
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)
    C.save_table(pd.DataFrame([{
        "training_rows": len(y_train),
        "training_negatives": n_neg,
        "training_positives": n_pos,
        "scale_pos_weight": round(scale_pos_weight, 6),
        "smotenc_sampling_strategy": SMOTENC_SAMPLING_STRATEGY,
        "positives_after_smotenc": int(n_neg * SMOTENC_SAMPLING_STRATEGY),
        "synthetic_positives_created": int(n_neg * SMOTENC_SAMPLING_STRATEGY) - n_pos,
        "tuning_search_iterations": N_ITER,
        "tuning_cv_folds": CV_FOLDS,
        "tuning_scoring": "average_precision (PR-AUC)",
        "tuning_subsample_rows": min(TUNING_SUBSAMPLE, len(y_train)),
    }]), "imbalance_and_tuning_configuration")
    X_tune, y_tune = tuning_subsample(X_train, y_train, TUNING_SUBSAMPLE)
    C.log_step(f"Tuning subsample: {len(y_tune):,} rows, {int(y_tune.sum())} positives")
    spaces = build_search_spaces(scale_pos_weight)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    results, tuning_records = [], []
    # ---- Majority-class benchmark ------------------------------------------
    bench = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    rec = C.evaluate_predictions(y_val, bench.predict_proba(X_val)[:, 1], 0.5,
                                 "Majority_Class_Benchmark", "validation")
    rec.update({"imbalance_strategy": "None (benchmark)",
                "base_model": "Majority_Class_Benchmark",
                "best_params": "{}", "cv_best_pr_auc": np.nan,
                "training_seconds": 0.0})
    results.append(rec)
    joblib.dump(bench, C.MODELS_DIR / "Majority_Class_Benchmark.joblib")
    C.log_step(f"Benchmark: accuracy {rec['accuracy']:.6f}, TP {rec['true_positives']}")
    # ---- Tune + train every algorithm x strategy ---------------------------
    for algo, spec in spaces.items():
        for strategy in ["class_weight", "smotenc"]:
            label = f"{algo}_{'class_weight' if strategy=='class_weight' else 'SMOTENC'}"
            C.log_step(f"Tuning {label} ...")
            t0 = time.time()
            scale_numeric = (algo == "Logistic_Regression")
            if strategy == "class_weight":
                pipe = R.build_class_weight_pipeline(
                    spec["class_weight"], numeric_features, categorical_features,
                    scale_numeric=scale_numeric)
            else:
                pipe = R.build_smotenc_pipeline(
                    spec["smotenc"], numeric_features, categorical_features,
                    X_train, SMOTENC_SAMPLING_STRATEGY, C.RANDOM_STATE)
            search = RandomizedSearchCV(
                pipe, spec["grid"], n_iter=N_ITER, scoring="average_precision",
                cv=cv, random_state=C.RANDOM_STATE, n_jobs=1, refit=False,
                error_score="raise")
            search.fit(X_tune, y_tune)
            best_params = search.best_params_
            cv_score = float(search.best_score_)
            C.log_step(f"  best CV PR-AUC {cv_score:.4f} | {best_params}")
            # Refit the winning configuration on the FULL training set
            final_pipe = pipe.set_params(**best_params)

            final_pipe.fit(X_train, y_train)
            elapsed = time.time() - t0
            proba = final_pipe.predict_proba(X_val)[:, 1]
            rec = C.evaluate_predictions(y_val, proba, 0.5, label, "validation")
            rec.update({
                "imbalance_strategy": ("Class weighting" if strategy == "class_weight"
                                       else f"SMOTENC ({SMOTENC_SAMPLING_STRATEGY})"),
                "base_model": algo,
                "best_params": json.dumps(
                    {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                     for k, v in best_params.items()}),
                "cv_best_pr_auc": round(cv_score, 6),
                "training_seconds": round(elapsed, 1),
            })
            results.append(rec)
            tuning_records.append({
                "model_name": label, "base_model": algo,
                "imbalance_strategy": rec["imbalance_strategy"],
                "search_iterations": N_ITER, "cv_folds": CV_FOLDS,
                "cv_scoring": "average_precision",
                "cv_best_pr_auc": round(cv_score, 6),
                "best_params": rec["best_params"],
                "tuning_plus_refit_seconds": round(elapsed, 1),
            })
            joblib.dump(final_pipe, C.MODELS_DIR / f"{label}_pipeline.joblib")
            pd.DataFrame({
                "row_index": X_val.index, "true_label": y_val.values,
                "predicted_label": (proba >= 0.5).astype(int),
                "predicted_probability": proba,
            }).to_csv(C.TABLES_DIR / f"validation_predictions_{label}.csv", index=False)
            C.log_step(f"  {label}: PR-AUC {rec['pr_auc_average_precision']:.4f} | "
                       f"P {rec['precision']:.4f} R {rec['recall']:.4f} "
                       f"F1 {rec['f1_score']:.4f} FP {rec['false_positives']} "
                       f"({elapsed:.0f}s)")
    metrics = pd.DataFrame(results)
    metrics["rank_by_pr_auc"] = metrics["pr_auc_average_precision"].rank(
        ascending=False, method="min").astype(int)
    metrics["rank_by_f1"] = metrics["f1_score"].rank(
        ascending=False, method="min").astype(int)
    C.save_table(metrics, "validation_metrics_all_models")
    C.save_table(pd.DataFrame(tuning_records), "hyperparameter_tuning_results")
    cols = ["model_name", "imbalance_strategy", "precision", "recall", "f1_score",
            "pr_auc_average_precision", "true_positives", "false_positives",
            "false_negatives"]
    print("\n" + metrics[cols].to_string(index=False))
    C.log_step("Step 05 complete.")
if __name__ == "__main__":
    main()
