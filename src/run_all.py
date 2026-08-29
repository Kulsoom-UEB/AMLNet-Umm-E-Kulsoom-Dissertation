"""
Run the complete AMLNet pipeline end to end.
    python src/run_all.py            # every step
    python src/run_all.py --from 05  # resume
    python src/run_all.py --only 08  # single step
"""
from __future__ import annotations
import argparse
import importlib
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import amlnet_common as C  # noqa: E402
STEPS = [
    ("00", "step00_setup", "Project setup and reproducibility"),
    ("01", "step01_data_understanding", "Data audit and analysis"),
    ("02", "step02_data_preparation", "Leakage removal and cleaning"),
    ("03", "step03_feature_engineering", "Feature engineering"),
    ("04", "step04_split", "Train / validation / test split"),
    ("05", "step05_model_development", "Model development + hyperparameter tuning"),
    ("06", "step06_threshold_optimization", "Best-model selection + threshold optimisation"),
    ("07", "step07_final_testing", "Final test-set evaluation"),
    ("08", "step08_explainability", "SHAP, feature and permutation importance"),
]
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="00")
    ap.add_argument("--only", dest="only", default=None)
    a = ap.parse_args()
    todo = ([s for s in STEPS if s[0] == a.only] if a.only
            else [s for s in STEPS if s[0] >= a.start])
    if not todo:
        raise SystemExit("No matching steps.")
    t_all = time.time()
    for code, mod, desc in todo:
        print("\n" + "=" * 78)
        print(f"STEP {code}: {desc}")
        print("=" * 78, flush=True)
        t0 = time.time()
        importlib.import_module(mod).main()
        print(f"--- step {code} finished in {time.time() - t0:.1f}s")
    print("\n" + "=" * 78)
    print(f"Pipeline finished in {(time.time() - t_all)/60:.1f} minutes")
    print(f"Tables : {C.TABLES_DIR}")
    print(f"Figures: {C.FIGURES_DIR}")
    print(f"Models : {C.MODELS_DIR}")
    print("Dashboard: streamlit run dashboard/app.py")
    print("=" * 78)
if __name__ == "__main__":
    main()
