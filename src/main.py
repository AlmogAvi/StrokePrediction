import argparse
from pathlib import Path
import pandas as pd

from .config import MODELS_DIR, REPORTS_DIR
from .data import load_data
from .models import get_models_and_grids
from .train import run_training
from .visualize import quick_eda
from .utils import save_json, seed_everything

def parse_args():
    p = argparse.ArgumentParser(description="Stroke Data Mining Pipeline")
    p.add_argument("--csv", type=str, required=True, help="Path to stroke_data.csv")
    p.add_argument("--models", type=str, default="logreg,rf,gb,svc,knn",
                   help="Comma-separated subset: logreg,rf,gb,svc,knn")
    p.add_argument("--eda", action="store_true", help="Run quick EDA plots")
    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(42)

    df = load_data(Path(args.csv))
    if args.eda:
        quick_eda(df, REPORTS_DIR)

    all_models, grids = get_models_and_grids()
    selected = [m.strip() for m in args.models.split(",") if m.strip() in all_models]
    models = {k: all_models[k] for k in selected}

    results = run_training(models, grids, df, MODELS_DIR, REPORTS_DIR)
    save_json(results, REPORTS_DIR / "summary.json")
    print("Done. See outputs/ for models and reports.")

if __name__ == "__main__":
    main()
