from pathlib import Path
from typing import Dict, Any
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from .config import TEST_SIZE, RANDOM_STATE, CV_FOLDS
from .features import build_preprocessor, split_Xy, get_feature_names
from .evaluate import evaluate_and_save

def split_data(df):
    X, y = split_Xy(df)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def train_single_model(name: str, base_clf, grid, X_train, y_train):
    pre  = build_preprocessor()
    pipe = Pipeline([("pre", pre), ("clf", base_clf)])
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs   = GridSearchCV(pipe, param_grid=grid, scoring="f1", cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def save_model(name: str, clf, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)

def run_training(models, grids, df, models_dir: Path, reports_dir: Path) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = split_data(df)
    results = {}
    last_best = None
    for name, clf in models.items():
        best, best_params, best_cv = train_single_model(name, clf, grids[name], X_train, y_train)
        save_model(name, best, models_dir / f"{name}.joblib")
        metrics = evaluate_and_save(name, best, X_test, y_test, reports_dir)
        results[name] = {"best_params": best_params, "cv_f1": float(best_cv), **metrics}
        last_best = best
    # save feature names from the fitted preprocessor
    if last_best is not None:
        pre = last_best.named_steps["pre"]
        from .utils import save_json
        save_json(get_feature_names(pre), reports_dir / "feature_names.json")
    return results
