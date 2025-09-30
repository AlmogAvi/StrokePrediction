from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, roc_auc_score
from .utils import proba_or_decision
import matplotlib
matplotlib.use("Agg")  # כתיבה לקבצים בלבד, בלי GUI


def save_clf_report(y_true, y_pred, path: Path):
    text = classification_report(y_true, y_pred, digits=3)
    path.write_text(text, encoding="utf-8")

def plot_confusion(y_true, y_pred, out: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["0","1"]); plt.yticks(ticks, ["0","1"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]:d}", ha="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True"); plt.xlabel("Predicted"); plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_roc(y_true, y_scores, out: Path):
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = float("nan")
    fig = plt.figure(figsize=(4,3))
    RocCurveDisplay.from_predictions(y_true, y_scores)
    plt.title(f"ROC Curve (AUC={auc:.3f})")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return auc

def evaluate_and_save(name: str, clf, X_test, y_test, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_pred = clf.predict(X_test)
    save_clf_report(y_test, y_pred, reports_dir / f"{name}_classification_report.txt")
    plot_confusion(y_test, y_pred, reports_dir / f"{name}_confusion_matrix.png")
    y_scores = proba_or_decision(clf, X_test)
    auc = plot_roc(y_test, y_scores, reports_dir / f"{name}_roc_curve.png")
    return {"roc_auc": float(auc)}
