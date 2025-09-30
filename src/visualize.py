from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # כתיבה לקבצים בלבד, בלי GUI


def quick_eda(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(4,3))
    df["stroke"].value_counts().sort_index().plot(kind="bar")
    plt.title("Target balance (0/1)")
    plt.tight_layout()
    fig.savefig(out_dir / "eda_target_balance.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(4,3))
    df["age"].plot(kind="hist", bins=30)
    plt.title("Age distribution"); plt.tight_layout()
    fig.savefig(out_dir / "eda_age_hist.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="stroke", y="avg_glucose_level", data=df)
    plt.title("Glucose by stroke"); plt.tight_layout()
    fig.savefig(out_dir / "eda_glucose_by_stroke.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(4,3))
    sns.boxplot(x="stroke", y="bmi", data=df)
    plt.title("BMI by stroke"); plt.tight_layout()
    fig.savefig(out_dir / "eda_bmi_by_stroke.png", dpi=150); plt.close(fig)
