import pandas as pd
from pathlib import Path
from .config import TARGET

EXPECTED = {
    "gender","age","hypertension","heart_disease","ever_married",
    "work_type","Residence_type","avg_glucose_level","bmi","smoking_status",TARGET
}

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    missing = EXPECTED - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    df = df.drop_duplicates()

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").clip(0, 120)
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").clip(10, 80)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df
