from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .config import TARGET

CATEGORICAL = ["gender","ever_married","work_type","Residence_type","smoking_status"]
NUMERICAL   = ["age","hypertension","heart_disease","avg_glucose_level","bmi"]

def split_Xy(df: pd.DataFrame):
    X = df[CATEGORICAL + NUMERICAL].copy()
    y = df[TARGET].copy()
    return X, y

def _ohe():
    # תואם scikit-learn חדשים/ישנים
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _ohe())
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERICAL),
        ("cat", cat_pipe, CATEGORICAL),
    ])

def get_feature_names(pre: ColumnTransformer) -> List[str]:
    num_feats = NUMERICAL
    cat_ohe   = pre.named_transformers_["cat"].named_steps["ohe"]
    cat_feats = cat_ohe.get_feature_names_out(CATEGORICAL).tolist()
    return num_feats + cat_feats
