from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = ["Anemia", "Osteoporosis", "Diabetes", "Cardio"]
NUMERIC = ["Age", "Height", "Weight", "BMI_calc", "Vitamin_D", "Iron", "Calcium", "Vitamin_B12", "Omega_3", "Zinc", "Magnesium", "Protein"]
CATEGORICAL = ["Gender", "Physical_Activity", "Diet_Quality", "BMI_Category"]
STATUS_COLUMNS = ["VitD_Status", "Iron_Status", "Calcium_Status"]

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for c in ["Height", "Weight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    height_m = df["Height"] / 100.0
    df["BMI_calc"] = df["Weight"] / height_m.pow(2)
    df.loc[~np.isfinite(df["BMI_calc"]) | (height_m <= 0) | (df["Weight"] <= 0), "BMI_calc"] = np.nan
    df["BMI_Category"] = pd.cut(df["BMI_calc"], bins=[0,18.5,25,30,np.inf], labels=["Underweight","Normal","Overweight","Obesity"], right=False).astype("object")
    return df

def make_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, NUMERIC), ("cat", categorical_pipe, CATEGORICAL)], remainder="drop")

def features_for_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    df = engineer_features(frame)
    # Explicitly keep only approved predictor columns; excludes all disease labels and derived status fields.
    return df[NUMERIC + CATEGORICAL].copy()
