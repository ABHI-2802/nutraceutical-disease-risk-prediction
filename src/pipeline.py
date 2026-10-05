from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = ["Anemia", "Osteoporosis", "Diabetes", "Cardio"]
STATUS_COLUMNS = ["VitD_Status", "Iron_Status", "Calcium_Status"]

DEFAULT_NUMERIC = [
    "Age", "Height", "Weight", "BMI_calc", "Waist", "Systolic_BP", "Diastolic_BP",
    "Vitamin_D", "Iron", "Calcium", "Vitamin_B12", "Omega_3", "Zinc", "Magnesium", "Protein",
    "Calories", "Sugars", "Fiber", "Total_Fat", "Sat_Fat", "Sodium", "Potassium",
    "Waist_to_Height_Ratio", "Sodium_to_Potassium_Ratio", "Calcium_to_Magnesium_Ratio",
    "VitD_Calcium_Index", "Sugar_to_Fiber_Ratio", "Sat_Fat_Ratio", "Pulse_Pressure",
    "MAP", "BMI_Age_Interaction", "Iron_to_Protein_Ratio"
]
DEFAULT_CATEGORICAL = ["Gender", "Physical_Activity", "Diet_Quality", "BMI_Category"]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    # Map height and weight to BMI_calc if not present
    if "Height" in df.columns and "Weight" in df.columns:
        h = pd.to_numeric(df["Height"], errors="coerce")
        w = pd.to_numeric(df["Weight"], errors="coerce")
        height_m = h / 100.0
        df["BMI_calc"] = w / height_m.pow(2)
        df.loc[~np.isfinite(df["BMI_calc"]) | (height_m <= 0) | (w <= 0), "BMI_calc"] = np.nan
        df["BMI_Category"] = pd.cut(df["BMI_calc"], bins=[0, 18.5, 25, 30, np.inf], labels=["Underweight", "Normal", "Overweight", "Obesity"], right=False).astype("object")

    # Map alias columns if present
    aliases = {
        "BMXHT": "Height", "BMXWT": "Weight", "BMXWAIST": "Waist",
        "BPXSY1": "Systolic_BP", "BPXDI1": "Diastolic_BP",
        "DR1TVD": "Vitamin_D", "DR1TFE": "Iron", "DR1TCA": "Calcium",
        "DR1TB12": "Vitamin_B12", "DR1TPFAT": "Omega_3", "DR1TZN": "Zinc",
        "DR1TMG": "Magnesium", "DR1TPROT": "Protein", "DR1TKCAL": "Calories",
        "DR1TSUGR": "Sugars", "DR1TFIBE": "Fiber", "DR1TTFAT": "Total_Fat",
        "DR1TSFAT": "Sat_Fat", "DR1TSODI": "Sodium", "DR1TPOTA": "Potassium"
    }
    for old_c, new_c in aliases.items():
        if old_c in df.columns and new_c not in df.columns:
            df[new_c] = df[old_c]

    # Clinical ratios
    if "Waist" in df.columns and "Height" in df.columns:
        df["Waist_to_Height_Ratio"] = df["Waist"] / (df["Height"] + 1e-5)
    if "Sodium" in df.columns and "Potassium" in df.columns:
        df["Sodium_to_Potassium_Ratio"] = df["Sodium"] / (df["Potassium"] + 1e-5)
    if "Calcium" in df.columns and "Magnesium" in df.columns:
        df["Calcium_to_Magnesium_Ratio"] = df["Calcium"] / (df["Magnesium"] + 1e-5)
    if "Vitamin_D" in df.columns and "Calcium" in df.columns:
        df["VitD_Calcium_Index"] = df["Vitamin_D"] * df["Calcium"]
    if "Sugars" in df.columns and "Fiber" in df.columns:
        df["Sugar_to_Fiber_Ratio"] = df["Sugars"] / (df["Fiber"] + 1e-5)
    if "Sat_Fat" in df.columns and "Total_Fat" in df.columns:
        df["Sat_Fat_Ratio"] = df["Sat_Fat"] / (df["Total_Fat"] + 1e-5)
    if "Systolic_BP" in df.columns and "Diastolic_BP" in df.columns:
        df["Pulse_Pressure"] = (df["Systolic_BP"] - df["Diastolic_BP"]).clip(lower=0)
        df["MAP"] = df["Diastolic_BP"] + (df["Pulse_Pressure"] / 3.0)
    if "BMI_calc" in df.columns and "Age" in df.columns:
        df["BMI_Age_Interaction"] = df["BMI_calc"] * df["Age"]
    if "Iron" in df.columns and "Protein" in df.columns:
        df["Iron_to_Protein_Ratio"] = df["Iron"] / (df["Protein"] + 1e-5)

    return df


def get_feature_lists(df: pd.DataFrame):
    num = [c for c in DEFAULT_NUMERIC if c in df.columns]
    cat = [c for c in DEFAULT_CATEGORICAL if c in df.columns]
    return num, cat


def make_preprocessor(df: pd.DataFrame | None = None):
    if df is not None:
        num, cat = get_feature_lists(df)
    else:
        num, cat = DEFAULT_NUMERIC, DEFAULT_CATEGORICAL

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([("num", numeric_pipe, num), ("cat", categorical_pipe, cat)], remainder="drop")


def features_for_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    df = engineer_features(frame)
    num, cat = get_feature_lists(df)
    return df[num + cat].copy()
