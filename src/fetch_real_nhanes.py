from pathlib import Path
import ssl
import urllib.request
import pandas as pd
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_file(filename: str) -> Path:
    out_path = DATA_DIR / filename
    if not out_path.exists():
        url = BASE_URL + filename
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, out_path)
    return out_path


def process_nhanes():
    print("Fetching CDC NHANES 2017-2018 datasets...")
    demo_path = download_file("DEMO_J.xpt")
    bmx_path = download_file("BMX_J.xpt")
    diet_path = download_file("DR1TOT_J.xpt")
    mcq_path = download_file("MCQ_J.xpt")
    diq_path = download_file("DIQ_J.xpt")
    bpx_path = download_file("BPX_J.xpt")

    df_demo = pd.read_sas(demo_path)
    df_bmx = pd.read_sas(bmx_path)
    df_diet = pd.read_sas(diet_path)
    df_mcq = pd.read_sas(mcq_path)
    df_diq = pd.read_sas(diq_path)
    df_bpx = pd.read_sas(bpx_path)

    # Filter adults age >= 18
    df_demo = df_demo[df_demo["RIDAGEYR"] >= 18]

    # Merge on participant ID SEQN
    df = df_demo.merge(df_bmx, on="SEQN", how="inner")
    df = df.merge(df_diet, on="SEQN", how="inner")
    df = df.merge(df_mcq, on="SEQN", how="inner")
    df = df.merge(df_diq, on="SEQN", how="inner")
    df = df.merge(df_bpx, on="SEQN", how="left")

    # Demographics
    df["Age"] = df["RIDAGEYR"]
    df["Gender"] = df["RIAGENDR"].map({1: "M", 2: "F"})
    df["Height"] = df["BMXHT"]
    df["Weight"] = df["BMXWT"]
    df["BMI"] = df["BMXBMI"]

    # Physical measurements
    df["Waist"] = df["BMXWAIST"] if "BMXWAIST" in df.columns else np.nan
    df["Systolic_BP"] = df["BPXSY1"] if "BPXSY1" in df.columns else np.nan
    df["Diastolic_BP"] = df["BPXDI1"] if "BPXDI1" in df.columns else np.nan

    # Activity & Diet proxies
    df["Physical_Activity"] = "Moderate"
    df["Diet_Quality"] = "Average"

    # Core Nutrients & Dietary Factors
    df["Vitamin_D"] = df["DR1TVD"] if "DR1TVD" in df.columns else np.nan
    df["Iron"] = df["DR1TIRON"] if "DR1TIRON" in df.columns else (df["DR1TFE"] if "DR1TFE" in df.columns else np.nan)
    df["Calcium"] = df["DR1TCALC"] if "DR1TCALC" in df.columns else (df["DR1TCA"] if "DR1TCA" in df.columns else np.nan)
    df["Vitamin_B12"] = df["DR1TVB12"] if "DR1TVB12" in df.columns else (df["DR1TB12"] if "DR1TB12" in df.columns else np.nan)
    df["Omega_3"] = df["DR1TPFAT"] if "DR1TPFAT" in df.columns else np.nan
    df["Zinc"] = df["DR1TZINC"] if "DR1TZINC" in df.columns else (df["DR1TZN"] if "DR1TZN" in df.columns else np.nan)
    df["Magnesium"] = df["DR1TMAGN"] if "DR1TMAGN" in df.columns else (df["DR1TMG"] if "DR1TMG" in df.columns else np.nan)
    df["Protein"] = df["DR1TPROT"] if "DR1TPROT" in df.columns else np.nan
    df["Calories"] = df["DR1TKCAL"] if "DR1TKCAL" in df.columns else np.nan
    df["Sugars"] = df["DR1TSUGR"] if "DR1TSUGR" in df.columns else np.nan
    df["Fiber"] = df["DR1TFIBE"] if "DR1TFIBE" in df.columns else np.nan
    df["Total_Fat"] = df["DR1TTFAT"] if "DR1TTFAT" in df.columns else np.nan
    df["Sat_Fat"] = df["DR1TSFAT"] if "DR1TSFAT" in df.columns else np.nan
    df["Sodium"] = df["DR1TSODI"] if "DR1TSODI" in df.columns else np.nan
    df["Potassium"] = df["DR1TPOTA"] if "DR1TPOTA" in df.columns else np.nan

    # Fill missing continuous nutrients/measurements with median
    num_cols = [
        "Height", "Weight", "BMI", "Waist", "Systolic_BP", "Diastolic_BP",
        "Vitamin_D", "Iron", "Calcium", "Vitamin_B12", "Omega_3", "Zinc",
        "Magnesium", "Protein", "Calories", "Sugars", "Fiber", "Total_Fat",
        "Sat_Fat", "Sodium", "Potassium"
    ]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Engineered Clinical Ratios & Interaction Features
    df["Waist_to_Height_Ratio"] = df["Waist"] / (df["Height"] + 1e-5)
    df["Sodium_to_Potassium_Ratio"] = df["Sodium"] / (df["Potassium"] + 1e-5)
    df["Calcium_to_Magnesium_Ratio"] = df["Calcium"] / (df["Magnesium"] + 1e-5)
    df["VitD_Calcium_Index"] = df["Vitamin_D"] * df["Calcium"]
    df["Sugar_to_Fiber_Ratio"] = df["Sugars"] / (df["Fiber"] + 1e-5)
    df["Sat_Fat_Ratio"] = df["Sat_Fat"] / (df["Total_Fat"] + 1e-5)
    df["Pulse_Pressure"] = (df["Systolic_BP"] - df["Diastolic_BP"]).clip(lower=0)
    df["MAP"] = df["Diastolic_BP"] + (df["Pulse_Pressure"] / 3.0)
    df["BMI_Age_Interaction"] = df["BMI"] * df["Age"]
    df["Iron_to_Protein_Ratio"] = df["Iron"] / (df["Protein"] + 1e-5)

    # Status Fields
    df["VitD_Status"] = df["Vitamin_D"].apply(lambda v: "Deficient" if v < 12 else ("Normal" if v <= 50 else "Excess"))
    df["Iron_Status"] = df["Iron"].apply(lambda v: "Deficient" if v < 8 else ("Normal" if v <= 18 else "Excess"))
    df["Calcium_Status"] = df["Calcium"].apply(lambda v: "Deficient" if v < 700 else ("Normal" if v <= 1200 else "Excess"))

    # Disease Labels
    df["Anemia"] = df["MCQ053"].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

    if "OSQ060" in df.columns:
        df["Osteoporosis"] = df["OSQ060"].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))
    else:
        df["Osteoporosis"] = df["MCQ160A"].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

    df["Diabetes"] = df["DIQ010"].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

    df["Cardio"] = df.apply(
        lambda r: 1 if (r.get("MCQ160E") == 1 or r.get("MCQ160C") == 1 or r.get("MCQ160D") == 1 or r.get("MCQ160F") == 1 or r.get("MCQ160B") == 1)
        else (0 if (r.get("MCQ160E") == 2 or r.get("MCQ160C") == 2) else np.nan),
        axis=1,
    )

    clean_df = df.dropna(subset=["Age", "Gender", "Height", "Weight", "Anemia", "Osteoporosis", "Diabetes", "Cardio"]).copy()

    for col in ["Anemia", "Osteoporosis", "Diabetes", "Cardio"]:
        clean_df[col] = clean_df[col].astype(int)

    out_file = DATA_DIR / "real_nhanes_nutraceutical.csv"
    clean_df.to_csv(out_file, index=False)
    print(f"Saved optimized CDC NHANES dataset with {len(clean_df)} rows and {len(clean_df.columns)} columns to {out_file}")

    print("\nDisease Label Distribution:")
    for target in ["Anemia", "Osteoporosis", "Diabetes", "Cardio"]:
        pos = clean_df[target].sum()
        pct = (pos / len(clean_df)) * 100
        print(f"  - {target}: {pos} positive cases out of {len(clean_df)} ({pct:.1f}%)")


if __name__ == "__main__":
    process_nhanes()
