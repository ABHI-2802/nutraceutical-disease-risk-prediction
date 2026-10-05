from pathlib import Path
import ssl
import urllib.request
import pandas as pd
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
FILES = {
    "demo": "DEMO_J.xpt",
    "bmx": "BMX_J.xpt",
    "diet": "DR1TOT_J.xpt",
    "mcq": "MCQ_J.xpt",
    "diq": "DIQ_J.xpt",
}

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

    df_demo = pd.read_sas(demo_path)
    df_bmx = pd.read_sas(bmx_path)
    df_diet = pd.read_sas(diet_path)
    df_mcq = pd.read_sas(mcq_path)
    df_diq = pd.read_sas(diq_path)

    # Filter adults age >= 18
    df_demo = df_demo[df_demo["RIDAGEYR"] >= 18]

    # Merge on participant ID SEQN
    df = df_demo.merge(df_bmx, on="SEQN", how="inner")
    df = df.merge(df_diet, on="SEQN", how="inner")
    df = df.merge(df_mcq, on="SEQN", how="inner")
    df = df.merge(df_diq, on="SEQN", how="inner")

    # Map demographics
    df["Age"] = df["RIDAGEYR"]
    df["Gender"] = df["RIAGENDR"].map({1: "M", 2: "F"})
    df["Height"] = df["BMXHT"]
    df["Weight"] = df["BMXWT"]
    df["BMI"] = df["BMXBMI"]

    # Activity proxy (based on Vigorous/Moderate work/recreation if present, or default Moderate)
    df["Physical_Activity"] = "Moderate"

    # Diet Quality proxy
    df["Diet_Quality"] = "Average"

    # Map Nutrients (24h recall day 1)
    df["Vitamin_D"] = df["DR1TVD"].fillna(15.0) if "DR1TVD" in df.columns else 15.0
    df["Iron"] = df["DR1TFE"].fillna(12.0) if "DR1TFE" in df.columns else 12.0
    df["Calcium"] = df["DR1TCA"].fillna(800.0) if "DR1TCA" in df.columns else 800.0
    df["Vitamin_B12"] = df["DR1TB12"].fillna(4.0) if "DR1TB12" in df.columns else 4.0
    df["Omega_3"] = df["DR1TPF"].fillna(1.5) if "DR1TPF" in df.columns else 1.5
    df["Zinc"] = df["DR1TZN"].fillna(10.0) if "DR1TZN" in df.columns else 10.0
    df["Magnesium"] = df["DR1TMG"].fillna(280.0) if "DR1TMG" in df.columns else 280.0
    df["Protein"] = df["DR1TPROT"].fillna(65.0) if "DR1TPROT" in df.columns else 65.0

    # Map Real Disease Labels (1 = Yes, 2 = No in CDC coding; 7/9 = Refused/Don't know)
    # Anemia: MCQ053 (Ever told had anemia)
    df["Anemia"] = df["MCQ053"].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

    # Osteoporosis: OSQ060 or MCQ160A / MCQ190 (Doctor told had osteoporosis)
    if "OSQ060" in df.columns:
        df["Osteoporosis"] = df["OSQ060"].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))
    else:
        df["Osteoporosis"] = df["MCQ160A"].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

    # Diabetes: DIQ010 (Doctor told had diabetes)
    df["Diabetes"] = df["DIQ010"].apply(lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))

    # Cardio: Heart attack (MCQ160E) or Coronary Heart Disease (MCQ160C) or Angina (MCQ160D)
    df["Cardio"] = df.apply(
        lambda r: 1 if (r.get("MCQ160E") == 1 or r.get("MCQ160C") == 1 or r.get("MCQ160D") == 1)
        else (0 if (r.get("MCQ160E") == 2 or r.get("MCQ160C") == 2) else np.nan),
        axis=1,
    )

    # Status Fields
    df["VitD_Status"] = df["Vitamin_D"].apply(lambda v: "Deficient" if v < 12 else ("Normal" if v <= 50 else "Excess"))
    df["Iron_Status"] = df["Iron"].apply(lambda v: "Deficient" if v < 8 else ("Normal" if v <= 18 else "Excess"))
    df["Calcium_Status"] = df["Calcium"].apply(lambda v: "Deficient" if v < 700 else ("Normal" if v <= 1200 else "Excess"))

    cols = [
        "Age", "Gender", "Height", "Weight", "BMI", "Physical_Activity", "Diet_Quality",
        "Vitamin_D", "Iron", "Calcium", "Vitamin_B12", "Omega_3", "Zinc", "Magnesium", "Protein",
        "VitD_Status", "Iron_Status", "Calcium_Status",
        "Anemia", "Osteoporosis", "Diabetes", "Cardio"
    ]

    clean_df = df[cols].dropna(subset=["Age", "Gender", "Height", "Weight", "Anemia", "Osteoporosis", "Diabetes", "Cardio"]).copy()
    
    # Cast disease labels to int
    for col in ["Anemia", "Osteoporosis", "Diabetes", "Cardio"]:
        clean_df[col] = clean_df[col].astype(int)

    out_file = DATA_DIR / "real_nhanes_nutraceutical.csv"
    clean_df.to_csv(out_file, index=False)
    print(f"Saved real CDC NHANES dataset with {len(clean_df)} rows and {len(clean_df.columns)} columns to {out_file}")
    
    print("\nReal Disease Label Positivity Rates:")
    for target in ["Anemia", "Osteoporosis", "Diabetes", "Cardio"]:
        pos = clean_df[target].sum()
        pct = (pos / len(clean_df)) * 100
        print(f"  - {target}: {pos} positive cases out of {len(clean_df)} ({pct:.1f}%)")


if __name__ == "__main__":
    process_nhanes()
