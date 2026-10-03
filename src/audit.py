from pathlib import Path
import json
import pandas as pd
from src.pipeline import TARGETS

ROOT = Path(__file__).resolve().parents[1]

def run_audit():
    df = pd.read_csv(ROOT / "data/raw/disease_risk_dataset.csv")
    out = {"shape": list(df.shape), "columns": list(df.columns), "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()), "targets": {}, "warnings": []}
    for t in TARGETS:
        if t not in df: raise ValueError(f"Required target missing: {t}")
        vals = set(pd.to_numeric(df[t], errors="coerce").dropna().unique().tolist())
        if not vals.issubset({0,1}): raise ValueError(f"Target {t} must be binary 0/1; found {vals}")
        counts = df[t].value_counts(dropna=False).to_dict()
        out["targets"][t] = {str(k): int(v) for k,v in counts.items()}
    out["warnings"].append("Dataset provenance, consent/license, and clinical label-generation method were not independently verified.")
    out["warnings"].append("Nutrient status summary columns are excluded from modeling to reduce possible target leakage.")
    out["warnings"].append("Small structured dataset; results may not generalize to real populations.")
    p=ROOT/"reports/metrics/data_audit.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    return out

if __name__ == "__main__": run_audit()
