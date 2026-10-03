import pandas as pd
from src.pipeline import TARGETS, features_for_target

def test_targets_are_present_and_binary():
    df=pd.read_csv('data/raw/disease_risk_dataset.csv')
    for t in TARGETS:
        assert t in df.columns
        assert set(df[t].dropna().unique()).issubset({0,1})

def test_no_target_or_status_leakage_in_features():
    df=pd.read_csv('data/raw/disease_risk_dataset.csv')
    for t in TARGETS:
        X=features_for_target(df.drop(columns=TARGETS,errors='ignore'),t)
        assert not any(c in X.columns for c in TARGETS)
        assert not any(c.endswith('_Status') for c in X.columns)

def test_bmi_engineering():
    df=pd.DataFrame([{'Age':30,'Gender':'F','Height':170,'Weight':68,'Physical_Activity':'Moderate','Diet_Quality':'Good','Vitamin_D':20,'Iron':15,'Calcium':900,'Vitamin_B12':2.5,'Omega_3':1.5,'Zinc':10,'Magnesium':300,'Protein':60}])
    X=features_for_target(df,'Cardio')
    assert abs(X['BMI_calc'].iloc[0]-23.5294)<0.02
