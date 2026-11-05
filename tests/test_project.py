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

def test_vercel_predict_api():
    from api.predict import predict_profile
    sample = {
        'Age': 40, 'Gender': 'F', 'Height': 170.0, 'Weight': 70.0,
        'Physical_Activity': 'Moderate', 'Diet_Quality': 'Good',
        'Vitamin_D': 25.0, 'Iron': 15.0, 'Calcium': 900.0, 'Vitamin_B12': 2.5,
        'Omega_3': 1.5, 'Zinc': 12.0, 'Magnesium': 300.0, 'Protein': 65.0
    }
    result = predict_profile(sample)
    assert result['success'] is True
    for t in TARGETS:
        assert t in result['predictions']
        assert 0.0 <= result['predictions'][t]['score'] <= 1.0

def test_dashboard_limitations_present():
    with open('public/index.html', 'r', encoding='utf-8') as f:
        html = f.read()
    assert "Measurement variability" in html
    assert "Input uncertainty" in html
    assert "Incomplete medical context" in html
    assert "Limited validation" in html

def test_vercel_entrypoint_exports():
    import app as app_module
    assert hasattr(app_module, "app")
    assert hasattr(app_module, "application")
    assert hasattr(app_module, "handler")
    assert callable(app_module.handler) or isinstance(app_module.handler, type)
