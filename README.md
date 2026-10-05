# Nutraceutical-Based Multi-Disease Risk Prediction

An end-to-end machine learning pipeline and interactive Streamlit web application that estimates disease risk scores for **Anemia, Osteoporosis, Type 2 Diabetes, and Cardiovascular Disease** from demographic, physical measurement, and 24-hour nutrient intake data.

> 🌐 **Real Dataset:** Trained on **5,073 real adult participants** from the **CDC NHANES (National Health and Nutrition Examination Survey)** dataset.

> ⚡ **Optimizations:** Enhanced with 10+ clinical/nutritional ratio features (Waist-to-Height ratio, Na/K ratio, Ca/Mg ratio, Pulse Pressure, MAP, Glycemic ratio), XGBoost & ExtraTrees ensembles, class-rebalancing, and out-of-fold probability threshold tuning.

> ⚠️ **Important Disclaimer:** This repository is an educational research and portfolio prototype. Predictions reflect statistical patterns within survey and nutrient data and **must not be used for medical diagnosis or clinical decision-making**.

---

## 📊 Optimized CDC NHANES Model Performance ($N = 5,073$)

All models were evaluated on an isolated **20% stratified holdout test set (1,015 adult participants)**.

### 1. Holdout Test Set Evaluation (N = 1,015 Adult Participants)

| Disease Target | Selected Model | Optimal Threshold | Accuracy | Recall (Sensitivity) | Precision | F1-Score | **ROC-AUC** | **PR-AUC** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cardiovascular** | `LogisticRegression` | 0.61 | **77.6%** | **66.9%** | 0.308 | **0.422** | **0.804** | **0.343** |
| **Osteoporosis** | `LogisticRegression` | 0.49 | **69.6%** | **77.2%** | 0.497 | **0.605** | **0.778** | **0.561** |
| **Type 2 Diabetes** | `ExtraTrees` | 0.33 | **74.4%** | **61.4%** | 0.343 | **0.440** | **0.777** | **0.350** |
| **Anemia** | `ExtraTrees` | 0.11 | **59.2%** | **64.2%** | 0.080 | **0.141** | **0.651** | **0.075** |

---

### 2. Key Optimization Techniques Applied

1. **Clinical Feature Engineering:**
   - `Waist_to_Height_Ratio`: Key visceral adiposity metric for cardiometabolic disease.
   - `Sodium_to_Potassium_Ratio`: Major determinant of blood pressure regulation.
   - `Calcium_to_Magnesium_Ratio`: Essential balance metric for bone mineral density.
   - `Pulse_Pressure` & `MAP` (Mean Arterial Pressure): Direct indicators of arterial stiffness.
   - `Sugar_to_Fiber_Ratio`: Glycemic load and insulin sensitivity proxy.
   - `BMI_Age_Interaction`: Non-linear risk multiplier for older age demographics.

2. **Probability Threshold Tuning:**
   - Instead of default 0.5 thresholding (which caused zero recall on highly imbalanced targets), optimal decision thresholds were derived from out-of-fold validation predictions to boost **Sensitivity / Recall (up to 77.2%)** while maintaining competitive accuracy.

3. **Class Balancing & Ensembles:**
   - Incorporated `XGBoost` (scale_pos_weight rebalancing), `ExtraTreesClassifier`, `RandomForestClassifier`, and regularized `LogisticRegression` with cost-sensitive class weights.

---

## 📈 CDC NHANES Dataset Summary

- **Source:** CDC National Health and Nutrition Examination Survey
- **Filtered Population:** 5,073 U.S. Adults (Age ≥ 18)
- **Features Extracted:** 36 demographic, physical, and nutritional predictors
- **Target Distribution:**
  - **Osteoporosis:** 1,532 positive / 5,073 total (**30.2%**)
  - **Diabetes:** 829 positive / 5,073 total (**16.3%**)
  - **Cardiovascular Disease:** 621 positive / 5,073 total (**12.2%**)
  - **Anemia:** 263 positive / 5,073 total (**5.2%**)

---

## 🚀 Deployment & Local Setup

```bash
# Clone repo & setup environment
git clone https://github.com/ABHI-2802/nutraceutical-disease-risk-prediction.git
cd nutraceutical-disease-risk-prediction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run dataset fetcher & optimized training
python -m src.fetch_real_nhanes
python -m src.train

# Launch Streamlit web app
streamlit run app.py
```
