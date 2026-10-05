# Nutraceutical-Based Multi-Disease Risk Prediction

An end-to-end machine learning pipeline and interactive Streamlit web application that estimates disease risk scores for **Anemia, Osteoporosis, Type 2 Diabetes, and Cardiovascular Disease** from demographic, physical measurement, and 24-hour nutrient intake data.

> 🌐 **Real Dataset:** Trained on **5,073 real adult participants** from the **CDC NHANES (National Health and Nutrition Examination Survey)** dataset.

> ⚡ **Optimizations:** Enhanced with 10+ clinical/nutritional ratio features (Waist-to-Height ratio, Na/K ratio, Ca/Mg ratio, Pulse Pressure, MAP, Glycemic ratio), XGBoost & ExtraTrees ensembles, class-rebalancing, and out-of-fold probability threshold tuning.

> ⚠️ **Important Disclaimer:** This repository is an educational research and portfolio prototype. Predictions reflect statistical patterns within survey and nutrient data and **must not be used for medical diagnosis or clinical decision-making**.

---

## 📊 Comprehensive Model Scores & Benchmark Metrics ($N = 5,073$)

All models were evaluated on an isolated **20% stratified holdout test set (1,015 adult participants)**. Prior to holdout testing, 5-fold cross-validation was conducted across 6 candidate algorithms on 4,058 training rows.

### 1. Primary Holdout Test Set Model Scores ($N = 1,015$ Adult Participants)

| Disease Target | Selected Model | Optimal Threshold | Accuracy | Recall (Sensitivity) | Precision | F1-Score | **ROC-AUC** | **PR-AUC** | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cardiovascular** | `LogisticRegression` | 0.61 | **77.6%** (0.776) | **66.9%** (0.669) | 0.309 | **0.422** | **0.804** | **0.343** | 0.189 |
| **Osteoporosis** | `LogisticRegression` | 0.49 | **69.6%** (0.696) | **77.2%** (0.772) | 0.498 | **0.605** | **0.778** | **0.561** | 0.195 |
| **Type 2 Diabetes** | `ExtraTrees` | 0.33 | **74.4%** (0.744) | **61.4%** (0.614) | 0.342 | **0.440** | **0.777** | **0.350** | 0.127 |
| **Anemia** | `ExtraTrees` | 0.11 | **59.2%** (0.592) | **64.2%** (0.642) | 0.079 | **0.141** | **0.651** | **0.075** | 0.056 |

---

### 2. Complete 5-Fold Cross-Validation Candidate Model Benchmark (Mean CV ROC-AUC ± Std)

During model exploration, 6 candidate models were benchmarked using 5-Fold Stratified Cross-Validation on the training split ($N = 4,058$):

| Candidate Algorithm | Cardio CV ROC-AUC | Diabetes CV ROC-AUC | Osteoporosis CV ROC-AUC | Anemia CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **DummyPrior (Baseline)** | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| **Logistic Regression** | **0.795 ± 0.015** | 0.789 ± 0.017 | **0.790 ± 0.011** | 0.656 ± 0.060 |
| **ExtraTrees** | 0.784 ± 0.022 | **0.790 ± 0.016** | 0.786 ± 0.011 | **0.684 ± 0.047** |
| **RandomForest** | 0.781 ± 0.021 | 0.784 ± 0.017 | 0.784 ± 0.012 | 0.682 ± 0.033 |
| **XGBoost** | 0.785 ± 0.023 | 0.772 ± 0.017 | 0.780 ± 0.011 | 0.654 ± 0.036 |
| **HistGradientBoosting** | 0.781 ± 0.022 | 0.777 ± 0.014 | 0.777 ± 0.013 | 0.673 ± 0.030 |

---

### 3. Detailed Holdout Classification Summary Averages

#### Cardiovascular Disease ($N = 1,015$)
- **Positive Class Support:** 124 cases (12.2% test prevalence)
- **Macro Average:** Precision = 0.627, Recall = 0.730, F1-Score = 0.642
- **Weighted Average:** Precision = 0.867, Recall = 0.776, F1-Score = 0.808

#### Type 2 Diabetes ($N = 1,015$)
- **Positive Class Support:** 166 cases (16.4% test prevalence)
- **Macro Average:** Precision = 0.627, Recall = 0.692, F1-Score = 0.637
- **Weighted Average:** Precision = 0.818, Recall = 0.744, F1-Score = 0.769

#### Osteoporosis ($N = 1,015$)
- **Positive Class Support:** 307 cases (30.2% test prevalence)
- **Macro Average:** Precision = 0.684, Recall = 0.717, F1-Score = 0.679
- **Weighted Average:** Precision = 0.758, Recall = 0.696, F1-Score = 0.708

#### Anemia ($N = 1,015$)
- **Positive Class Support:** 53 cases (5.2% test prevalence)
- **Macro Average:** Precision = 0.523, Recall = 0.615, F1-Score = 0.437
- **Weighted Average:** Precision = 0.921, Recall = 0.592, F1-Score = 0.702

---

## 📈 Real CDC NHANES Dataset Summary

- **Source:** CDC National Health and Nutrition Examination Survey (2017–2018 Cycle)
- **Filtered Population:** 5,073 U.S. Adults (Age ≥ 18)
- **Features Extracted:** 36 demographic, physical, and nutritional predictors
- **Target Distribution (Full Dataset $N = 5,073$):**
  - **Osteoporosis:** 1,532 positive / 5,073 total (**30.2%**)
  - **Diabetes:** 829 positive / 5,073 total (**16.3%**)
  - **Cardiovascular Disease:** 621 positive / 5,073 total (**12.2%**)
  - **Anemia:** 263 positive / 5,073 total (**5.2%**)

---

## 🛠️ Automated Real Data Fetcher Script

This repository includes an automated script `src/fetch_real_nhanes.py` that downloads raw SAS Transport (`.xpt`) files directly from CDC servers and merges demographics, body measurements, 24-hour recall nutrient intakes, and doctor-diagnosed medical conditions:

```bash
# Fetch fresh real CDC NHANES data and train models
python -m src.fetch_real_nhanes
python -m src.train
```

---

## 🚀 How to Deploy on Render (Step-by-Step)

Host this application live on **Render** (free web service tier) so it can be accessed from any device.

### Render Deployment Instructions:
1. Sign up or log into **[Render.com](https://render.com/)**.
2. Click **New +** → Select **Web Service**.
3. Select your repository: `ABHI-2802/nutraceutical-disease-risk-prediction`.
4. Configure settings:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:**
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - **Instance Type:** Free
5. Click **Deploy Web Service**.

---

## 💻 Local Quick Start (Mac/Linux)

```bash
# 1. Clone repository
git clone https://github.com/ABHI-2802/nutraceutical-disease-risk-prediction.git
cd nutraceutical-disease-risk-prediction

# 2. Setup virtual environment & dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Download CDC NHANES data & train models
python -m src.fetch_real_nhanes
python -m src.train

# 4. Launch Streamlit UI
streamlit run app.py
```
