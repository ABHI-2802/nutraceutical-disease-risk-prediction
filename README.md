# Nutraceutical-Based Multi-Disease Risk Prediction

An end-to-end machine learning pipeline and interactive Streamlit web application that estimates disease risk scores for **Anemia, Osteoporosis, Type 2 Diabetes, and Cardiovascular Disease** from demographic, physical measurement, and 24-hour nutrient intake data.

> 🌐 **Real Dataset:** Trained on **4,988 real adult participants** from the **CDC NHANES (National Health and Nutrition Examination Survey)** 2017–2018 dataset.

> ⚠️ **Important Disclaimer:** This repository is an educational research and portfolio prototype. Predictions reflect statistical patterns within survey and nutrient data and **must not be used for medical diagnosis or clinical decision-making**.

---

## 📊 Real-World CDC NHANES Model Benchmarks (N = 4,988)

The models were evaluated on an isolated **20% stratified holdout test set (998 adult participants)**. Prior to holdout testing, 5-fold cross-validation was conducted across candidate algorithms on 3,990 training rows.

### 1. Holdout Test Set Evaluation (N = 998 Adult Participants)

| Disease Target | Selected Model | Class Prevalence (Test Set) | Accuracy | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Osteoporosis** | `LogisticRegression` | 30.0% (299 / 998) | **72.9%** | **73.2%** | 0.491 | **0.588** | **0.771** | 0.560 | 0.187 |
| **Type 2 Diabetes** | `LogisticRegression` | 16.2% (161 / 998) | **71.2%** | **75.2%** | 0.316 | **0.446** | **0.774** | 0.383 | 0.198 |
| **Cardiovascular** | `LogisticRegression` | 8.2% (82 / 998) | **70.9%** | **78.0%** | 0.197 | **0.316** | **0.819** | 0.249 | 0.201 |
| **Anemia** | `HistGradientBoosting` | 5.1% (51 / 998) | **94.9%** | **0.0%** | 0.000 | **0.000** | **0.680** | 0.113 | 0.048 |

---

### 2. 5-Fold Cross-Validation Model Comparison (Mean ROC-AUC ± Std)

| Candidate Model | Osteoporosis CV ROC-AUC | Diabetes CV ROC-AUC | Cardio CV ROC-AUC | Anemia CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **DummyPrior (Baseline)** | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| **Logistic Regression (Selected for 3 targets)** | **0.778 ± 0.008** | **0.775 ± 0.008** | **0.816 ± 0.013** | 0.677 ± 0.017 |
| **RandomForest** | 0.748 ± 0.011 | 0.730 ± 0.016 | 0.774 ± 0.019 | 0.655 ± 0.021 |
| **ExtraTrees** | 0.751 ± 0.010 | 0.733 ± 0.016 | 0.778 ± 0.018 | 0.659 ± 0.020 |
| **HistGradientBoosting** | 0.750 ± 0.011 | 0.737 ± 0.015 | 0.781 ± 0.017 | **0.678 ± 0.019** |

---

## 📈 Real CDC NHANES Dataset Statistics

- **Source:** CDC National Health and Nutrition Examination Survey (2017–2018 Cycle)
- **Filtered Population:** 4,988 U.S. Adults (Age ≥ 18)
- **Missing Data / Duplicates:** Handled via median/mode imputation & clean deduplication
- **Disease Prevalence in Sample:**
  - **Osteoporosis:** 1,494 positive / 4,988 total (**30.0%**)
  - **Diabetes:** 807 positive / 4,988 total (**16.2%**)
  - **Cardiovascular Disease:** 408 positive / 4,988 total (**8.2%**)
  - **Anemia:** 254 positive / 4,988 total (**5.1%**)

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
