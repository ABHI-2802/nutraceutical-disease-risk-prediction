# Nutraceutical-Based Multi-Disease Risk Prediction

An end-to-end educational machine learning pipeline and interactive Streamlit web application that estimates dataset-pattern risk scores for **Anemia, Osteoporosis, Type 2 Diabetes, and Cardiovascular Disease** from demographic, anthropometric, lifestyle, and nutrient-intake features.

> ⚠️ **Important Disclaimer:** This repository is an educational research and portfolio prototype. Predictions reflect patterns present within the supplied dataset and **must not be used to diagnose, treat, or rule out disease**. The dataset's provenance and label-generation process should be independently verified prior to any scientific or clinical application.

---

## 📊 Comprehensive Model Performance & Evaluation Metrics

All models were evaluated on a strict **200-row (20%) stratified holdout test set** that was completely isolated from feature processing and model selection. Prior to holdout evaluation, candidates were compared using **5-Fold Cross-Validation** on the 800-row training set.

### 1. Final Holdout Test Set Evaluation (N = 200)

| Disease Target | Selected Model | Test Set Positive Rate | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anemia** | `RandomForest` | 37.5% (75 / 200) | **100.0%** (1.000) | 1.000 | 1.000 | **1.000** | **1.000** | 1.000 | 0.0216 |
| **Osteoporosis** | `RandomForest` | 49.5% (99 / 200) | **99.5%** (0.995) | 0.990 | 1.000 | **0.995** | **1.000** | 1.000 | 0.0272 |
| **Type 2 Diabetes** | `RandomForest` | 44.5% (89 / 200) | **100.0%** (1.000) | 1.000 | 1.000 | **1.000** | **1.000** | 1.000 | 0.0042 |
| **Cardiovascular** | `RandomForest` | 13.0% (26 / 200) | **99.0%** (0.990) | 1.000 | 0.923 | **0.960** | **1.000** | 1.000 | 0.0118 |

---

### 2. 5-Fold Cross-Validation Model Comparison (Mean ROC-AUC ± Std)

During model exploration, 5 distinct algorithms were benchmarked across the training set (`N = 800`). `RandomForest` achieved top performance across all targets.

| Candidate Model | Anemia CV ROC-AUC | Osteoporosis CV ROC-AUC | Diabetes CV ROC-AUC | Cardio CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **DummyPrior (Baseline)** | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| **Logistic Regression** | 0.893 ± 0.027 | 0.933 ± 0.014 | 0.998 ± 0.002 | 0.981 ± 0.005 |
| **ExtraTrees** | 0.974 ± 0.011 | 0.986 ± 0.003 | 0.998 ± 0.002 | 0.988 ± 0.006 |
| **HistGradientBoosting** | 1.000 ± 0.000 | 0.999 ± 0.003 | 1.000 ± 0.000 | 0.997 ± 0.006 |
| **RandomForest (Selected)** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |

---

### 3. Top Key Permutation Feature Importance

Permutation importance highlights the most influential predictor variables for each disease target:

- **Anemia**: `Iron` (+0.321 mean importance drop), `Vitamin_B12` (+0.184)
- **Osteoporosis**: `Calcium` (+0.393 mean importance drop), `Vitamin_D` (+0.175)
- **Type 2 Diabetes**: `BMI_calc` (+0.143 mean importance drop), `BMI_Category` (+0.0002)
- **Cardiovascular Disease**: `Omega_3` (+0.189 mean importance drop), `BMI_calc` (+0.031)

---

## 📈 Dataset Audit Summary

- **Total Sample Size:** 1,000 rows × 22 columns
- **Missing Values:** 0 null cells across all columns
- **Exact Duplicates:** 0 duplicate rows
- **Target Distribution (Full Dataset N=1,000):**
  - **Anemia:** 376 positive (37.6%), 624 negative (62.4%)
  - **Osteoporosis:** 496 positive (49.6%), 504 negative (50.4%)
  - **Diabetes:** 447 positive (44.7%), 553 negative (553.3%)
  - **Cardiovascular:** 130 positive (13.0%), 870 negative (87.0%)
- **Data Safeguards:** Deterministic nutrient status summary fields (`VitD_Status`, `Iron_Status`, `Calcium_Status`) and other target disease labels are strictly excluded from predictor matrices to prevent data leakage.

---

## 🚀 How to Deploy on Render (Step-by-Step)

You can host this application live on **Render** (free web service tier) so it can be accessed from any device (phone, laptop, tablet).

### Option 1: Direct Deployment via Render Dashboard (Recommended)

1. Sign up or log into **[Render.com](https://render.com/)**.
2. Click **New +** → Select **Web Service**.
3. Connect your GitHub account and select the repository:
   `ABHI-2802/nutraceutical-disease-risk-prediction`
4. Configure the Web Service settings:
   - **Name:** `nutraceutical-risk-prediction`
   - **Region:** Choose closest region (e.g., Singapore, Frankfurt, Oregon)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:**
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
   - **Instance Type:** Free
5. Click **Deploy Web Service**.

### Option 2: Render Blueprint Deployment

This repository includes a pre-configured `render.yaml` file.
1. On Render, click **New +** → Select **Blueprint**.
2. Select `ABHI-2802/nutraceutical-disease-risk-prediction`.
3. Render will automatically read `render.yaml` and provision the Web Service.

Once deployed, Render provides a public HTTPS link (e.g., `https://nutraceutical-risk-prediction.onrender.com`) that works on **any device, anywhere in the world**.

---

## 💻 Local Quick Start (Mac/Linux)

```bash
# 1. Clone the repository
git clone https://github.com/ABHI-2802/nutraceutical-disease-risk-prediction.git
cd nutraceutical-disease-risk-prediction

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install exact requirements
pip install -r requirements.txt

# 4. Train models and generate metrics
python -m src.train

# 5. Launch Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure

```text
├── data/
│   └── raw/                   Raw dataset (1,000 rows × 22 columns)
├── models/                    Serialized scikit-learn pipeline artifacts (.joblib)
├── reports/
│   ├── figures/               Confusion matrices, ROC, and PR curves
│   └── metrics/               Model evaluations, CV benchmarks, data audit JSONs
├── src/
│   ├── audit.py               Data quality audit module
│   ├── pipeline.py            Preprocessing and model architecture definitions
│   └── train.py               Model training and cross-validation execution
├── app.py                     Interactive Streamlit web application
├── render.yaml                Render deployment blueprint configuration
├── requirements.txt           Pinned Python dependency specifications
└── README.md                  Project documentation & empirical metrics
```

---

## 🔬 Responsible Interpretation & Limitations

- **Dataset Constraints:** The model was trained on a small, highly structured 1,000-row synthetic/demo dataset. Perfect or near-perfect evaluation scores (e.g. 100% accuracy) reflect dataset structure rather than clinical real-world performance.
- **Clinical Relevance:** Input features do not include diagnostic lab values (e.g., serum ferritin, bone mineral density, HbA1c, lipid panel). This application is strictly an educational ML prototype.
