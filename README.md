# Nutraceutical-Based Multi-Disease Risk Prediction

An end-to-end educational ML project that estimates dataset-pattern risk scores for **Anemia, Osteoporosis, Type 2 Diabetes, and Cardiovascular disease** from demographic, anthropometric, lifestyle, and nutrient-intake features.

> **Important:** This is a research/demo model, not a diagnostic or clinical decision tool. Predictions reflect patterns in the supplied dataset and must not be used to diagnose, treat, or rule out disease. The dataset's provenance and label-generation process must be independently verified before any scientific or clinical claim.

## What is included
- Data audit and target/feature checks
- Leakage-conscious preprocessing and feature engineering
- Separate binary classifier for each of the four outcomes
- Stratified holdout test set + cross-validation on training data
- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC/PR curves
- Baseline and tree-based model comparison
- Feature permutation importance, serialized pipelines, Streamlit app
- SQL schema/queries, smoke tests, limitations and reproducibility instructions

## Dataset audit caveat
The supplied CSV has 1,000 rows, 22 columns, no missing cells, and no exact duplicate rows at initial inspection. Target counts are written to `reports/metrics/data_audit.json` when the audit runs. Nutrient status fields (`VitD_Status`, `Iron_Status`, `Calcium_Status`) are excluded from predictors because they are deterministic/derived summaries that may reveal label construction. All four disease labels are retained as separate targets. The dataset is small and appears highly structured; high scores must not be interpreted as clinical validity. Check source, consent/license, label definitions, and whether labels were rule-generated before publishing results.

## Quick start (Mac/Linux)
```bash
cd Nutraceutical_Disease_Risk_Prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```
Training creates fitted pipelines under `models/` and reports under `reports/`. Run training before launching the app.

## Methodology
1. The four labels are modeled independently; each row can have multiple positive labels.
2. The target column and all other disease-label columns are excluded from the feature matrix for that target.
3. Nutrient status summary columns are excluded from predictors to reduce direct target-rule leakage.
4. BMI is recalculated from height and weight where valid. Raw BMI is not used as a predictor; the calculated BMI is.
5. A stratified 80/20 split is made once per target. Hyperparameters are not tuned on the test set. Cross-validation occurs only on the training portion.
6. Candidate models: DummyClassifier (reference), Logistic Regression, Random Forest, HistGradientBoosting. Models are compared by mean training-CV ROC-AUC where defined; the holdout test set is evaluated once for the selected candidate.
7. For imbalanced labels, PR-AUC and recall are reported alongside ROC-AUC and accuracy. Threshold 0.5 is used for the primary reported confusion matrix; it is not claimed to be clinically optimal.

## Feature set
Age, gender, height, weight, physical activity, diet quality, nutrient intakes (Vitamin D, Iron, Calcium, B12, Omega-3, Zinc, Magnesium, Protein), and engineered BMI/BMI category. Status fields and other disease labels are never predictors.

## Project structure
```text
data/raw/             supplied dataset
src/                  audit, training, shared pipeline
models/               fitted pipelines and metadata (generated)
reports/metrics/      audit + evaluation metrics (generated)
reports/figures/      diagnostic plots (generated)
sql/                  database schema and example queries
tests/                lightweight checks
app.py                Streamlit demo
```

## Responsible interpretation
- Do not claim this predicts real-world disease risk unless labels are clinically validated and the model is externally validated.
- Do not report accuracy alone. Discuss class imbalance, sensitivity/recall, precision, PR-AUC, calibration, and confidence intervals if appropriate.
- The input features do not include clinical measurements such as blood counts, bone density, glucose/HbA1c, blood pressure, or lipid profile; therefore this project cannot establish clinical diagnosis.
- For CV, describe this as a prototype built on a supplied structured dataset, and state its limitations transparently.
