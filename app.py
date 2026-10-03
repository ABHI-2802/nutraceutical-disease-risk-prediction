from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.pipeline import TARGETS, features_for_target

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Nutraceutical Health Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #08111f 0%, #0b1220 42%, #0e1728 100%);
        color: #e8eef7;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 17, 31, 0.80);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07101d 0%, #0b1524 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 28px 30px 24px 30px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(17, 35, 58, 0.98), rgba(12, 24, 42, 0.90));
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22);
        margin-bottom: 18px;
    }

    .eyebrow {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #8fb8ff;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.20);
        padding: 6px 10px;
        border-radius: 999px;
        margin-bottom: 10px;
    }

    .hero h1 {
        font-size: 2.25rem;
        line-height: 1.15;
        margin: 0 0 10px 0;
        color: #f8fafc;
    }

    .hero p {
        color: #b8c4d6;
        font-size: 1.02rem;
        margin: 0;
        max-width: 920px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.20rem;
        font-weight: 750;
        margin: 4px 0 12px 2px;
    }

    .subtle {
        color: #91a0b5;
        font-size: 0.88rem;
    }

    .panel {
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: rgba(15, 27, 46, 0.78);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .metric-card {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 17px;
        padding: 16px 17px;
        background: rgba(15, 27, 46, 0.80);
        min-height: 105px;
    }

    .metric-label {
        color: #93a2b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 6px;
    }

    .metric-help {
        color: #7f8da2;
        font-size: 0.78rem;
        margin-top: 3px;
    }

    .risk-card {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 18px;
        background: linear-gradient(145deg, rgba(19, 34, 55, 0.95), rgba(11, 22, 38, 0.95));
        height: 100%;
    }

    .risk-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .risk-name {
        font-size: 1rem;
        font-weight: 750;
        color: #e8eef7;
    }

    .risk-score {
        font-size: 1.65rem;
        font-weight: 850;
        color: #f8fafc;
    }

    .bar-bg {
        background: #172235;
        height: 9px;
        border-radius: 999px;
        overflow: hidden;
        margin: 10px 0 9px 0;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #60a5fa, #8b5cf6);
    }

    .status {
        color: #9fb0c7;
        font-size: 0.80rem;
    }

    .pill {
        display: inline-block;
        color: #c9d6e9;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(148, 163, 184, 0.07);
        border-radius: 999px;
        padding: 5px 9px;
        font-size: 0.72rem;
        margin: 3px 4px 0 0;
    }

    .side-title {
        color: #f8fafc;
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 3px;
    }

    .side-copy {
        color: #91a0b5;
        font-size: 0.83rem;
        line-height: 1.55;
    }

    .footer {
        text-align: center;
        color: #66758b;
        font-size: 0.75rem;
        padding-top: 22px;
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 1.3rem;
        background: rgba(15, 27, 46, 0.62);
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 11px;
        font-weight: 750;
        min-height: 46px;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px;
    }

    .disclaimer {
        border-left: 3px solid #64748b;
        padding: 10px 12px;
        background: rgba(100, 116, 139, 0.07);
        color: #9aa9bd;
        border-radius: 0 10px 10px 0;
        font-size: 0.80rem;
        line-height: 1.55;
        margin: 8px 0 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Model helpers
# -----------------------------
models = {t: ROOT / f"models/{t.lower()}_pipeline.joblib" for t in TARGETS}

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


def disease_label(target: str) -> str:
    labels = {
        "Anemia": "Anemia",
        "Osteoporosis": "Osteoporosis",
        "Diabetes": "Type 2 Diabetes",
        "Cardio": "Cardiovascular Disease",
    }
    return labels.get(target, target)


def score_state(score: float) -> str:
    # UI descriptor only. It is intentionally not called a clinical risk category.
    if score < 0.33:
        return "Lower model score"
    if score < 0.67:
        return "Intermediate model score"
    return "Higher model score"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown('<div class="side-title">🧬 Health Intelligence</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="side-copy">Nutraceutical & lifestyle based multi-disease prediction research prototype.</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Pipeline**")
for item in [
    "Profile input",
    "Feature engineering",
    "Disease-specific models",
    "Model-score comparison",
]:
    st.sidebar.markdown(f"<span class='pill'>{item}</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model availability**")
for target, path in models.items():
    icon = "✅" if path.exists() else "⏳"
    st.sidebar.write(f"{icon} {disease_label(target)}")

st.sidebar.markdown("---")
st.sidebar.caption("Built for academic research and portfolio demonstration. Not a clinical diagnostic system.")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">RESEARCH PROTOTYPE · MULTI-DISEASE ML</div>
        <h1>Nutraceutical Health Intelligence</h1>
        <p>
            Explore how demographic, lifestyle, body-measurement and nutraceutical features
            are translated into disease-specific model scores.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer"><b>Important:</b> This application is an educational research prototype. '
    'The outputs are model scores generated from the supplied dataset and should not be interpreted as '
    'diagnosis, medical advice, or calibrated personal disease probabilities.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Profile input
# -----------------------------
st.markdown('<div class="section-title">1. Build a health profile</div>', unsafe_allow_html=True)

with st.form("profile_form"):
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("**Demographics & body metrics**")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=40, step=1)
        gender = st.selectbox("Gender", ["F", "M"])
        height = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=0.5)
        weight = st.number_input("Weight (kg)", min_value=25.0, max_value=250.0, value=70.0, step=0.5)
        activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])

    with c2:
        st.markdown("**Diet & core nutrients**")
        diet = st.selectbox("Diet quality", ["Poor", "Average", "Good", "Excellent"])
        vitamin_d = st.number_input("Vitamin D (dataset units)", 0.0, 200.0, 25.0, 0.5)
        iron = st.number_input("Iron (dataset units)", 0.0, 100.0, 15.0, 0.5)
        calcium = st.number_input("Calcium (dataset units)", 0.0, 3000.0, 900.0, 10.0)
        b12 = st.number_input("Vitamin B12 (dataset units)", 0.0, 20.0, 2.5, 0.1)

    with c3:
        st.markdown("**Additional nutrients**")
        omega = st.number_input("Omega-3 (dataset units)", 0.0, 20.0, 1.5, 0.1)
        zinc = st.number_input("Zinc (dataset units)", 0.0, 50.0, 12.0, 0.5)
        magnesium = st.number_input("Magnesium (dataset units)", 0.0, 1000.0, 300.0, 5.0)
        protein = st.number_input("Protein (dataset units)", 0.0, 250.0, 65.0, 1.0)
        st.caption("Use the same units/definitions used in the training dataset.")

    submitted = st.form_submit_button("🔎 Generate disease model scores", use_container_width=True)

# -----------------------------
# Results
# -----------------------------
if submitted:
    row = pd.DataFrame([
        {
            "Age": age,
            "Gender": gender,
            "Height": height,
            "Weight": weight,
            "Physical_Activity": activity,
            "Diet_Quality": diet,
            "Vitamin_D": vitamin_d,
            "Iron": iron,
            "Calcium": calcium,
            "Vitamin_B12": b12,
            "Omega_3": omega,
            "Zinc": zinc,
            "Magnesium": magnesium,
            "Protein": protein,
        }
    ])

    bmi = weight / ((height / 100) ** 2)

    st.markdown('<div class="section-title">2. Profile summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    cards = [
        ("BMI", f"{bmi:.1f}", "derived from height + weight"),
        ("Activity", activity, "profile input"),
        ("Diet quality", diet, "profile input"),
        ("Nutrients entered", "8 / 8", "vitamins, minerals & protein"),
    ]
    for col, (label, value, help_text) in zip([m1, m2, m3, m4], cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-help">{help_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title" style="margin-top:22px;">3. Disease model outputs</div>', unsafe_allow_html=True)

    available = []
    missing = []
    for target in TARGETS:
        if models[target].exists():
            available.append(target)
        else:
            missing.append(target)

    if missing:
        missing_names = ", ".join(disease_label(t) for t in missing)
        st.warning(
            f"Model files not found for: {missing_names}. Run `python -m src.train` to train the full model set."
        )

    if available:
        cols = st.columns(len(available), gap="medium")
        outputs = []
        for col, target in zip(cols, available):
            try:
                pipe = load_model(str(models[target]))
                X = features_for_target(row, target)
                score = float(pipe.predict_proba(X)[0, 1])
                outputs.append((target, score))
                with col:
                    pct = score * 100
                    st.markdown(
                        f"""
                        <div class="risk-card">
                            <div class="risk-top">
                                <div class="risk-name">{disease_label(target)}</div>
                                <div class="risk-score">{pct:.1f}%</div>
                            </div>
                            <div class="bar-bg">
                                <div class="bar-fill" style="width:{max(0.0, min(100.0, pct))}%;"></div>
                            </div>
                            <div class="status">{score_state(score)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                with col:
                    st.error(f"Could not load {disease_label(target)} model: {exc}")

        if outputs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**How to read this screen**")
            st.caption(
                "Each percentage is the positive-class score returned by that disease-specific model. "
                "It is useful for comparing model outputs for a supplied profile, but it is not a clinical probability."
            )

            result_df = pd.DataFrame(
                [{"Disease": disease_label(t), "Model score": s} for t, s in outputs]
            ).sort_values("Model score", ascending=False)

            with st.expander("View numeric output table"):
                st.dataframe(
                    result_df.style.format({"Model score": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

st.markdown(
    '<div class="footer">Nutraceutical Disease Risk Prediction · Academic ML research prototype · Streamlit</div>',
    unsafe_allow_html=True,
)
