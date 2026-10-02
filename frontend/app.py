import os
import requests
import streamlit as st
from dotenv import load_dotenv

# load .env to env vars
load_dotenv()

API_URL = os.getenv("API_URL")

st.set_page_config(
    page_title="HeartRisk.ML",
    page_icon="⚕️",
    layout="centered"
)


# a bit of styling (not necessary)
st.markdown("""
<style>

.block-container {padding-top: 2rem; max-width: 900px;}
.result-card {
    position: relative;
    padding: 1.2rem; border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.3);
    text-align: center;
}
.result-label {font-size: 2.2rem; font-weight: 700;}
.result-pct {
    position: absolute; top: 10px; right: 14px;
    font-size: 1.1rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# the labels have been verified against the documentation of the exact heart.csv dataset used
SEX = {"Female": 0, "Male": 1}
CP = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
YESNO = {"No": 0, "Yes": 1}
ECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}


# risk probability at the bottom right
def risk_band(p: float):
    if p < 0.30:
        return "Low", "#3DD68C"
    if p < 0.60:
        return "Moderate", "#F5A524"
    return "High", "#E5484D"



st.title("🫀 HeartRisk.ML")
st.caption("Heart disease risk estimation from clinical measurements")

with st.sidebar:
    st.header("About this model")
    st.write("Random forest trained on a heart disease dataset, tuned with "
             "grouped cross-validation to avoid duplicate-row leakage.")
    st.warning("For learning and demonstration only. Not for clinical use.")

st.subheader("Enter patient details and click **Estimate risk**")


# inputs are grouped into cols
col1, col2, col3  = st.tabs(["Demographics", "Vitals", "Exercise & ECG"])

with col1:
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 1, 120, 52)
    sex_label = c2.selectbox("Sex", list(SEX), index=0)

with col2:
    c1, c2, c3 = st.columns(3)
    trestbps = c1.number_input("Resting blood pressure (mm Hg)", 0, 250, 125)
    chol = c2.number_input("Cholesterol (mg/dl)", 0, 600, 212)
    fbs_label = c3.selectbox("Fasting blood sugar > 120 mg/dl", list(YESNO), index=0)

with col3:
    c1, c2, c3 = st.columns(3)
    cp_label = c1.selectbox("Chest pain type", list(CP), index=0)
    restecg_label = c2.selectbox("Resting ECG", list(ECG), index=1)
    thalach = c3.number_input("Max heart rate (bpm)", 0, 250, 168)
    exang_label = c1.selectbox("Exercise-induced angina", list(YESNO), index=0)
    oldpeak = c2.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope_label = c3.selectbox("ST slope", list(SLOPE), index=2)
    ca = c1.number_input("Major vessels (ca)", 0, 4, 0)
    thal = c2.number_input("Thal (dataset code)", 0, 3, 2)

# predict button (estimate risk)
if st.button("🔍 Estimate risk", type="primary", use_container_width=True):
    input_data = {
        "age": age,
        "sex": SEX[sex_label],
        "cp": CP[cp_label],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": YESNO[fbs_label],
        "restecg": ECG[restecg_label],
        "thalach": thalach,
        "exang": YESNO[exang_label],
        "oldpeak": oldpeak,
        "slope": SLOPE[slope_label],
        "ca": ca,
        "thal": thal,
    }

    try:
        response = requests.post(API_URL, json=input_data, timeout=10)
    except requests.exceptions.RequestException:
        st.error("Could not reach the prediction service. Try again later.")
        st.stop()

    if response.status_code != 200:
        st.error("Oops! Something went wrong. Try again later...")
    else:
        result = response.json()
        probability = result["probability"]

        label, color = risk_band(probability)
        pct = probability * 100

        st.divider()
        left, right = st.columns([3, 2])
        right.markdown(
            f"<div class='result-card'>"
            f"<div class='result-pct' style='color:{color}'>{pct:.0f}%</div>"
            f"Estimated risk"
            f"<div class='result-label' style='color:{color}'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        right.caption("Not a diagnosis. Discuss any health concerns with a clinician.")