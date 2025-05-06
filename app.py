import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import google.generativeai as genai
import dotenv
import os

# ────────────────── Load environment variables ──────────────────
dotenv.load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ────────────────── Page Settings ──────────────────
st.set_page_config(page_title="Cardio Risk Predictor + Chatbot", layout="wide")
st.markdown("""
    <style>
        .stAlert {border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.15);}
        .metric-card {border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.15);}
    </style>
""", unsafe_allow_html=True)

# ────────────────── Load Trained Model ──────────────────
@st.cache_resource
def load_model():
    return joblib.load("cardio_pipeline.pkl")

model_rf = load_model()

# ────────────────── Sidebar Inputs ──────────────────
with st.sidebar:
    st.title("📝 Patient Details")

    age = st.slider("Age (years)", 0, 120, 50)
    height = st.slider("Height (cm)", 100, 250, 170)
    weight = st.slider("Weight (kg)", 30, 200, 70)
    systolic = st.slider("Systolic BP (mmHg)", 80, 250, 120)
    diastolic = st.slider("Diastolic BP (mmHg)", 50, 150, 80)

    gluc = st.selectbox("Glucose Category", [1, 2, 3],
        format_func=lambda x: {
            1: "Normal (≤ 99 mg/dL)",
            2: "Above normal (100-125)",
            3: "Well above normal (≥ 126)",
        }[x]
    )

    cholesterol = st.selectbox("Cholesterol Category", [1, 2, 3],
        format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[x]
    )

    gender = st.selectbox("Biological sex", [1, 2],
        format_func=lambda x: {1: "Female", 2: "Male"}[x]
    )

    smoke = st.selectbox("Smokes?", [0, 1])
    alco = st.selectbox("Consumes alcohol?", [0, 1])
    active = st.selectbox("Physically active?", [0, 1])
    predict_btn = st.button("🔮 Predict risk")

# ────────────────── Feature Engineering ──────────────────
def assemble_features():
    bmi = round(weight / (height / 100) ** 2, 2)
    return pd.DataFrame({
        "age_yrs": [age],
        "height": [height],
        "weight": [weight],
        "ap_hi": [systolic],
        "ap_lo": [diastolic],
        "gluc": [gluc],
        "bmi": [bmi],
        "chol_bmi_int": [cholesterol * bmi],
        "gender": [gender],
        "cholesterol": [cholesterol],
        "smoke": [smoke],
        "alco": [alco],
        "active": [active],
        "bp_prehyp": [int((120 <= systolic <= 139) or (80 <= diastolic <= 89))],
        "bp_hyp": [int((systolic >= 140) or (diastolic >= 90))],
        "age_mid": [int(45 <= age <= 59)],
        "age_old": [int(age >= 60)],
        "smoke_alco_int": [smoke * alco],
    })

# ────────────────── Gauge Visualization ──────────────────
def gauge(prob: float):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
    ax.axis("equal"); ax.axis("off")
    ax.add_patch(Wedge((0, 0), 1, 0, 360, width=0.25, facecolor="#ECECEC"))
    colour = "#2ecc71" if prob <= 35 else "#f1c40f" if prob <= 80 else "#e74c3c"
    ax.add_patch(Wedge((0, 0), 1, 90, 90 + prob * 3.6, width=0.25, facecolor=colour))
    ax.add_patch(Circle((0, 0), 0.65, facecolor="white", zorder=3))
    ax.text(0, 0, f"{prob:.1f} %", ha="center", va="center", fontsize=16, weight="bold")
    return fig

# ────────────────── Prediction Panel ──────────────────
if predict_btn:
    features_df = assemble_features()
    risk_prob = model_rf.predict_proba(features_df)[0][1] * 100
    colour = "🟢 Low risk" if risk_prob <= 35 else "🟡 Moderate risk" if risk_prob <= 80 else "🔴 High risk"
    delta = "↓" if risk_prob <= 35 else "→" if risk_prob <= 80 else "↑"
    st.metric(label=colour, value=f"{risk_prob:.1f} %", delta=delta)
    st.caption("Disclaimer: This prediction is not a diagnosis. Consult a healthcare provider for medical advice.")
    with st.expander("Show engineered features"):
        st.dataframe(features_df.T, use_container_width=True)
    csv = features_df.assign(predicted_probability=risk_prob / 100).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download result (CSV)", csv, "cardio_risk_result.csv", "text/csv")
else:
    st.markdown("👉 Use the sidebar to enter details and click **Predict risk**.")

# ────────────────── Chatbot Section ──────────────────
st.divider()
st.subheader("💬 Talk to your Heart Health Assistant")

def fetch_chat_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "user", "parts": (
                f"You are a heart health assistant. Use this user profile to guide your responses:\n"
                f"Age: {age}, Gender: {gender}, Systolic BP: {systolic}, Diastolic BP: {diastolic}, "
                f"Cholesterol Level: {cholesterol}, Glucose Level: {gluc}, Smoker: {smoke}, "
                f"Alcohol: {alco}, Physically Active: {active}.\n"
                f"Give advice on lifestyle, prevention, and when to see a doctor. Be friendly and supportive."
            )}
        ]
    return st.session_state["messages"]

user_input = st.chat_input("Ask a heart-related question...")
if user_input:
    messages = fetch_chat_history()
    messages.append({"role": "user", "parts": user_input})

    try:
        response = model.generate_content(messages)
        reply = response.candidates[0].content.parts[0].text
    except Exception as e:
        reply = f"⚠️ Error: {e}"

    messages.append({"role": "model", "parts": reply})

    for message in messages:
        if message["role"] == "user" and "profile" not in message["parts"]:
            st.markdown(f"**You:** {message['parts']}")
        elif message["role"] == "model":
            st.markdown(f"**Heart Health Assistant:** {message['parts']}")
