import streamlit as st
import joblib
import numpy as np

# Load trained artifacts
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("🩺 Breast Cancer Prediction System")
st.write("Enter tumor measurements to predict whether it is **Benign or Malignant**.")

st.sidebar.header("Tumor Feature Inputs")

input_data = []

for feature in feature_names:
    value = st.sidebar.number_input(
        label=feature,
        min_value=0.0,
        format="%.5f"
    )
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

if st.button("🔍 Predict"):
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("## 🧾 Prediction Result")

    if prediction == 1:
        st.error("🚨 **Malignant Tumor Detected**")
    else:
        st.success("✅ **Benign Tumor Detected**")

    st.markdown("### 📊 Prediction Probability")
    st.write(f"Benign: {probability[0]*100:.2f}%")
    st.write(f"Malignant: {probability[1]*100:.2f}%")

st.caption("⚠️ For educational use only.")

