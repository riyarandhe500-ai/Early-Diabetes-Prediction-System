import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🩺 Early Diabetes Prediction System")

st.write("Enter the patient's health details below.")


# ------------------------------------
# User Input Fields
# ------------------------------------

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)

glucose = st.number_input("Glucose", min_value=0)

blood_pressure = st.number_input("Blood Pressure", min_value=0)

skin_thickness = st.number_input("Skin Thickness", min_value=0)

insulin = st.number_input("Insulin", min_value=0)

bmi = st.number_input("BMI", min_value=0.0, format="%.2f")

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    format="%.3f"
)

age = st.number_input("Age", min_value=1, step=1)

# Predict Button
if st.button("Predict Diabetes"):

    # Collect user input
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])

    # Scale the input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)

    # Display Result
    if prediction[0] == 1:
        st.error("🔴 The person is Diabetic.")
    else:
        st.success("🟢 The person is Non-Diabetic.")

        