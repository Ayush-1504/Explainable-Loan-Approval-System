import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Explainable Loan Approval System",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Explainable Loan Approval System")

st.markdown("""
This application predicts whether a loan will be approved or not based on applicant details.
""")

st.divider()

# User Inputs

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    [0, 1, 2, 3]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=1000,
    max_value=100000,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    max_value=50000,
    value=2000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=10,
    max_value=1000,
    value=150
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=12,
    max_value=480,
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    ["Good", "Bad"]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# Divider
st.divider()

# Convert categorical values into numerical format

gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

education = 0 if education == "Graduate" else 1

self_employed = 1 if self_employed == "Yes" else 0

credit_history = 1 if credit_history == "Good" else 0

property_area_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

property_area = property_area_map[property_area]

# Prediction Button

if st.button("Predict Loan Approval"):

    # Create input array
    input_data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    # Output Result

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    # Show probability
    st.subheader("Prediction Confidence")

    st.progress(int(probability * 100))

    st.write(f"Approval Probability: **{probability:.2%}**")

    # Feature Summary
    st.subheader("Applicant Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self Employed",
            "Applicant Income",
            "Coapplicant Income",
            "Loan Amount",
            "Loan Amount Term",
            "Credit History",
            "Property Area"
        ],
        "Value": [
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area
        ]
    })

    st.dataframe(summary, use_container_width=True)