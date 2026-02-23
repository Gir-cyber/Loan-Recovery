import streamlit as st
import joblib
import pandas as pd
from src.data_preprocessing import FeatureBinner


# Load model
model = joblib.load("Pipeline.pkl")

st.title("Loan Default Prediction App")

st.write("Enter applicant details:")

# Example inputs (change according to your dataset columns)

age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Monthly Income", min_value=0)
dependents = st.number_input("Dependents", min_value=0)
debt_ratio = st.number_input("Debt Income Ratio", min_value=0.0)
credit_util = st.number_input("Credit Utilization", min_value=0.0)
open_credit = st.number_input("Open Credit Lines")
past_30 = st.number_input("Past 30 days due", min_value=0.0)
past_60 = st.number_input("Past 60 days due", min_value=0.0)
past_90 = st.number_input("Past 90 days due", min_value=0.0)
real_estate_loan = st.number_input("Real_Estate_Loans", min_value=0.0)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Age": [age],
        "Monthly_Income": [income],
        "Dependents": [dependents],
        "Debt_Income_Ratio": [debt_ratio],
        "Credit_Utilization":[credit_util],
        "Past_Due30_59":[past_30],
        "Open_Credit_Lines":[open_credit],
        "Past_Due90":[past_90],
        "Past_Due60_89":[past_60],
        "Real_Estate_Loans":[real_estate_loan]
        
    })

    

    proba = model.predict_proba(input_df)[0][1]

    st.write(f"Default Probability: {proba:.2%}")


    if proba > 0.6:
        st.error("High Risk of Default")
    else:
        st.success("Low Risk")

