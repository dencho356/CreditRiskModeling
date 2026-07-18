import joblib
import pandas as pd
import streamlit as st


# Load the trained model and the exact category mappings used during training.
model = joblib.load("extra_trees_credit_model")
encoder_columns = ["Sex", "Housing", "Saving accounts", "Checking account"]
encoders = {
    column: joblib.load(f"{column}_encoder.pkl")
    for column in encoder_columns
}
target_encoder = joblib.load("target_encoder.pkl")

st.title("Credit Risk Prediction App")
st.write("Enter the applicant's information to predict their credit risk.")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", encoders["Sex"].classes_)
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", encoders["Housing"].classes_)
saving_accounts = st.selectbox(
    "Saving accounts", encoders["Saving accounts"].classes_
)
checking_account = st.selectbox(
    "Checking account", encoders["Checking account"].classes_
)
credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# Build one model-ready record using the same feature names and order as training.
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
})

if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    risk_label = target_encoder.inverse_transform([prediction])[0]

    if risk_label == "good":
        st.success("The predicted credit risk is: GOOD")
    else:
        st.error("The predicted credit risk is: BAD")
