import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model


# load model
model = load_model("models/churn_model")

# start writing streamlit

# title
st.title("Customer Churn Prediction Web App")

# indicate to the user to enter details
st.write("Enter customer details to predict churn")

# User Inputs
tenure = st.number_input("Tenure (months)", min_value=0)
monthlycharges = st.number_input("Monthly Charges", min_value=0.0)
totalcharges = st.number_input("Total Charges", min_value=0.0)

phone_service = st.selectbox("Phone Service", ["Yes","No"])

contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes","No"])
payment_method = st.selectbox("Payment Method", 
                               [
                                "Electronic check",
                                "Mailed check",
                                "Bank transfer (automatic)",
                                "Credit card (automatic)"
                                
                                ])

# create input dataframe
input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthlycharges ,
    "TotalCharges": totalcharges,
    "PhoneService":phone_service,
    "Contract":contract,
    "PaperlessBilling":paperless_billing,
    "PaymentMethod":payment_method
}])


# predict
if st.button("Predict Churn"):
    prediction = predict_model(model, data = input_data)
    result = prediction["prediction_label"][0]
    if result == "Yes":
        st.error(" Customer is likely to churn")

    else:
        st.error(" Customer is not likely to churn")