import streamlit as st
import pandas as pd
import joblib

# Load the trained model and column names
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict whether they may leave the service.")

# Customer information form
with st.form("customer_form"):

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior_citizen = st.selectbox(
        "Senior Citizen", [0, 1]
    )

    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (months)", 0, 72, 12)

    phone_service = st.selectbox(
        "Phone Service", ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    submitted = st.form_submit_button("Predict Churn")

# Prepare customer data after form submission
if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Encode the input using the training column names
    encoded_customer = {}

    numeric_columns = {
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    }

    for column in model_columns:

        if column in numeric_columns:
            encoded_customer[column] = customer[column]

        else:
            feature, category = column.split("_", 1)

            encoded_customer[column] = int(
                customer.get(feature) == category
            )

    input_data = pd.DataFrame(
        [encoded_customer],
        columns=model_columns
    )

    # Predict churn
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The model predicts that this customer may churn.")
    else:
        st.success("The model predicts that this customer may stay.")

    # Display estimated churn probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

        st.write(
            f"Estimated churn probability: {probability:.1%}"
        )