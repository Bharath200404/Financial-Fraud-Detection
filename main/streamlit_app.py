import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# PAGE CONFIG

st.set_page_config(
    page_title="E-Commerce Fraud Detection Platform",
    layout="wide"
)


# LOAD MODEL

model = joblib.load("fraud_model.pkl")


# TITLE

st.title("🛡️ Intelligent E-Commerce Fraud Detection & Risk Analytics Platform")

st.markdown("""
Detect suspicious e-commerce transactions using machine learning and behavioral analytics.
""")


# SIDEBAR INPUTS

st.sidebar.header("Transaction Details")

transaction_amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["PayPal", "bank transfer", "credit card", "debit card"]
)

product_category = st.sidebar.selectbox(
    "Product Category",
    ["electronics", "clothing", "home & garden", "health & beauty", "toys & games"]
)

quantity = st.sidebar.slider(
    "Quantity",
    1,
    10,
    1
)

customer_age = st.sidebar.slider(
    "Customer Age",
    18,
    80,
    30
)

customer_location = st.sidebar.selectbox(
    "Customer Location",
    ["Amandaborough", "East Timothy", "Davismouth", "Lynnberg"]
)

device_used = st.sidebar.selectbox(
    "Device Used",
    ["desktop", "mobile", "tablet"]
)

account_age_days = st.sidebar.slider(
    "Account Age (Days)",
    1,
    365,
    100
)

transaction_hour = st.sidebar.slider(
    "Transaction Hour",
    0,
    23,
    12
)

address_mismatch_input = st.sidebar.selectbox(
    "Address Mismatch",
    ["No", "Yes"]
)


# FEATURE ENGINEERING

night_transaction = 1 if transaction_hour <= 5 else 0

new_account = 1 if account_age_days < 30 else 0

high_value_transaction = 1 if transaction_amount > 500 else 0

address_mismatch = 1 if address_mismatch_input == "Yes" else 0


# ENCODING MAPS

payment_mapping = {
    "PayPal": 0,
    "bank transfer": 1,
    "credit card": 2,
    "debit card": 3
}

product_mapping = {
    "clothing": 0,
    "electronics": 1,
    "health & beauty": 2,
    "home & garden": 3,
    "toys & games": 4
}

device_mapping = {
    "desktop": 0,
    "mobile": 1,
    "tablet": 2
}

location_mapping = {
    "Amandaborough": 0,
    "East Timothy": 1,
    "Davismouth": 2,
    "Lynnberg": 3
}

# ---------------------------------------------------
# INPUT DATAFRAME
# ---------------------------------------------------
input_data = pd.DataFrame({
    'Transaction Amount': [transaction_amount],
    'Payment Method': [payment_mapping[payment_method]],
    'Product Category': [product_mapping[product_category]],
    'Quantity': [quantity],
    'Customer Age': [customer_age],
    'Customer Location': [location_mapping[customer_location]],
    'Device Used': [device_mapping[device_used]],
    'Account Age Days': [account_age_days],
    'Transaction Hour': [transaction_hour],
    'night_transaction': [night_transaction],
    'new_account': [new_account],
    'high_value_transaction': [high_value_transaction],
    'address_mismatch': [address_mismatch]
})


# PREDICTION

if st.sidebar.button("Predict Fraud Risk"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    
    # RISK CATEGORY
    
    if probability < 0.30:
        risk = "Low Risk"
        risk_color = "green"

    elif probability < 0.70:
        risk = "Medium Risk"
        risk_color = "orange"

    else:
        risk = "High Risk"
        risk_color = "red"

    # DISPLAY RESULT

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")

    else:
        st.success("✅ Legitimate Transaction")

    st.markdown(f"### Fraud Probability: `{probability:.2%}`")

    st.markdown(f"### Risk Level: :{risk_color}[{risk}]")

# DASHBOARD SECTION

st.markdown("---")

st.subheader("Fraud Analytics Dashboard")

dashboard_data = pd.DataFrame({
    "Payment Method": [
        "PayPal",
        "Bank Transfer",
        "Credit Card",
        "Debit Card"
    ],
    "Fraud Rate": [
        5.02,
        5.04,
        4.97,
        5.01
    ]
})

fig = px.bar(
    dashboard_data,
    x="Payment Method",
    y="Fraud Rate",
    title="Fraud Rate by Payment Method"
)

st.plotly_chart(fig, use_container_width=True)


# BUSINESS INSIGHTS

st.markdown("---")

st.subheader("Business Insights")

st.info("""
• Late-night transactions (12 AM – 5 AM) show significantly higher fraud probability.

• Newly created accounts exhibit elevated fraud risk.

• High-value transactions are more likely to be suspicious.

• Mobile-based purchases show slightly higher fraud occurrence.

• Address mismatches may indicate potentially risky transactions.
""")