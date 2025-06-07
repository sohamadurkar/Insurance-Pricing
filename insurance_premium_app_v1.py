import streamlit as st
import pandas as pd
import joblib

# Load model, encoders, and training column names
rf_model = joblib.load("rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Car Insurance Premium Predictor")

# Use encoder class lists to populate dropdowns
#INSR_TYPE_options = label_encoders['INSR_TYPE'].classes_
MAKE_options = label_encoders['MAKE'].classes_
USAGE_options = label_encoders['USAGE'].classes_

# UI inputs
SEX = st.selectbox("Sex", [0, 1])
EFFECTIVE_YR = st.number_input("Effective Year", min_value=2000, max_value=2025)
#INSR_TYPE = st.selectbox("Insurance Type", INSR_TYPE_options)
INSR_TYPE = st.selectbox("Insurance Type", [0, 1, 2, 3])
INSURED_VALUE = st.number_input("Insured Value", min_value=0)
MAKE = st.selectbox("Make", MAKE_options)
USAGE = st.selectbox("Usage", USAGE_options)
CLAIM_PAID_FLAG = st.selectbox("Claim Paid Flag", [0, 1])

# Construct input DataFrame
X_input = pd.DataFrame({
    'SEX': [SEX],
    'EFFECTIVE_YR': [EFFECTIVE_YR],
    'INSR_TYPE': [INSR_TYPE],
    'INSURED_VALUE': [INSURED_VALUE],
    'MAKE': [MAKE],
    'USAGE': [USAGE],
    'CLAIM_PAID_FLAG': [CLAIM_PAID_FLAG]
})

# Encode using saved LabelEncoders
for col in ['MAKE', 'USAGE']:
    le = label_encoders[col]
    try:
        X_input[col] = le.transform(X_input[col])
    except ValueError:
        st.error(f"Invalid input: '{X_input[col].values[0]}' is not recognized in {col}.")
        st.stop()

# Ensure correct column order
X_input = X_input.reindex(columns=model_columns, fill_value=0)

# Predict and display
if st.button("Predict Premium"):
    premium_pred = rf_model.predict(X_input)
    st.success(f"Predicted Premium: {premium_pred[0]:.2f}")