import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(layout='wide')
st.title("Employee Attrition Dashboard & Predictor")

DATA_CLEAN = Path("data/cleaned_employee_data.csv")
MODEL_PATH = Path("models/attrition_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")

# Load data if exists
if DATA_CLEAN.exists():
    df = pd.read_csv(DATA_CLEAN)
else:
    df = None

# Sidebar - options
st.sidebar.header("Options")
if st.sidebar.button("Show Dataset (cleaned)"):
    if df is not None:
        st.dataframe(df.head(200))
    else:
        st.warning("Cleaned data not found. Run scripts/data_preprocessing.py first.")

st.sidebar.header("Predict single employee")
# If model exists, load
model = None
scaler = None
if Path(MODEL_PATH).exists() and Path(SCALER_PATH).exists():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# Build input form from columns if available
if df is not None:
    cols = [c for c in df.columns if c != 'Attrition']
    sample = df[cols].iloc[0]
    with st.form("input_form"):
        inputs = {}
        for c in cols:
            val = sample[c]
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
                inputs[c] = st.number_input(c, value=float(val))
            else:
                # default text input for unusual types
                inputs[c] = st.text_input(c, value=str(val))
        submitted = st.form_submit_button("Predict")
        if submitted:
            if model is None or scaler is None:
                st.error("Model not found. Run scripts/model_training.py to train and create models/attrition_model.pkl")
            else:
                import numpy as np
                X = pd.DataFrame([inputs])
                # align columns
                X = X.reindex(columns=cols, fill_value=0)
                X_scaled = scaler.transform(X.values.astype(float))
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0,1]
                st.write("**Prediction:**", 'Will Leave (1)' if pred==1 else 'Will Stay (0)')
                st.write(f"**Probability of leaving:** {proba*100:.2f}%")
else:
    st.info("Run data preprocessing to create cleaned data first.")

# Simple EDA charts
if df is not None:
    st.header('Quick EDA')
    if 'Attrition' in df.columns:
        st.subheader('Attrition distribution')
        st.bar_chart(df['Attrition'].value_counts())
