import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Climate Extreme Temperature Predictor", layout="centered")

st.title("🌍 Climate Extreme Temperature Predictor")
st.write("Enter environmental values to predict extreme temperature event:")

# =======================
# Safe model & scaler load
# =======================
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "climate_model.pkl")
scaler_path = os.path.join(base_path, "scaler.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    st.error("Model or scaler file not found. Make sure climate_model.pkl and scaler.pkl are in the same folder as app.py")
    st.stop()

# =======================
# User input
# =======================
temperature = st.number_input("Temperature (°C)", value=20.0)
co2 = st.number_input("CO₂ Emissions", value=400.0)
sea_level = st.number_input("Sea Level Rise", value=0.5)
precipitation = st.number_input("Precipitation", value=20.0)
humidity = st.number_input("Humidity (%)", value=50.0)
wind = st.number_input("Wind Speed", value=10.0)
year = st.number_input("Year", value=2024)
month = st.number_input("Month", min_value=1, max_value=12, value=1)

# =======================
# Prediction
# =======================
if st.button("Predict"):
    input_data = np.array([[temperature, co2, sea_level, precipitation, humidity, wind, year, month]])
    
    # Apply scaler before prediction
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0]
    probability = proba[1] if prediction[0] == 1 else proba[0]

    if prediction[0] == 1:
        st.error(f"⚠️ Extreme Temperature Event Predicted! Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Normal Temperature Predicted. Probability: {probability*100:.2f}%")
