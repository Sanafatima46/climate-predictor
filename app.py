
import streamlit as st
import numpy as np
import joblib

model = joblib.load("climate_model.pkl")

st.title("🌍 Climate Extreme Temperature Predictor")
st.write("Enter environmental values to predict extreme temperature event:")

temperature = st.number_input("Temperature (°C)", value=20.0)
co2 = st.number_input("CO₂ Emissions", value=400.0)
sea_level = st.number_input("Sea Level Rise", value=0.5)
precipitation = st.number_input("Precipitation", value=20.0)
humidity = st.number_input("Humidity (%)", value=50.0)
wind = st.number_input("Wind Speed", value=10.0)
year = st.number_input("Year", value=2024)
month = st.number_input("Month", min_value=1, max_value=12, value=1)

if st.button("Predict"):
    input_data = np.array([[temperature, co2, sea_level, precipitation, humidity, wind, year, month]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ Extreme Temperature Event Predicted")
    else:
        st.success("✅ Normal Temperature Predicted")
