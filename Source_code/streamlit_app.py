import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load(r"C:\Disaster-Forecasting\disaster_forecast_model.pkl")

st.title("🌍 AI-Powered Disaster Forecasting System")
st.write("Predict the probability of a natural disaster based on weather conditions.")

# Input fields must match training features EXACTLY
temp_max = st.number_input("Max Temperature (°C)", min_value=-50.0, max_value=60.0, value=30.0)
temp_min = st.number_input("Min Temperature (°C)", min_value=-50.0, max_value=60.0, value=20.0)
temp_mean = st.number_input("Mean Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
precipitation_sum = st.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, value=5.0)
windspeed_max = st.number_input("Max Wind Speed (km/h)", min_value=0.0, max_value=300.0, value=50.0)
windgusts_max = st.number_input("Max Wind Gust (km/h)", min_value=0.0, max_value=300.0, value=80.0)
sunshine_duration = st.number_input("Sunshine Duration (hours)", min_value=0.0, max_value=24.0, value=8.0)

# Create dataframe matching training feature names
input_data = pd.DataFrame([{
    'Temp_Max': temp_max,
    'Temp_Min': temp_min,
    'Temp_Mean': temp_mean,
    'Precipitation_Sum': precipitation_sum,
    'Windspeed_Max': windspeed_max,
    'Windgusts_Max': windgusts_max,
    'Sunshine_Duration': sunshine_duration
}])

# Prediction
if st.button("🔮 Predict Disaster Risk"):
    prediction = model.predict_proba(input_data)[0][1]
    st.success(f"Predicted Disaster Probability: {prediction:.2%}")

    if prediction > 0.7:
        st.error("⚠️ High disaster risk! Immediate attention required.")
    elif prediction > 0.4:
        st.warning("⚠️ Moderate disaster risk.")
    else:
        st.info("✅ Low disaster risk.")
