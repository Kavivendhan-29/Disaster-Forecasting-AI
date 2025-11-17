import pandas as pd
import joblib
import os

def predict_future_disasters():
    print("🚨 Generating future disaster risk forecast...")

    model_path = r"C:\Disaster-Forecasting\disaster_forecast_model.pkl"
    forecast_path = r"C:\Disaster-Forecasting\data\weather_forecast.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Cannot find model file: {model_path}")
    if not os.path.exists(forecast_path):
        raise FileNotFoundError(f"❌ Cannot find forecast file: {forecast_path}")

    model = joblib.load(model_path)
    forecast = pd.read_csv(forecast_path)

    # Rename Prophet output columns
    forecast.rename(columns={'Predicted_Temp': 'Temp_Mean'}, inplace=True)

    # Derive missing weather variables (approximations)
    forecast['Temp_Max'] = forecast['Temp_Mean'] + 2
    forecast['Temp_Min'] = forecast['Temp_Mean'] - 2
    forecast['Precipitation_Sum'] = forecast['Temp_Mean'] * 0.15
    forecast['Windspeed_Max'] = forecast['Temp_Mean'] * 0.3
    forecast['Windgusts_Max'] = forecast['Temp_Mean'] * 0.4
    forecast['Sunshine_Duration'] = 8 + (forecast['Temp_Mean'] % 5)  # simulate daily sunlight hours

    # ✅ Align with training features
    expected_features = list(model.feature_names_in_)
    for col in expected_features:
        if col not in forecast.columns:
            forecast[col] = 0  # add missing feature if absent

    # Sort columns in same order as model training
    X_future = forecast[expected_features].copy()

    # Predict disaster probability
    forecast['Predicted_Disaster_Risk'] = model.predict_proba(X_future)[:, 1]

    # Save results
    os.makedirs("data", exist_ok=True)
    output_path = "data/future_disaster_forecast.csv"
    forecast.to_csv(output_path, index=False)

    print(f"✅ Future disaster risk forecast saved to: {output_path}")
    print(forecast[['Date', 'Predicted_Disaster_Risk']].head())

    return forecast

if __name__ == "__main__":
    predict_future_disasters()
