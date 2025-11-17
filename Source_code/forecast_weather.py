import pandas as pd
from prophet import Prophet
import joblib
import os

def forecast_weather(input_csv=r"C:\Disaster-Forecasting\data\Gobal_Weather_Data.csv", months_ahead=60):
    print(f"🔮 Forecasting weather trends for the next {months_ahead} months...")

    # Check file
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"❌ Cannot find input file: {input_csv}")

    # Load data
    df = pd.read_csv(input_csv)
    if 'Date' not in df.columns or 'Temp_Mean' not in df.columns:
        raise ValueError("❌ Input CSV must contain 'Date' and 'Temp_Mean' columns.")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by='Date')

    # Aggregate daily averages
    df_mean = df.groupby('Date', as_index=False)['Temp_Mean'].mean()
    df_mean = df_mean.rename(columns={'Date': 'ds', 'Temp_Mean': 'y'})

    # Initialize Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_mean)

    # Forecast future months (default: 5 years)
    future = m.make_future_dataframe(periods=months_ahead, freq='M')
    forecast = m.predict(future)

    # Clean forecast dataframe
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Temp'}, inplace=True)

    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save outputs
    output_path = os.path.join("data", "weather_forecast.csv")
    forecast.to_csv(output_path, index=False)
    print(f"✅ {months_ahead}-month weather forecast saved to '{output_path}'.")

    model_path = os.path.join("models", "weather_prophet_model.pkl")
    joblib.dump(m, model_path)
    print(f"💾 Prophet model saved to {model_path}")

    # Display preview
    print("\n📊 Forecast Preview:")
    print(forecast.head())

    return forecast


if __name__ == "__main__":
    # Example: forecast for 60 months (5 years)
    forecast_weather(months_ahead=60)
