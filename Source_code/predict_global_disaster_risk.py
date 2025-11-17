import pandas as pd
import joblib
import os
import numpy as np

def predict_country_disaster_risk(start_year=2025, end_year=2030):
    print(f"🌍 Generating country-level disaster risk forecasts for {start_year}–{end_year}...")

    # Required files
    weather_path = r"C:\Disaster-Forecasting\data\weather_forecast.csv"
    disaster_path = r"C:\Disaster-Forecasting\data\cleaned_emdat.csv"
    model_path = r"C:\Disaster-Forecasting\disaster_forecast_model.pkl"

    if not all(os.path.exists(p) for p in [weather_path, disaster_path, model_path]):
        raise FileNotFoundError("❌ Required files missing. Ensure forecast, dataset, and model exist.")

    weather = pd.read_csv(weather_path)
    disaster = pd.read_csv(disaster_path)
    model = joblib.load(model_path)

    # --- Clean & prep weather data ---
    weather["Date"] = pd.to_datetime(weather["Date"], errors="coerce")
    weather["Year"] = weather["Date"].dt.year

    # --- Simplify disaster data ---
    disaster = disaster.groupby("Country", as_index=False)[["Total Deaths"]].mean()
    disaster.rename(columns={"Total Deaths": "Historical_Risk"}, inplace=True)

    all_forecasts = []

    for year in range(start_year, end_year + 1):
        print(f"📅 Processing year {year}...")

        year_weather = weather.copy()
        year_weather["Year"] = year  # extend existing trends forward
        countries = disaster["Country"].unique()

        yearly_data = []
        for c in countries:
            sampled = year_weather.copy()

            # Randomized plausible global weather variation
            sampled["Temp_Mean"] = sampled["Predicted_Temp"] + np.random.uniform(-2, 2)
            sampled["Precipitation_Sum"] = np.random.uniform(0, 50)
            sampled["Windspeed_Max"] = np.random.uniform(0, 40)
            sampled["Sunshine_Duration"] = np.random.uniform(100, 300)
            sampled["Country"] = c
            yearly_data.append(sampled)

        combined = pd.concat(yearly_data, ignore_index=True)

        # --- Prepare model input ---
        expected_features = model.feature_names_in_
        for col in expected_features:
            if col not in combined.columns:
                combined[col] = 0

        X_future = combined[expected_features]
        combined["Predicted_Disaster_Risk"] = model.predict_proba(X_future)[:, 1]

        # --- Merge with historical risk ---
        combined = combined.merge(disaster, on="Country", how="left")
        combined["Combined_Risk"] = (
            0.7 * combined["Predicted_Disaster_Risk"] +
            0.3 * (combined["Historical_Risk"].fillna(combined["Historical_Risk"].mean()) /
                   combined["Historical_Risk"].max())
        )

        # --- Aggregate per country ---
        country_risk = combined.groupby("Country", as_index=False)[
            ["Predicted_Disaster_Risk", "Historical_Risk", "Combined_Risk"]
        ].mean()
        country_risk["Year"] = year

        all_forecasts.append(country_risk)

    # --- Combine all years ---
    final_forecast = pd.concat(all_forecasts, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    save_path = "data/country_disaster_forecast_2025_2030.csv"
    final_forecast.to_csv(save_path, index=False)

    print(f"✅ Multi-year country-level forecast saved to: {save_path}")
    print(final_forecast.head(10))
    return final_forecast


if __name__ == "__main__":
    predict_country_disaster_risk(2025, 2030)
