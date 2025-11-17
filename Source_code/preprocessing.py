import pandas as pd

def load_and_prepare_data():
    print("🔍 Loading datasets...")

    # Load data
    weather = pd.read_csv(r"C:\Disaster-Forecasting\data\Gobal_Weather_Data.csv")
    disaster = pd.read_csv(r"C:\Disaster-Forecasting\data\cleaned_emdat.csv")

    print(f"✅ Weather shape: {weather.shape}")
    print(f"✅ Disaster shape: {disaster.shape}")

    # Convert date column in weather data
    if 'Date' in weather.columns:
        weather['Date'] = pd.to_datetime(weather['Date'], dayfirst=True, errors='coerce')
        weather['Year'] = weather['Date'].dt.year
        weather['Month'] = weather['Date'].dt.month
        weather['Day'] = weather['Date'].dt.day
        weather = weather.drop(columns=['Date'])  # remove the original string column

    # Merge datasets
    data = pd.merge(weather, disaster[['Country', 'Disaster_Occurred']], on='Country', how='left')
    print(f"✅ Merged dataset shape before sampling: {data.shape}")

    # Sample 10% of the data for faster training
    sample_fraction = 0.10
    data = data.sample(frac=sample_fraction, random_state=42)
    print(f"✅ Sampled dataset shape: {data.shape}")

    # Fill missing values
    data['Disaster_Occurred'] = data['Disaster_Occurred'].fillna(0)
    data = data.fillna(0)

    # Drop non-numeric columns (like Country)
    X = data.drop(columns=['Disaster_Occurred', 'Country'])
    X = X.select_dtypes(include=['float64', 'int64'])
    y = data['Disaster_Occurred']

    print("📊 Disaster distribution:")
    print(y.value_counts())

    print("✅ Data ready for training.")
    return X, y
