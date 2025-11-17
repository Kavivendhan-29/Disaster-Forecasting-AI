# main.py
from preprocessing import load_and_prepare_data
from train_model import train_and_evaluate_model

if __name__ == "__main__":
    print("🔍 Loading and preparing data...")
    X, y = load_and_prepare_data()

    print("📊 Training model now...")
    results = train_and_evaluate_model(X, y)

    print("\n✅ Training complete. Model results:")
    print(results)
