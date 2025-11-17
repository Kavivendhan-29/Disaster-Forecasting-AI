import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import joblib

def train_and_evaluate_model(X, y):
    print("📊 Balancing the dataset...")

    # Step 1: Resample (equal disaster/no-disaster samples)
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

    print(f"✅ Resampled dataset shape: {X_res.shape}")
    print("📊 Class distribution after resampling:")
    print(pd.Series(y_res).value_counts())

    # Step 2: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # Step 3: Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    print("🚀 Training Random Forest...")
    model.fit(X_train, y_train)

    # Step 4: Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(f"\n✅ Random Forest Accuracy: {acc:.4f}")
    print(f"✅ ROC-AUC: {auc:.4f}")

    # Step 5: Feature importance plot
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(X.columns)[indices], importance[indices])
    plt.xlabel("Importance")
    plt.title("Top 10 Important Features (Random Forest)")
    plt.tight_layout()
    plt.show()

    # Step 6: Save trained model
    joblib.dump(model, "disaster_forecast_model.pkl")
    print("\n💾 Model saved as 'disaster_forecast_model.pkl'")

    return model, acc, auc
