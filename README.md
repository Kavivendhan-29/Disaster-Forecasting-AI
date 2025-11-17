Disaster-Forecasting-AI
AI-Driven Global Disaster Forecasting & Early Warning System

An end-to-end AI system that predicts natural disaster risks globally by combining historical disaster records, multi-country weather data, machine learning classification, and time-series forecasting.

Features
1. Disaster Prediction Model

Uses Random Forest Classifier
Balanced using RandomUnderSampler

Achieves:
Accuracy: 0.7896
ROC-AUC: 0.7896

2. Weather Forecasting (Prophet)

Forecasts 12 months of future weather trends
Predicts temperature patterns using Facebook Prophet
Generates: weather_forecast.csv

3. Future Disaster Risk Prediction

Combines forecasted weather + historical disaster data
Predicts disaster probability for upcoming months
Outputs: future_disaster_forecast.csv

4. Streamlit Dashboard

Includes:
Historical disaster visualization
Future weather prediction charts
Global + country-wise disaster risk
Interactive controls for year-wise forecasts

Technologies Used
Languages:-
Python 3
Libraries:-
ML: scikit-learn, imbalanced-learn
Forecasting: Prophet, cmdstanpy
Data Processing: Pandas, NumPy
Visualization: Plotly
Dashboard: Streamlit

Model Performance

Metric	Score
Accuracy	0.7896
ROC-AUC	0.7896
Balanced Dataset	Yes
Train/Test Split	80/20

System Architecture

Load weather + disaster datasets
Clean, merge, and engineer features
Balance dataset using undersampling
Train Random Forest model
Forecast weather using Prophet
Predict future disaster probabilities
Display insights on Streamlit dashboard

Why This Project Matters

Global disasters are increasing every year.
This system enables:

✔ Early preparedness
✔ Research insights
✔ Climate-risk monitoring
✔ Policy-level decision support
