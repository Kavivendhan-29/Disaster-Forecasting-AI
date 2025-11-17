import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="🌍 AI-Driven Disaster Risk Dashboard", layout="wide")

# --- Load forecast data ---
data_path = r"C:\Disaster-Forecasting\data\country_disaster_forecast_2025_2030.csv"

if not os.path.exists(data_path):
    st.error("❌ Forecast file not found! Please run `predict_country_disaster_risk.py` first.")
    st.stop()

df = pd.read_csv(data_path)

# --- Sidebar Controls ---
st.sidebar.header("🔧 Controls")
year_selected = st.sidebar.selectbox("Select Forecast Year", sorted(df["Year"].unique()))
top_n = st.sidebar.slider("Select Top N High-Risk Countries", 5, 20, 10)

# --- Filter Data ---
filtered = df[df["Year"] == year_selected].copy()
filtered = filtered.sort_values("Combined_Risk", ascending=False)

# --- Dashboard Title ---
st.title("🌍 AI-Driven Disaster Risk Forecasting Dashboard")
st.markdown(f"### 📅 Year Selected: **{year_selected}**")
st.caption("Data generated using Random Forest Model + Prophet Forecasting (2025–2030)")

# --- Top Risk Chart ---
st.subheader(f"🔥 Top {top_n} Countries by Disaster Risk — {year_selected}")
top_countries = filtered.head(top_n)
fig = px.bar(
    top_countries,
    x="Combined_Risk",
    y="Country",
    orientation="h",
    color="Combined_Risk",
    color_continuous_scale="Reds",
    title=f"Top {top_n} High-Risk Countries for {year_selected}",
)
st.plotly_chart(fig, use_container_width=True)

# --- Table of All Countries ---
st.subheader("🌐 Country-Level Forecast Data")
st.dataframe(filtered[["Country", "Predicted_Disaster_Risk", "Historical_Risk", "Combined_Risk"]]
             .sort_values("Combined_Risk", ascending=False)
             .reset_index(drop=True))

# --- Summary Metrics ---
st.subheader("📊 Global Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Global Risk", f"{filtered['Combined_Risk'].mean():.2f}")
col2.metric("Highest Risk Country", filtered.iloc[0]['Country'])
col3.metric("Lowest Risk Country", filtered.iloc[-1]['Country'])

# --- Footer ---
st.markdown("---")
st.markdown(
    "📘 *Developed as part of the Intelligent AI-Driven Disaster Forecasting & Early Warning System Project*"
)
