import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import shap

from xgboost import XGBClassifier

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Disease Surge Prediction", layout="wide")

st.title("🧠 Seasonal Disease Surge Prediction System")
st.markdown("AI-powered system to predict disease outbreaks **2–3 weeks in advance**")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tamil_nadu_disease_dataset.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

data = load_data()

# -----------------------------
# Required Columns Check
# -----------------------------
required_cols = ['district','rainfall','temperature','opd_visits','risk','disease','disease_cases']

missing = [col for col in required_cols if col not in data.columns]

if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

# -----------------------------
# 🌍 LIVE WEATHER INPUT (UPDATED)
# -----------------------------
st.sidebar.header("🌍 Live Weather Input")

# 🔥 UPDATED: dynamic districts from dataset
district_list = sorted(data['district'].unique())

city = st.sidebar.selectbox("Select District", district_list)

API_KEY = "d68c839433a113d970341e5746f9aa6a"

# -----------------------------
# Current Weather
# -----------------------------
def get_weather(city):
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if str(data.get("cod")) != "200":
        return None, None

    temp = data['main']['temp']
    rain = data.get('rain', {}).get('1h', 0)

    return temp, rain

temp_live, rainfall_live = get_weather(city)

if temp_live is None:
    st.sidebar.warning("⚠️ Using default values")
    temp_live = 30
    rainfall_live = 50
else:
    st.sidebar.success("✅ Live Weather Loaded")

opd_visits = int(200 + (temp_live * 5))

st.sidebar.write(f"🌡 Temperature: {temp_live}°C")
st.sidebar.write(f"🌧 Rainfall: {rainfall_live} mm")
st.sidebar.write(f"🏥 OPD Visits: {opd_visits}")

temperature = temp_live
rainfall = rainfall_live

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df[['rainfall', 'temperature', 'opd_visits']]
    y = df['risk']

    model = XGBClassifier(n_estimators=150, learning_rate=0.05)
    model.fit(X, y)

    return model

model = train_model(data)

# -----------------------------
# Prediction
# -----------------------------
input_df = pd.DataFrame([[rainfall, temperature, opd_visits]],
                        columns=['rainfall','temperature','opd_visits'])

probability = model.predict_proba(input_df)[0][1]

# -----------------------------
# Risk Logic
# -----------------------------
def get_risk_label(prob):
    if prob > 0.7:
        return "🔴 HIGH", "error"
    elif prob > 0.4:
        return "🟡 MEDIUM", "warning"
    else:
        return "🟢 LOW", "success"

risk_label, _ = get_risk_label(probability)

# -----------------------------
# Display Current Prediction
# -----------------------------
st.subheader("📊 Current Prediction")

st.metric("Risk Level", risk_label)
st.metric("Confidence", f"{round(probability*100,2)}%")

# -----------------------------
# 🔮 FORECAST FUNCTION
# -----------------------------
def get_forecast(city):
    url = "https://api.openweathermap.org/data/2.5/forecast"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    forecast = []

    if str(data.get("cod")) != "200":
        return forecast

    for item in data["list"][:5]:
        temp = item["main"]["temp"]
        rain = item.get("rain", {}).get("3h", 0)
        opd = int(200 + temp * 5)

        forecast.append([rain, temp, opd])

    return forecast

# -----------------------------
# 🔮 FUTURE PREDICTION
# -----------------------------
st.subheader("🔮 Future Outbreak Prediction")

forecast_data = get_forecast(city)

if forecast_data:
    for i, vals in enumerate(forecast_data):
        future_df = pd.DataFrame([vals],
            columns=['rainfall','temperature','opd_visits'])

        prob = model.predict_proba(future_df)[0][1]
        label, level = get_risk_label(prob)

        if level == "error":
            st.error(f"Day {i+1}: {label} ({round(prob*100,1)}%)")
        elif level == "warning":
            st.warning(f"Day {i+1}: {label} ({round(prob*100,1)}%)")
        else:
            st.success(f"Day {i+1}: {label} ({round(prob*100,1)}%)")
else:
    st.warning("Forecast not available")

# -----------------------------
# 🗺 FUTURE RISK HEATMAP (UPDATED)
# -----------------------------
st.subheader("🗺 Future Risk Heatmap")

# 🔥 Use dataset districts
coords = data[['district']].drop_duplicates().copy()

# Dummy lat/lon mapping (can upgrade later)
lat_map = np.linspace(8.0, 13.5, len(coords))
lon_map = np.linspace(76.0, 80.5, len(coords))

coords["lat"] = lat_map
coords["lon"] = lon_map

future_risks = []

for d in coords["district"]:
    forecast = get_forecast(d)

    if forecast:
        vals = forecast[0]
        df_input = pd.DataFrame([vals],
            columns=['rainfall','temperature','opd_visits'])
        p = model.predict_proba(df_input)[0][1]
    else:
        p = 0

    future_risks.append(p)

coords["risk"] = future_risks

fig_map = px.scatter_mapbox(
    coords,
    lat="lat",
    lon="lon",
    size="risk",
    color="risk",
    hover_name="district",
    zoom=5,
    height=500
)

fig_map.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_map)

# -----------------------------
# SHAP Explainability
# -----------------------------
st.subheader("🧠 Why this prediction?")

explainer = shap.Explainer(model)
shap_values = explainer(input_df)

fig, ax = plt.subplots()
shap.plots.bar(shap_values, show=False)
st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🚀 AI-Based Early Warning System for Smart Healthcare")