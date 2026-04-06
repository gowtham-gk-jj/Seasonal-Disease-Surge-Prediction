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
    df = pd.read_csv("Book 1(Sheet1).csv")
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

city = st.sidebar.selectbox(
    "Select District",
    ["Chennai", "Coimbatore", "Madurai", "Salem", "Trichy"]
)

def get_weather(city):
    try:
        api_key = "YOUR_API_KEY"  # 🔥 Replace this
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url).json()

        temperature = res['main']['temp']
        humidity = res['main']['humidity']
        rainfall = res.get('rain', {}).get('1h', 0)

        return temperature, humidity, rainfall
    except:
        return None, None, None

temp_live, humidity, rainfall_live = get_weather(city)

# Fallback
if temp_live is None:
    st.sidebar.warning("⚠️ Weather API failed. Using default values.")
    temp_live = 30
    rainfall_live = 50

# Estimate OPD based on temperature
opd_visits = int(200 + (temp_live * 5))

# Display live values
st.sidebar.write(f"🌡 Temperature: {temp_live}°C")
st.sidebar.write(f"🌧 Rainfall: {rainfall_live} mm")
st.sidebar.write(f"🏥 OPD Visits (estimated): {opd_visits}")

# Assign values
temperature = temp_live
rainfall = rainfall_live

# -----------------------------
# Train Model (XGBoost)
# -----------------------------
@st.cache_resource
def train_model(df):
    features = ['rainfall', 'temperature', 'opd_visits']
    
    X = df[features]
    y = df['risk']

    model = XGBClassifier(n_estimators=150, learning_rate=0.05)
    model.fit(X, y)

    return model, features

model, features = train_model(data)

# -----------------------------
# Find Closest District
# -----------------------------
def find_closest_district(df, rainfall, temperature, opd):
    df = df.copy()
    df['distance'] = (
        (df['rainfall'] - rainfall)**2 +
        (df['temperature'] - temperature)**2 +
        (df['opd_visits'] - opd)**2
    )
    return df.loc[df['distance'].idxmin()]

closest = find_closest_district(data, rainfall, temperature, opd_visits)
predicted_district = closest['district']

# -----------------------------
# Prediction
# -----------------------------
input_df = pd.DataFrame([[rainfall, temperature, opd_visits]], columns=features)
probability = model.predict_proba(input_df)[0][1]
predicted_disease = closest['disease']

# -----------------------------
# Risk Logic
# -----------------------------
if probability > 0.7:
    risk_level = "🔴 HIGH RISK"
    alert = "⚠️ Outbreak expected in next 2–3 weeks!"
elif probability > 0.4:
    risk_level = "🟡 MEDIUM RISK"
    alert = "⚠️ Moderate risk detected"
else:
    risk_level = "🟢 LOW RISK"
    alert = "✅ No major outbreak expected"

# -----------------------------
# Display Results
# -----------------------------
st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("📍 District", predicted_district)
    st.metric("Risk Level", risk_level)
    st.metric("Confidence", f"{round(probability*100,2)}%")

with col2:
    st.write("### 🦠 Predicted Disease")
    st.info(predicted_disease)

# Alerts
if "HIGH" in risk_level:
    st.error(alert)
elif "MEDIUM" in risk_level:
    st.warning(alert)
else:
    st.success(alert)

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
# Map Visualization
# -----------------------------
st.subheader("🌍 District Risk Map")

coords = pd.DataFrame({
    "district": ["Chennai", "Coimbatore", "Madurai", "Salem", "Trichy"],
    "lat": [13.08, 11.01, 9.92, 11.66, 10.79],
    "lon": [80.27, 76.96, 78.12, 78.14, 78.70],
})

merged = pd.merge(coords, data.groupby('district')['risk'].mean().reset_index(), on="district")

fig_map = px.scatter_mapbox(
    merged,
    lat="lat",
    lon="lon",
    size="risk",
    color="risk",
    hover_name="district",
    zoom=5,
    height=400
)

fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map)

# -----------------------------
# Trend Visualization
# -----------------------------
st.subheader(f"📈 Disease Trend in {predicted_district}")

district_data = data[data['district'] == predicted_district].copy()
district_data["time"] = range(len(district_data))

fig3 = px.line(
    district_data,
    x="time",
    y="disease_cases",
    markers=True
)

st.plotly_chart(fig3)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🚀 AI-Based Early Warning System for Smart Healthcare")