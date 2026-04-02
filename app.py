import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px
import requests

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Disease Surge Prediction", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🧠 Seasonal Disease Surge Prediction System")
st.markdown("AI-powered system to predict disease outbreaks **2–3 weeks in advance**")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Book 1(Sheet1).csv")

data = load_data()
data.columns = ['district', 'rainfall', 'temperature', 'opd_visits', 'disease_cases', 'risk', 'disease']

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📥 Input Parameters")

rainfall = st.sidebar.slider("Rainfall (mm)", 0, 200, 50)
temperature = st.sidebar.slider("Temperature (°C)", 20, 45, 30)
opd_visits = st.sidebar.slider("OPD Visits", 50, 1000, 200)

# -----------------------------
# Train Models
# -----------------------------
@st.cache_resource
def train_models(df):
    X = df[['rainfall', 'temperature', 'opd_visits']]

    rf_model = RandomForestClassifier()
    rf_model.fit(X, df['risk'])

    disease_model = RandomForestClassifier()
    disease_model.fit(X, df['disease'])

    return rf_model, disease_model

model, disease_model = train_models(data)

# -----------------------------
# Auto Detect District
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
input_data = np.array([[rainfall, temperature, opd_visits]])

probability = model.predict_proba(input_data)[0][1]
predicted_disease = disease_model.predict(input_data)[0]

# -----------------------------
# Risk Logic
# -----------------------------
if probability > 0.7:
    risk_level = "🔴 HIGH RISK"
    alert = "⚠️ High outbreak expected in next 2–3 weeks!"
elif probability > 0.4:
    risk_level = "🟡 MEDIUM RISK"
    alert = "⚠️ Moderate risk detected."
else:
    risk_level = "🟢 LOW RISK"
    alert = "✅ No major outbreak expected."

# -----------------------------
# Display Results
# -----------------------------
st.subheader("📊 Live Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("📍 Predicted District", predicted_district)
    st.metric("Risk Level", risk_level)
    st.metric("Confidence", f"{round(probability*100,2)}%")

with col2:
    st.write("### 🦠 Predicted Disease")
    st.info(predicted_disease)

# Alert
if "HIGH" in risk_level:
    st.error(alert)
elif "MEDIUM" in risk_level:
    st.warning(alert)
else:
    st.success(alert)

# -----------------------------
# 🌍 Map Visualization
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
# 📡 Live Weather API
# -----------------------------
def get_weather(city):
    try:
        api_key = "YOUR_API_KEY"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url).json()
        return res['main']['temp'], res['main']['humidity']
    except:
        return None, None

temp_live, humidity = get_weather(predicted_district)

if temp_live:
    st.info(f"🌡️ Live Temp: {temp_live}°C | 💧 Humidity: {humidity}%")

# -----------------------------
# 🧠 LSTM Model (Simple)
# -----------------------------
try:
    from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
    from tensorflow.keras.layers import LSTM, Dense

    def train_lstm(df):
        X = df[['rainfall', 'temperature', 'opd_visits']].values
        y = df['risk'].values

        X = X.reshape((X.shape[0], 1, X.shape[1]))

        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(1, 3)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X, y, epochs=5, verbose=0)

        return model

    lstm_model = train_lstm(data)

    lstm_input = np.array([[rainfall, temperature, opd_visits]]).reshape(1,1,3)
    lstm_pred = lstm_model.predict(lstm_input)[0][0]

    st.write(f"🧠 Deep Learning Risk Score: {round(lstm_pred*100,2)}%")

except:
    st.warning("⚠️ TensorFlow not installed (LSTM skipped)")

# -----------------------------
# Visualization
# -----------------------------
st.subheader(f"📈 Disease Trend in {predicted_district}")

district_data = data[data['district'] == predicted_district]

fig, ax = plt.subplots()
ax.plot(district_data['disease_cases'], marker='o')
st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🚀 AI-Based Early Warning System for Smart Healthcare")