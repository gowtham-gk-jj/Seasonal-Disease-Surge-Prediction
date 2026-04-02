import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Disease Surge Prediction", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("🧠 Seasonal Disease Surge Prediction System")
st.markdown("Predict disease outbreaks **2–3 weeks in advance** using AI/ML")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Book 1(Sheet1).csv")

data = load_data()

# -----------------------------
# Rename Columns (UPDATED)
# -----------------------------
data.columns = ['rainfall', 'temperature', 'opd_visits', 'disease_cases', 'risk', 'disease']

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📥 Input Parameters")

rainfall = st.sidebar.slider("Rainfall (mm)", 0, 200, 50)
temperature = st.sidebar.slider("Temperature (°C)", 20, 45, 30)
opd_visits = st.sidebar.slider("OPD Visits", 50, 1000, 200)

# -----------------------------
# Prepare Data (Risk Model)
# -----------------------------
X = data[['rainfall', 'temperature', 'opd_visits']]
y = data['risk']

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# Disease Prediction Model
# -----------------------------
X_disease = data[['rainfall', 'temperature', 'opd_visits']]
y_disease = data['disease']

disease_model = RandomForestClassifier()
disease_model.fit(X_disease, y_disease)

# -----------------------------
# Prediction
# -----------------------------
input_data = np.array([[rainfall, temperature, opd_visits]])

# Risk Prediction
probability = model.predict_proba(input_data)[0][1]

# Disease Prediction
predicted_disease = disease_model.predict(input_data)[0]

# -----------------------------
# Risk Level Logic
# -----------------------------
if probability > 0.7:
    risk_level = "🔴 HIGH RISK"
    alert = "⚠️ High outbreak expected in next 2–3 weeks!"
elif probability > 0.4:
    risk_level = "🟡 MEDIUM RISK"
    alert = "⚠️ Moderate risk detected. Monitor closely."
else:
    risk_level = "🟢 LOW RISK"
    alert = "✅ No major outbreak expected."

# -----------------------------
# Display Results
# -----------------------------
st.subheader("📊 Prediction Result")

st.write(f"### Risk Level: {risk_level}")
st.write(f"Prediction Confidence: {round(probability*100,2)}%")

st.write("### 🦠 Predicted Disease")
st.info(predicted_disease)

if "HIGH" in risk_level:
    st.error(alert)
elif "MEDIUM" in risk_level:
    st.warning(alert)
else:
    st.success(alert)

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📈 Disease Cases Trend")

fig, ax = plt.subplots()
ax.plot(data['disease_cases'], marker='o')
ax.set_title("Disease Cases Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Cases")

st.pyplot(fig)

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(data)

# -----------------------------
# District Risk Demo
# -----------------------------
st.subheader("📍 District Risk Overview")

district_data = pd.DataFrame({
    "District": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "Risk Level": ["High", "Medium", "Low", "Medium"]
})

st.table(district_data)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🚀 AI-Based Early Warning System for Healthcare")