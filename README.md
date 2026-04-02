# Seasonal Disease Surge Prediction System

## Overview

The **Seasonal Disease Surge Prediction System** is an AI-powered application that predicts disease outbreaks **2–3 weeks in advance** using environmental and healthcare data.
It helps healthcare authorities take **proactive measures** instead of reacting after outbreaks occur.

---

## Key Features

*  Predicts **disease outbreak risk** (Low / Medium / High)
*  Identifies **likely disease type** (Dengue, Flu, Typhoid, etc.)
*  Interactive dashboard using Streamlit
*  Early warning alerts for proactive healthcare planning
*  Trend visualization of disease cases

---

##  How It Works

1. Collects input data:

   * Rainfall
   * Temperature
   * OPD (hospital visits)

2. Uses **Machine Learning (Random Forest)** to:

   * Predict outbreak risk
   * Classify disease type

3. Displays results via:

   * Risk level indicator
   * Disease prediction
   * Graphs and dashboard

---

##  Dataset

The dataset includes:

* Rainfall
* Temperature
* OPD Visits
* Disease Cases
* Risk Level
* Disease Name

---

##  Tech Stack

* **Programming Language:** Python
* **Framework:** Streamlit
* **Libraries:**

  * Pandas
  * NumPy
  * Scikit-learn
  * Matplotlib

---

##  How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Seasonal-Disease-Surge-Prediction.git
cd Seasonal-Disease-Surge-Prediction
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

##  Demo Inputs

Example:

* Rainfall: 120 mm
* Temperature: 34°C
* OPD Visits: 500

 Output:

*  High Risk
*  Outbreak Alert
*  Predicted Disease: Dengue

---

##  Use Case

* Government healthcare departments
* Hospitals and clinics
* Public health monitoring systems

---

##  Impact

* Enables **early detection of outbreaks**
* Improves **resource planning (beds, staff, medicines)**
* Reduces **hospital overload**
* Helps save lives through proactive action

---

##  Future Enhancements

*  District-wise heatmap visualization
*  Real-time data integration
*  Advanced models (LSTM, XGBoost)
*  Cloud deployment


---

##  Acknowledgement

This project demonstrates how **AI can transform public healthcare** by enabling predictive and data-driven decision-making.

---
