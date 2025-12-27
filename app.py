import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# -------------------------------
# Background Styling (CSS)
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            rgba(60,0,0,0.85),
            rgba(120,0,0,0.85)
        );
        background-attachment: fixed;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: #ffffff !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #fff5f5;
        color: black;
    }

    .stButton button {
        background-color: #8B0000;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 8px 20px;
    }

    .stButton button:hover {
        background-color: #5a0000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Title
# -------------------------------
st.title("🍷 Wine Quality Prediction")
st.write("Enter wine parameters to predict quality")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("winequality-red.csv")

data = load_data()

x = data.drop("quality", axis=1)
y = data["quality"]

# -------------------------------
# Train Model
# -------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(x, y)

# -------------------------------
# User Inputs
# -------------------------------
fixed_acidity = st.number_input("Fixed Acidity", value=7.9)
volatile_acidity = st.number_input("Volatile Acidity", value=0.35)
citric_acid = st.number_input("Citric Acid", value=0.46)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.078)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=15)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=37)
density = st.number_input("Density", value=0.9973)
pH = st.number_input("pH", value=3.35)
sulphates = st.number_input("Sulphates", value=0.86)
alcohol = st.number_input("Alcohol", value=12.8)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Wine Quality"):
    sample_data = np.array([
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]).reshape(1, -1)

    prediction = model.predict(sample_data)
    st.success(f"🍷 Predicted Quality of Wine: {prediction[0]}")
