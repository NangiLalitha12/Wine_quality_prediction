import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
    return pd.read_csv("winequality-red.csv")  # keep file in same folder

data = load_data()

x = data.drop("quality", axis=1)
y = data["quality"]

# -------------------------------
# Train Model
# -------------------------------
model = RandomForestClassifier()
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