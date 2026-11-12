import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Use Streamlit's caching so we only load the model once per session
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    target_names = joblib.load('target_names.joblib')
    return model, scaler, feature_names, target_names

model, scaler, feature_names, target_names = load_artifacts()

st.title("🍷 Wine Quality Classifier")
st.markdown("Enter the chemical properties of the wine below to predict its origin class.")

# Dynamically generate input fields in a 3-column grid
st.subheader("Wine Features")
cols = st.columns(3)
input_data = []

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        # Format feature names to be more readable (e.g., "malic_acid" -> "Malic Acid")
        label = feature.replace('_', ' ').title()
        # Default value set to 0.0 for simplicity
        val = st.number_input(label, value=0.00, format="%.2f")
        input_data.append(val)

st.markdown("---")

# Prediction Execution
if st.button("Predict Wine Class", type="primary", use_container_width=True):

    #create a DataFrame with the exact same column order the scaler expects
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # 1. Scale the raw input exactly as we did during training
    scaled_features = scaler.transform(input_df)

    #class map
    class_map = {
        0: "Class 1",
        1: "Class 2",
        2: "Class 3"
    }
    
    # 2. Generate prediction
    prediction_idx = model.predict(scaled_features)[0]
    predicted_class = class_map[prediction_idx]

    # 3. Display result
    st.success(f"### Predicted Origin: **{predicted_class}**")
    st.balloons()