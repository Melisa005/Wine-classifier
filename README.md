# 🍷 Wine Classifier

A Streamlit web application that predicts the origin class of wine based on its chemical properties using a Machine Learning model.

## Overview

This project uses the classic **Wine Recognition Dataset** from scikit-learn. The model classifies wine into one of three origin classes based on 13 chemical features.

**Original Class Labels:**
- Class 1
- Class 2
- Class 3

## Features

- Interactive input form for all 13 wine chemical properties
- Real-time prediction using a trained Machine Learning model
- Clean and simple user interface built with Streamlit
- Proper feature scaling using the same scaler used during training

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- joblib

## Project Structure

Wine-classifier/ ├── app/ │ ├── app.py │ ├── model.joblib │ ├── scaler.joblib │ ├── feature_names.joblib │ └── target_names.joblib ├── requirements.txt └── README.md

## How to Run Locally

  1. Clone the repository:
     
     ```powershell
     clone https://github.com/Melisa005/Wine-classifier.git
     cd Wine-classifier

  2. Create a virtual environment:
     
     ```powershell
      python -m venv .venv
      .venv\Scripts\activate

  3. Install dependencies:

     ```powershell
      pip install -r requirements.txt

  4. Run the app:

     ```powershell
      streamlit run app/app.py

## Model Information

- Algorithm: Trained classification model (scikit-learn)
- Features used: 13 chemical properties of wine
- Preprocessing: StandardScaler
- Target: Wine origin class (1, 2, or 3)

## Author

Chinwendu Melisa Ottah

Aspiring ML Engineer


Feel free to star ⭐ the repository if you find it useful!


