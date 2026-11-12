import pandas as pd
import joblib
import shap
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def run_pipeline():
    print("Loading Wine dataset...")
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    print("Splitting and scaling...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    print("Exporting model and scaler...")
    joblib.dump(clf, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Export feature names for the Streamlit UI
    joblib.dump(data.feature_names, 'feature_names.joblib')
    joblib.dump(data.target_names, 'target_names.joblib')
    
    print("Pipeline complete! Artifacts saved.")

if __name__ == "__main__":
    run_pipeline()