# heart_disease_data.py
# Download and preprocess the UCI Heart Disease dataset for federated learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_heart_disease_data():
    # Download dataset
    url = "app/processed.cleveland.data.txt"
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(url, names=columns)
    # Replace missing values with median
    df = df.replace("?", np.nan)
    df = df.fillna(df.median(numeric_only=True))
    # Convert all columns to float
    df = df.astype(float)
    # Binary classification: target 0 (healthy) vs 1 (disease)
    df["target"] = (df["target"] > 0).astype(int)
    # Features and labels
    X = df.drop("target", axis=1).values
    y = df["target"].values
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def split_for_clients(X, y, num_clients=3):
    # Split data for federated clients
    X_splits = np.array_split(X, num_clients)
    y_splits = np.array_split(y, num_clients)
    return list(zip(X_splits, y_splits))

if __name__ == "__main__":
    X, y = load_heart_disease_data()
    splits = split_for_clients(X, y)
    for i, (Xc, yc) in enumerate(splits):
        print(f"Client {i+1}: {Xc.shape[0]} samples")
