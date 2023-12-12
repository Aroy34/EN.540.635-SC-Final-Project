import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matminer.datasets import load_dataset
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump,load
import os


def train_load_model(data_set, target_variable, features_to_drop):
    df = load_dataset(data_set)

    y = df[target_variable].values
    X = df.drop(features_to_drop, axis=1)

    scaler_save_path = f"{data_set}_model_scaler.joblib"
    rf_model_save_path = f"{data_set}_rf_model.joblib"

    if os.path.isfile(scaler_save_path) and os.path.isfile(rf_model_save_path):
        print("Loading the existing Random Forest model...")
        scaler = load(scaler_save_path)
        rf_model = load(rf_model_save_path)
    else:
        print("No existing models found. Training new Random Forest model...")
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.33, random_state=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
        rf_model.fit(X_train, y_train.ravel())

        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(y_train, 'r', label='Actual')
        plt.plot(y_train_pred, 'b', label='Predicted')
        plt.title('Training Data')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(y_test, 'r', label='Actual')
        plt.plot(y_test_pred, 'b', label='Predicted')
        plt.title('Testing Data')
        plt.legend()
        plt.tight_layout()
        plt.show()

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        print("Train RMSE:", train_rmse)
        print("Test RMSE:", test_rmse)

        dump(rf_model, rf_model_save_path)
        dump(scaler, scaler_save_path)

    print("Model training complete. Model and scaler saved.")


if __name__ == "__main__":
    # dataset_name = "steel_strength"
    # target_variable = "yield strength"
    # features_to_drop = ['formula', 'yield strength', 'tensile strength', 'elongation']
    train_load_model(dataset_name, target_variable, features_to_drop)