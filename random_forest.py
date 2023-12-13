import pandas as pd
from sklearn.model_selection import train_test_split
from matminer.datasets import load_dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
import os
import pydotplus
from joblib import dump, load
import os


def train_load_model(data_set, target_variable, features_to_drop):
    """Loads or trains a Random Forest model for material property prediction.

    If a model is already saved, it loads that model; otherwise,
    it trains a new model. This function also plots
    the performance of the model on training and testing data.

    Args:
        data_set (str): The name of the dataset to use.
        target_variable (str): The target variable to predict.
        features_to_drop (list): List of feature columns
        to drop from the dataset.
    """
    df = load_dataset(data_set)

    y = df[target_variable].values
    X = df.drop(features_to_drop, axis=1)

    rf_model_save_path = f"{data_set}_rf_model.joblib"

    if os.path.isfile(rf_model_save_path):
        print("Loading the existing Random Forest model...")
        # https://stackoverflow.com/questions/20662023/save-python-random-forest-model-to-file
        rf_model = load(rf_model_save_path)
    else:
        print("No existing models found.... training a new model...")
        # https://stackoverflow.com/questions/57754373/train-test-split-method-of-scikit-learn
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=1)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
        # We can try with SVM but parameters needs to be optimized
        # svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
        rf_model.fit(X_train, y_train.ravel())

        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)

        # Calculate R^2 values
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Calculating the min and max values for plotting straight line
        min_val = min(min(y_train), min(y_train_pred),
                      min(y_test), min(y_test_pred))
        max_val = max(max(y_train), max(y_train_pred),
                      max(y_test), max(y_test_pred))

        # predicted vs. actual values(training)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(
            y_train,
            y_train_pred,
            alpha=0.3,
            color="black",
            label=f"R² = {r2_train:.2f}",
        )
        plt.plot([min_val, max_val], [min_val, max_val], "k--")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Training Data - Predicted vs. Actual [Random Forest]")
        plt.legend()

        # Plot predicted vs. actual values(test)
        plt.subplot(1, 2, 2)
        plt.scatter(
            y_test, y_test_pred, alpha=0.3,
            color="black", label=f"R² = {r2_test:.2f}"
        )
        plt.plot([min_val, max_val], [min_val, max_val], "k--")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Testing Data - Predicted vs. Actual[Random Forest]")
        plt.legend()

        plt.tight_layout()
        plt.savefig("Random Forest Model.png")
        plt.show()

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        print("Train RMSE:", train_rmse)
        print("Test RMSE:", test_rmse)

        # https://stackoverflow.com/questions/20662023/save-python-random-forest-model-to-file
        dump(rf_model, rf_model_save_path)

    print("Model training complete. Model saved.")


if __name__ == "__main__":
    train_load_model(dataset_name, target_variable, features_to_drop)
