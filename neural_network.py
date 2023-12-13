import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from matminer.datasets import load_dataset
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random

# https://stackoverflow.com/questions/59003591/how-to-implement-dropout-in-pytorch-and-where-to-apply-it


class NeuralNet(nn.Module):
    """A neural network for predicting material properties.

    Attributes:
        layer1 (nn.Linear): First linear layer.
        layer2 (nn.Linear): Second linear layer.
        layer3 (nn.Linear): Third linear layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        batchnorm1 (nn.BatchNorm1d): Batch normalization for the first layer.
        batchnorm2 (nn.BatchNorm1d): Batch normalization for the second layer.
        output (nn.Linear): Output linear layer.
        relu (nn.ReLU): ReLU activation function.

    Methods:
        forward(x): Defines the forward pass of the neural network.
    """
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.output(x)
        return x


def train_load_model(data_set, target_variable, features_to_drop):
    """Loads or trains a neural network model for material property prediction.

    Loads an existing model if available, otherwise
    trains a new model. Also plots and
    saves the training and testing results.

    Args:
        data_set (str): The name of the dataset to use.
        target_variable (str): The target variable to predict.
        features_to_drop (list): List of feature
        columns to drop from the dataset.

    Returns:
        int: The input size for the neural network model.
    """
    df = load_dataset(data_set)

    y = df[target_variable].values
    X = df.drop(features_to_drop, axis=1)

    input_size = X.shape[1]

    model_save_path = f"{data_set}_neural_network_model.pth"

    if os.path.isfile(model_save_path):
        print("Loading the existing model...")
        improved_model = NeuralNet(input_size)
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        improved_model.load_state_dict(torch.load(model_save_path))
        improved_model.eval()
    else:
        print("No existing models found.... training a new model...")

        # https://stackoverflow.com/questions/75927752/pytorch-deeplearning
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=1
        )

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64)

        # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
        improved_model = NeuralNet(X_train.shape[1])

        # Loss function
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = optim.Adam(improved_model.parameters(), lr=0.001)

        # Training the Model
        num_epochs = 1000
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                # Forward pass
                outputs = improved_model(inputs)
                loss = criterion(outputs.squeeze(), targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}],\
                    Loss: {loss.item():.4f}"
                )

        # Evaluate the model
        improved_model.eval()
        with torch.no_grad():
            y_train_pred = improved_model(X_train).squeeze()
            y_test_pred = improved_model(X_test).squeeze()

        y_train_pred = y_train_pred.numpy()
        y_test_pred = y_test_pred.numpy()
        y_train = y_train.numpy()
        y_test = y_test.numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(y_train, "r", label="Actual")
        plt.plot(y_train_pred, "b", label="Predicted")
        plt.title("Training Data - Predicted vs. Actual [Neural Network]")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(y_test, "r", label="Actual")
        plt.plot(y_test_pred, "b", label="Predicted")
        plt.title("Testing Data - Predicted vs. Actual[Neural Network]")
        plt.legend()

        plt.tight_layout()
        plt.savefig("Neural Network Model.png")
        plt.show()

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        print("Train RMSE:", train_rmse)
        print("Test RMSE:", test_rmse)

        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(improved_model.state_dict(), model_save_path)

    return input_size


if __name__ == "__main__":
    train_load_model(dataset_name, target_variable, features_to_drop)
