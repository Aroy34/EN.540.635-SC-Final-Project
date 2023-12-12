import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matminer.datasets import load_dataset
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective
from sklearn.ensemble import RandomForestRegressor
import os
from joblib import dump, load
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms


# Improved Neural Network with Normalization
class ImprovedNeuralNet(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.1)
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
    # Load dataset
    df = load_dataset(data_set)

    # Prepare the features and target variable
    y = df[target_variable].values
    X = df.drop(features_to_drop, axis=1)

    # Define input_size
    input_size = X.shape[1]

    # Check if the model file exists
    model_save_path = f"{data_set}_neural_network_model.pth"
    scaler_save_path = f"{data_set}_model_scaler.joblib"
    if os.path.isfile(model_save_path) and os.path.isfile(scaler_save_path):
        print("Loading the existing model...")
        improved_model = ImprovedNeuralNet(input_size)
        improved_model.load_state_dict(torch.load(model_save_path))
        improved_model.eval()
        scaler = load(scaler_save_path)
    else:
        print("No existing model found. Training a new model...")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.33, random_state=1)

        # Normalize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Create datasets and dataloaders
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64)
        
        # Initialize the improved model
        improved_model = ImprovedNeuralNet(X_train.shape[1])

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

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print loss every 10 epochs (or choose a frequency that suits your patience)
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model
        improved_model.eval()
        with torch.no_grad():
            y_train_pred = improved_model(X_train).squeeze()
            y_test_pred = improved_model(X_test).squeeze()

        # Convert predictions to numpy for plotting
        y_train_pred = y_train_pred.numpy()
        y_test_pred = y_test_pred.numpy()
        y_train = y_train.numpy()
        y_test = y_test.numpy()

        # Plot training data predictions
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(y_train, 'r', label='Actual')
        plt.plot(y_train_pred, 'b', label='Predicted')
        plt.title('Training Data')
        plt.legend()

        # Plot testing data predictions
        plt.subplot(1, 2, 2)
        plt.plot(y_test, 'r', label='Actual')
        plt.plot(y_test_pred, 'b', label='Predicted')
        plt.title('Testing Data')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Calculate and print the RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        print("Train RMSE:", train_rmse)
        print("Test RMSE:", test_rmse)

        # Saving the model
        torch.save(improved_model.state_dict(), model_save_path)
        dump(scaler, scaler_save_path)
    
    return input_size
    

if __name__ == "__main__":
    # dataset_name = "steel_strength"
    # target_variable = "yield strength"
    # features_to_drop = ['formula', 'yield strength', 'tensile strength', 'elongation']
    train_load_model(dataset_name, target_variable, features_to_drop)