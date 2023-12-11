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

# Function to predict yield strength
def predict_yield_strength(composition, model, scaler):
    # Assuming 'composition' is a dictionary with features as keys and their values
    # Convert composition to a DataFrame
    input_df = pd.DataFrame([composition])
    
    # Ensure the input columns match the training features
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Normalize the features using the same scaler used for training
    input_normalized = scaler.transform(input_df)
    
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Convert the prediction tensor to a scalar
    predicted_strength = prediction.item()
    
    return predicted_strength

# Load dataset
df = load_dataset("steel_strength")

# Prepare the features and target variable
y = df["yield strength"].values
X = df.drop(['formula', 'yield strength', 'tensile strength', 'elongation'], axis=1)

# Define input_size
input_size = X.shape[1]

# Check if the model file exists
model_save_path = "improved_steel_strength_model.pth"
scaler_save_path = "steel_strength_scaler.joblib"
if os.path.isfile(model_save_path) and os.path.isfile(scaler_save_path):
    print("Loading the existing model...")
    improved_model = ImprovedNeuralNet(input_size)
    improved_model.load_state_dict(torch.load(model_save_path))
    improved_model.eval()
    scaler = load(scaler_save_path)
else:
    print("No existing model found. Training a new model...")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=1)

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


new_composition = {
    'c': 1.0, 
    'mn': 0.0, 
    'si': 1.0, 
    'cr': 10.0, 
    'ni': 0.0, 
    'mo': 0.0, 
    'v': 0.0, 
    'n': 1.0, 
    'nb': 1.0, 
    'co': 10.0, 
    'w': 0.0, 
    'al': 0.0, 
    'ti': 0.0
}

# Get the predicted yield strength
predicted_strength = predict_yield_strength(new_composition, improved_model, scaler)
print(f"The predicted yield strength is: {predicted_strength}")
# quit()

def random_composition():
    """Generate a random composition within specified bounds."""
    composition = {
        'c': random.uniform(0.0, 1.0),
        'mn': random.uniform(0.0, 1.0),
        'si': random.uniform(0.0, 1.0),
        'cr': random.uniform(10.0, 20.0),
        'ni': random.uniform(0.0, 1.0),
        'mo': random.uniform(0.0, 1.0),
        'v': random.uniform(0.0, 1.0),
        'n': random.uniform(0.0, 1.0),
        'nb': random.uniform(0.0, 1.0),
        'co': random.uniform(10.0, 20.0),
        'w': random.uniform(0.0, 1.0),
        'al': random.uniform(0.0, 1.0),
        'ti': random.uniform(0.0, 1.0),
    }
    return composition

def normalize_composition(composition):
    """Normalize the composition so that the total sum is 40."""
    total_sum = sum(composition.values())
    return {k: v * 40 / total_sum for k, v in composition.items()}

# Parameters for the random search
num_samples = 1000
best_strength = -float('inf')
best_composition = None
strengths = []

# Perform the random search
for _ in range(num_samples):
    print(_)
    composition = random_composition()
    normalized_composition = normalize_composition(composition)
    strength = predict_yield_strength(normalized_composition, improved_model, scaler)
    strengths.append(strength)
    
    if strength > best_strength:
        best_strength = strength
        best_composition = normalized_composition

# Plotting the results
plt.figure(figsize=(10, 5))

# Plotting the yield strengths
plt.subplot(1, 2, 1)
plt.bar(range(num_samples), strengths, color='blue')
plt.xlabel('Sample')
plt.ylabel('Yield Strength')
plt.title('Yield Strength of Each Sample')

# Plotting the best composition
plt.subplot(1, 2, 2)
elements = list(best_composition.keys())
values = list(best_composition.values())
plt.bar(elements, values, color='green')
plt.xlabel('Element')
plt.ylabel('Concentration')
plt.title('Best Composition')

plt.tight_layout()


print("Best Composition:", best_composition)
print("Best Predicted Yield Strength:", best_strength)
plt.show()

