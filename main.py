import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from joblib import load
import neural_network
import random_forest 
from neural_network import ImprovedNeuralNet
from random_forest import RandomForestRegressor
import pandas as pd

data_set = "steel_strength"

input_size = neural_network.train_load_model("steel_strength","yield strength",['formula', 'yield strength', 'tensile strength', 'elongation'])
random_forest.train_load_model("steel_strength","yield strength",['formula', 'yield strength', 'tensile strength', 'elongation'])


# Specify the paths to your saved models and scaler
model_nn_path = f"{data_set}_neural_network_model.pth"
model_rf_path = f"{data_set}_rf_model.joblib"
scaler_path = f"{data_set}_model_scaler.joblib"

# Load the models and the scaler
scaler = load(scaler_path)


# Load Neural Network Model
print(input_size)

nn_model = ImprovedNeuralNet(input_size)
nn_model.load_state_dict(torch.load(model_nn_path))
nn_model.eval()

# Load Random Forest Model
rf_model = load(model_rf_path)

# Function to generate a random composition
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

# Function to normalize the composition
def normalize_composition(composition):
    """Normalize the composition so that the total sum is 40."""
    total_sum = sum(composition.values())
    return {k: v * 40 / total_sum for k, v in composition.items()}


feature_columns = ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']

# Function for yield strength prediction adapted for both model types
def predict_yield_strength(composition, model, scaler):
    input_df = pd.DataFrame([composition])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_normalized = scaler.transform(input_df)

    if isinstance(model, ImprovedNeuralNet):
        input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)
        predicted_strength = prediction.item()
    elif isinstance(model, RandomForestRegressor):
        predicted_strength = model.predict(input_normalized)[0]

    return predicted_strength

# Initialize data for plotting
strengths_nn = []
strengths_rf = []

# Search for Best Composition
num_samples = 1000
best_strength_nn = -float('inf')
best_strength_rf = -float('inf')
best_composition_nn = None
best_composition_rf = None

for _ in range(num_samples):
    composition = random_composition()
    normalized_composition = normalize_composition(composition)

    # Predict with Neural Network
    strength_nn = predict_yield_strength(normalized_composition, nn_model, scaler)
    strengths_nn.append(strength_nn)
    if strength_nn > best_strength_nn:
        best_strength_nn = strength_nn
        best_composition_nn = normalized_composition

    # Predict with Random Forest
    strength_rf = predict_yield_strength(normalized_composition, rf_model, scaler)
    strengths_rf.append(strength_rf)
    if strength_rf > best_strength_rf:
        best_strength_rf = strength_rf
        best_composition_rf = normalized_composition

# Display the best results
print("Best Composition (Neural Network):", best_composition_nn)
print("Best Predicted Yield Strength (Neural Network):", best_strength_nn)
print("Best Composition (Random Forest):", best_composition_rf)
print("Best Predicted Yield Strength (Random Forest):", best_strength_rf)

# Plotting the results
# plt.figure(figsize=(12, 6))

# Plotting the yield strengths
# plt.subplot(1, 2, 1)
# plt.plot(strengths_nn, 'b-', label='Neural Network')
# plt.plot(strengths_rf, 'r-', label='Random Forest')
# plt.xlabel('Sample')
# plt.ylabel('Yield Strength')
# plt.title('Yield Strength Predictions')
# plt.legend()

# # Plotting the best composition (Neural Network vs Random Forest)
# plt.subplot(1, 2, 2)
elements = list(best_composition_nn.keys())
values_nn = list(best_composition_nn.values())
values_rf = list(best_composition_rf.values())
bar_width = 0.35
index = np.arange(len(elements))

plt.bar(index, values_nn, bar_width, label='Neural Network', color='b')
plt.bar(index + bar_width, values_rf, bar_width, label='Random Forest', color='r')
plt.xlabel('Element')
plt.ylabel('Concentration')
plt.title('Best Composition Comparison')
plt.xticks(index + bar_width / 2, elements)
plt.legend()

plt.show()
