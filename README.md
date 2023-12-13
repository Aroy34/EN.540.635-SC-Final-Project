
# EN.540.635-SC-Final-Project

## Description
This project involves the development of machine learning models to analyze and predict properties from a selected dataset. It includes implementations of a Neural Network and a Random Forest model, integrated into a main script that handles data processing, model training, and user interactions.

## Data
The data for the project is sourced from the `matminer` datasets, used for training both the Neural Network and Random Forest models.
NOTE: Dataset will be downloaded on the go

## Codes
Following scripts make up the working code

### main.py
The main script of the project, responsible for integrating various components. It handles user interactions, data processing, and model operations. It utilizes libraries like `torch`, `numpy`, `pandas`, and `matplotlib`, and orchestrates the workflow involving neural network and random forest models.

### neural_network.py
This script is dedicated to building, training, and evaluating a Neural Network model. It includes data preprocessing, model construction, training phases, and performance evaluation. Libraries such as `torch`, `pandas`, `matplotlib`, and `sklearn` are used.

### random_forest.py
This script focuses on the Random Forest model. It handles data preprocessing, training, and evaluating the model's performance. It uses libraries like `pandas`, `sklearn`, and `matplotlib` and assesses the model using metrics like mean squared error and R2 score.

## Pseudocode

### Start of the Program
- **Load Data**:
  - Get data from `matminer`.
  - Generate the features if necessary, identify the target values and the fatueres which can help in prediction
  - Prepare the data by splitting it into two parts: one for training (learning) and one for testing.

### Neural Network model
1. **Build the Neural Network**:
   - Create a structure of layers
   - Set up the number of layers and their connections.
2. **Train the Neural Network**:
   - Feed the training data into the network.
   - Allow the network to adjust itself based on this data.
3. **Test the Neural Network**:
   - Use the test data to evaluate how well the network learned.
   - Measure its performance and show it in the form of plot

### Random Forest model
1. **Set Up the Random Forest**:
   - Prepare a group of decision trees.
2. **Train the Random Forest**:
   - Train the forest using the prepared training data.
3. **Test the Random Forest**:
   - Assess the model's learning using the test data.
   - Calculate mean squared error and R^2 score for evaluation.

~~~
Model Case - Prediction of 'yield strength' and then then optimising the composition to get the maximum yield strength
~~~

## How to use the GUI
The GUI is a simple and the sole purpose of the GUi is to select different database and then identify the target variables

![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/img_1.png)

![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/img_2.png)

## ML Model
Following are the architecture of the ML model for Neural Network and Random Forest repectively.
Neural Network:
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/neural_network_architecture.png)

Random Forest:
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/steel_strength_decision_tree.png)

### Neural Network
![Random Forest Model](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/Neural%20Network%20Model.png)

### Random Forest
![Neural Model](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/Random%20Forest%20Model.png)

## ML Model Prediction
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/Predicted%20Composition%20-%20Neural%20Network.png)

![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/Predicted%20Composition%20-%20Random%20Forest.png)

![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/img_3.png)


## Citation
Citations are provided with the code

# Environment
- Python 3
