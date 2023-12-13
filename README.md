
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

~~~
Model Case - Prediction of 'yield strength' and then then optimising the composition to get the maximum yield strength
~~~

## ML Model
Following are the architecture of the ML model for Neural Network and Random Forest repectively.

![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/neural_network_architecture.png)
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/steel_strength_decision_tree.png)


## Algorithm

### Neural Network
![Random Forest Model](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/Neural%20Network%20Model.png)

### Random Forest
![Neural Model](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/Random%20Forest%20Model.png)

## How to Use
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/img_1.png)
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/img_2.png)


## ML Model Prediction
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/Predicted%20Composition%20-%20Neural%20Network.png)
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/Predicted%20Composition%20-%20Random%20Forest.png)
![Image](https://github.com/Aroy34/EN.540.635-SC-Final-Project/blob/main/img_3.png)


## Citation
Citations are provided with the code

# Environment
- Python 3
