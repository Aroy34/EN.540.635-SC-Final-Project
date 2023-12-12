import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from joblib import load
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import tkinter as tk
from matminer.datasets import load_dataset
import neural_network
import random_forest
from neural_network import NeuralNet

###### Link ######
class Utility:
    """Class to handle user interactions for selecting the datasets.

    Attributes:
        selections (dict): to store user selections for datasets.
        datasets (list): to store list of available datasets .

    Methods:
        dataset_button_clicked(dataset_name): deccides what to do when a dataset button is clicked.
        user_interface(): Sets up the main user interface.
    """
    def __init__(self):
        self.selections = {}
        self.datasets = [
            "elastic_tensor_2015",
            "matbench_steels",
            "steel_strength",
            "superconductivity2018",
            "More Datasets To Come...",]

    def dataset_button_clicked(self, dataset_name, root_window):
        """Deccides what to do when a dataset button is clicked. 

        Loads the selected dataset from matminer 
        and opens a new window for selcting the target and dropped columns.

        Args:
            dataset_name (str): The name of the dataset selected by the user.
        """
        # Load the dataset
        df = load_dataset(dataset_name)
        print(f"Loaded dataset: {dataset_name}")

        new_window = tk.Toplevel()
        new_window.title(f"Options for {dataset_name}")
        target_vars = {}
        drop_vars = {}

        # Use a for loop to populate the dictionaries
        for col in df.columns:
            target_vars[col] = tk.BooleanVar()
            drop_vars[col] = tk.BooleanVar()

        # Create a grid of options
        tk.Label(new_window, text="Column").grid(row=0, column=0)
        tk.Label(new_window, text="Target").grid(row=0, column=1)
        tk.Label(new_window, text="To be Dropped").grid(row=0, column=2)

        for i, col in enumerate(df.columns):
            tk.Label(new_window, text=col).grid(row=i + 1, column=0)

            # Checkbox for target column
            chk_box_target = tk.Checkbutton(
                new_window, variable=target_vars[col])
            chk_box_target.grid(row=i + 1, column=1)

            # Checkbox for column to drop
            chk_box_drop = tk.Checkbutton(new_window, variable=drop_vars[col])
            chk_box_drop.grid(row=i + 1, column=2)

        def apply():
            """Decides what to do when user clicks ont he 'Apply'
            Reads the boxes which have been ticked for each column, updates the
            `selections` dictionary, and writes these selections to a text file.
            
            Closes the all the windows.
            """

            target_columns = []
            columns_to_drop = []

            # Setting up the tick boxes
            for col, var in target_vars.items():
                if var.get():
                    target_columns.append(col)

            for col, var in drop_vars.items():
                if var.get():
                    columns_to_drop.append(col)
            self.selections[dataset_name] = {
                "Target": ", ".join(target_columns),
                "To be Dropped": ", ".join(columns_to_drop),
            }
            print(f"Selections for {dataset_name}:\
                {self.selections[dataset_name]}")

            # Save the selections to a text file
            with open("dataset_selections.txt", "w") as file:
                file.write(f"Dataset: {dataset_name}\n")
                file.write(f"Target Columns: {', '.join(target_columns)}\n")
                file.write(f"Columns to Drop: {', '.join(columns_to_drop)}\n")
                file.write("\n")

            new_window.destroy()
            # root_window.destroy()

        # Apply button
        apply_button = tk.Button(new_window, text="Apply", command=apply)
        apply_button.grid(row=len(df.columns) + 1, column=0, columnspan=3)

    def user_interface(self):
        """Starts thw page and displays the main user interface window.

        The window lists available datasets as buttons, user can only seelct one dataset
        """
        root = tk.Tk()
        root.title("Prediction of Material Properties")

        for dataset_name in self.datasets:
            button = tk.Button(
                root,
                text=dataset_name,
                command=lambda n=dataset_name, r=root: self.dataset_button_clicked(n, r),
            )
            button.pack()

        root.mainloop()
    


class ReadUserInput:
    """A class to read user selections from a file.

    Attributes:
        filename (str): The name of the file containing the selections user made

    Methods:
        read_data(): Reads the name of selected dataset,
        target columns, and columns to drop for the ML model
    """
    def __init__(self, filename):
        self.filename = filename

    def read_data(self):
        """Reads the user data selections from the specified file.

        Parses the file for dataset name, target columns, and columns to be dropped.

        Returns:
            lists: containing the dataset name, target column names and list of columns to be dropped.
        """
        data_set = []
        target = []
        to_be_dropped = []

        # Open and read the file line by line
        with open(self.filename, "r") as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith("Dataset:"):
                data_set = line.split(":")[1].strip()
            elif line.startswith("Target Columns:"):
                target_cols = line.split(":")[1].split(",")
                target = []
                for col in target_cols:
                    target.append(col.strip())
            elif line.startswith("Columns to Drop:"):
                drop_cols = line.split(":")[1].split(",")
                for col in drop_cols:
                    to_be_dropped.append(col.strip())

        return data_set, target, to_be_dropped


class CompositionGenerator:
    """Generates random compositions for material analysis. with higher conc. for Nickel and Chromiums

    Attributes:
        feature_columns (list): A list of element names to be used in composition generation.

    Methods:
        random_composition(): Generates a random composition using the defined elements.
    """
    def __init__(self):
        self.feature_columns = [
            "c",
            "mn",
            "si",
            "cr",
            "ni",
            "mo",
            "v",
            "n",
            "nb",
            "co",
            "w",
            "al",
            "ti",
        ]

    def random_composition(self):
        """Generates a random composition for material analysis.

        Returns:
            dict: dictionary representing the material composition with elements as keys and their concentrations as values.
        """
        composition = {}
        for element in self.feature_columns:
            if element in ["cr", "co"]:
                composition[element] = random.uniform(10.0, 25.0)
            else:
                composition[element] = random.uniform(0.0, 1.0)
        return composition


# Define a class to predict yield strength using a neural network
class NeuralNetworkPredictor:
    """Predicts material properties using a neural network model.

    Attributes:
        nn_model: The neural network model for prediction.
        feature_columns (list): A list of feature columns for input to the model.

    Methods:
        predict_yield_strength(composition): Predicts the yield strength of a given material composition.
    """
    def __init__(self, nn_model, feature_columns):
        self.nn_model = nn_model
        self.feature_columns = feature_columns

    def predict_yield_strength(self, composition):
        """Predicts the yield strength of a given material composition using a neural network model.

        Args:
            composition (dict): A dictionary containing the material composition.

        Returns:
            float: The predicted yield strength of the material.
        """
        
        input_df = pd.DataFrame([composition])

        # input_normalized = self.scaler.transform(input_df)
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

        # Evaluate the neural network model
        self.nn_model.eval()
        with torch.no_grad():
            prediction = self.nn_model(input_tensor)
        predicted_strength = prediction.item()

        return predicted_strength


class RandomForestPredictor:
    """Predicts material properties using a Random Forest model.

    Attributes:
        rf_model: The Random Forest model for prediction.
        feature_columns (list): A list of feature columns for input to the model.

    Methods:
        predict_yield_strength(composition): Predicts the yield strength of a given material composition.
    """
    def __init__(self, rf_model, feature_columns):
        self.rf_model = rf_model
        self.feature_columns = feature_columns

    def predict_yield_strength(self, composition):
        """Predicts the yield strength of a given material composition using a Random Forest model.

        Args:
            composition (dict): A dictionary containing the material composition.

        Returns:
            float: The predicted yield strength of the material.
        """
        input_df = pd.DataFrame([composition])

        # Predict using the RandomForest model
        predicted_strength = self.rf_model.predict(input_df)[0]

        return predicted_strength


# Define a class to optimize yield strength predictions
class YieldStrengthOptimizer:
    """Optimizes yield strength predictions using neural network and random forest models.

    Attributes:
        best_strength_nn (float): Best yield strength predicted by neural network.
        best_strength_rf (float): Best yield strength predicted by random forest.
        best_composition_nn (dict): Best composition as predicted by neural network.
        best_composition_rf (dict): Best composition as predicted by random forest.

    Methods:
        find_best_composition(): Finds the best composition for yield strength using both models.
    """

    def __init__(self,
                 num_samples, composition_generator,
                 nn_predictor, rf_predictor):
        self.num_samples = num_samples
        self.composition_generator = composition_generator
        self.nn_predictor = nn_predictor
        self.rf_predictor = rf_predictor
        self.best_strength_nn = 0
        self.best_strength_rf = 0
        self.best_composition_nn = []
        self.best_composition_rf = []
    
    ###### Link ######
    def find_best_composition(self):
        """Finds the best composition for yield strength using both neural network and random forest models.

        Iterates over a number of random compositions, predicts their yield strength, and identifies the best one.

        Returns:
            list: a list containing the best compositions and their corresponding strengths as predicted by both models.
        """
        strengths_nn = []
        strengths_rf = []

        for itr in range(self.num_samples):
            composition = self.composition_generator.random_composition()

            # Predict with Neural Network
            strength_nn = self.nn_predictor.predict_yield_strength(composition)
            strengths_nn.append(strength_nn)
            # Bubble sort
            if strength_nn > self.best_strength_nn:
                self.best_strength_nn = strength_nn
                self.best_composition_nn = composition

            # Predict with Random Forest
            strength_rf = self.rf_predictor.predict_yield_strength(composition)
            strengths_rf.append(strength_rf)
            if strength_rf > self.best_strength_rf:
                self.best_strength_rf = strength_rf
                self.best_composition_rf = composition

        return (self.best_composition_nn,
                self.best_strength_nn,
                self.best_composition_rf,
                self.best_strength_rf)


# Define a class to plot the composition comparison
class Plotter:
    """Plots comparisons of material compositions as predicted by different models.

    Attributes:
        elements (list): A list of elements in the composition.
        values_nn (list): Predicted values of elements by the Neural Network.
        values_rf (list): Predicted values of elements by the Random Forest.

    Methods:
        plot_composition_comparison(): Plots a comparison of compositions from both models.
    """
    def __init__(self, elements, values_nn, values_rf):
        self.elements = elements
        self.values_nn = values_nn
        self.values_rf = values_rf

    def plot_composition_comparison(self):
        """Plots a comparison of material compositions as predicted by models.

        Generates bar charts for the element concentrations in the best compositions predicted by each model.
        """
        index = np.arange(len(self.elements))

        plt.figure(1)
        plt.bar(index, self.values_nn, 0.35, label="Neural Network", color="b")
        plt.xlabel("Element")
        plt.ylabel("Concentration")
        plt.title("Best Composition Comparison")
        plt.xticks(index, self.elements)
        plt.legend()
        plt.savefig("Predicted Composition - Neural Network.png")

        plt.figure(2)
        plt.bar(index, self.values_rf, 0.35, label="Random Forest", color="r")
        plt.xlabel("Element")
        plt.ylabel("Concentration")
        plt.title("Best Composition Comparison")
        plt.xticks(index, self.elements)
        plt.legend()
        plt.savefig("Predicted Composition - Random Forest.png")
        plt.show()


if __name__ == "__main__":
    # Read data from a file
    u = Utility()
    u.user_interface()  # Add refrence
    data_reader = ReadUserInput("dataset_selections.txt")
    data_set, target, to_be_dropped = data_reader.read_data()

    # Load and train neural network and random forest models
    input_size = neural_network.train_load_model(
        data_set, target[0], to_be_dropped)
    print(input_size)
    random_forest.train_load_model(data_set, target[0], to_be_dropped)

    # Load saved models and scaler
    model_nn_path = f"{data_set}_neural_network_model.pth"
    model_rf_path = f"{data_set}_rf_model.joblib"

    nn_model = NeuralNet(input_size)
    nn_model.load_state_dict(torch.load(model_nn_path))
    nn_model.eval()

    rf_model = load(model_rf_path)

    composition_generator = CompositionGenerator()
    feature_columns = [
        "c",
        "mn",
        "si",
        "cr",
        "ni",
        "mo",
        "v",
        "n",
        "nb",
        "co",
        "w",
        "al",
        "ti",
    ]

    nn_predictor = NeuralNetworkPredictor(nn_model, feature_columns)
    rf_predictor = RandomForestPredictor(rf_model, feature_columns)

    optimizer = YieldStrengthOptimizer(
        1000, composition_generator, nn_predictor, rf_predictor
    )

    (
        best_composition_nn,
        best_strength_nn,
        best_composition_rf,
        best_strength_rf,
    ) = optimizer.find_best_composition()

    # Display the best results
    print("Best Composition (Neural Network):", best_composition_nn)
    print("Best Predicted Yield Strength (Neural Network):", best_strength_nn)
    print("Best Composition (Random Forest):", best_composition_rf)
    print("Best Predicted Yield Strength (Random Forest):", best_strength_rf)

    plotter = Plotter(best_composition_nn.keys(),
                      best_composition_nn.values(),
                      best_composition_rf.values())

    plotter.plot_composition_comparison()
