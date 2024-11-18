import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Get the current working directory
current_dir = os.getcwd()

# Specify the desired file name for the plot
plot_file = "accuracy_plot.png"

# Create the absolute file path for the plot
plot_path = os.path.join(current_dir, plot_file)

# Load the Iris dataset
data = load_iris(as_frame=True)
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create lists to store accuracies and parameter values
accuracies = []
n_estimators_list = []
max_depth_list = []

# Create a function to train and log metrics
def train_and_log_metrics(n_estimators, max_depth, iteration):
    # Create and train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and generate a classification report
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    n_estimators_list.append(n_estimators)
    max_depth_list.append(max_depth)

    # Log metrics and artifacts to MLflow for each iteration
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })

        # Log the accuracy for this iteration
        mlflow.log_metric(f"accuracy_iteration_{iteration}", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log the classification report as an artifact for the final iteration
        if iteration == 4:  # Assuming 0-based indexing, the final iteration is 4
            with open("final_classification_report.txt", "w") as text_file:
                text_file.write(classification_report(y_test, y_pred))
            mlflow.log_artifact("final_classification_report.txt")

# Continuously monitor and adjust the pipeline
for iteration in range(5):
    n_estimators = 100 + iteration * 50
    max_depth = 5 + iteration
    train_and_log_metrics(n_estimators, max_depth, iteration)

# Create a single plot showing all the accuracies
plt.figure()
plt.title("Accuracy Over Time")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.plot(range(1, 6), accuracies, marker='o', linestyle='-')
plt.xticks(range(1, 6), [f"Iteration {i}" for i in range(1, 6)])
plt.grid(True)

# Save the plot with the absolute file path
plt.savefig(plot_path)

# Log metrics to MLflow
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_params({
        "n_estimators_list": n_estimators_list,
        "max_depth_list": max_depth_list
    })

    # Log the final accuracy as the MLflow accuracy metric
    mlflow.log_metric("accuracy", accuracies[-1])

    # Log the plot as an artifact to MLflow using the absolute file path
    mlflow.log_artifact(plot_path)
