# Import of necessary libraries and packages
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


# Activation function: Sigmoid function with clipping
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1.0 / (1 + np.exp(-z))


# Computation of gradient descent to iteratively optimize the loss function
# Forward and backward propagation
def propagate(w, b, X, Y):
    m = X.shape[1]
    epsilon = 1e-8  # Small value to avoid log(0)

    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1. / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

    # Backward propagation
    dw = (1. / m) * np.dot(X, (A - Y).T)
    db = (1. / m) * np.sum(A - Y)

    grads = {"dw": dw, "db": db}
    return grads, cost


# Accuracy calculation
def compute_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred) * 100


# Logistic Regression Model Development
def logistic_regression(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.001):
    # Initialize weights and bias
    dim = X_train.shape[0]
    w, b = np.zeros((dim, 1)), 0.0

    # Train the regression model
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X_train, Y_train)
        dw, db = grads["dw"], grads["db"]

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost:.6f}")

    # Logistic regression model predictions
    Y_train_pred = sigmoid(np.dot(w.T, X_train) + b)
    Y_test_pred = sigmoid(np.dot(w.T, X_test) + b)

    # Calculate model accuracies
    train_accuracy = compute_accuracy(Y_train, Y_train_pred)
    test_accuracy = compute_accuracy(Y_test, Y_test_pred)

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return costs


# Model validation on real data set for classification problems


# Load dataset
# data = pd.read_csv("/mnt/data/SampleForTest.csv")
print("Dataset loaded successfully")

# Validate and preprocess data
if 'LABEL' not in data.columns:
    raise ValueError("The dataset must contain a 'LABEL' column for the target.")

# Ensure 'LABEL' is binary
data['LABEL'] = data['LABEL'].apply(lambda x: 1 if x > 0 else 0)

# Define features and target
X = data.drop(columns=["LABEL"])
Y = data["LABEL"].values.reshape(1, -1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).T

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled.T, Y.T, test_size=0.2, random_state=42)
Y_train = Y_train.reshape(1, -1)
Y_test = Y_test.reshape(1, -1)

# Train logistic regression for LABEL classification
print("\n--- Training Logistic Regression for LABEL Classification ---")
costs = logistic_regression(X_train.T, Y_train, X_test.T, Y_test, num_iterations=2000, learning_rate=0.001)

del costs
