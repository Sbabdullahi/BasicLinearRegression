# BasicLinearRegression
Binary Logistic Regression Model with Custom Implementation
# Human Action Classification using Binary Logistic Regression with Custom Implementation

This repository contains a custom implementation of a Logistic Regression model to classify human actions across daily life activities. The dataset used for this project is labeled, allowing for supervised learning.

## Table of Contents
- Introduction
- Installation
- Usage
- Dataset
- Model Details
- Results
- License
- Contributing
- Acknowledgements

## Introduction
This project aims to classify human actions based on daily activities using a custom Logistic Regression model. The model is implemented from scratch, providing a clear understanding of the underlying mechanics of logistic regression.

## Installation
To get started, clone this repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/human-action-classification.git
cd human-action-classification
pip install -r requirements.txt

## Usage
1. **Load the dataset**: Ensure your dataset is in the correct format and path.
2. **Run the script**: Execute the main script to train and evaluate the model.
python main.py

## Dataset
The dataset used in this project contains labeled data representing various human actions performed during daily activities. Ensure the dataset includes a `LABEL` column for the target variable.

## Model Details
The Logistic Regression model is implemented from scratch, including:
- Sigmoid activation function
- Forward and backward propagation
- Gradient descent optimization

### Key Functions:
- `sigmoid(z)`: Computes the sigmoid activation.
- `propagate(w, b, X, Y)`: Performs forward and backward propagation.
- `compute_accuracy(y_true, y_pred)`: Calculates the accuracy of predictions.
- `logistic_regression(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate)`: Trains the logistic regression model.

## Results
The model's performance is evaluated based on training and test accuracy. The results are printed during the training process.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements
Special thanks to the contributors and the open-source community for their invaluable support and resources.
