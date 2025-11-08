import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Churn_Modelling.csv")

X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y = df["Exited"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --------------------------------------------------------------
# 🧠 THEORY & CONCEPTS
#
# 🔹 Goal:
# Predict whether a bank customer will leave (churn) or stay based on their data.
#
# 🔹 Dataset:
# Churn_Modelling.csv — contains details like CreditScore, Geography, Gender, Age, Balance, etc.
# 'Exited' column (0 or 1) is the target variable showing whether the customer left.
#
# 🔹 Steps:
# 1. Read the dataset using pandas.
# 2. Drop unnecessary columns (like CustomerId, Surname) that don't affect prediction.
# 3. Convert categorical columns ('Geography', 'Gender') into numeric using pd.get_dummies().
# 4. Split the dataset into training (80%) and testing (20%) sets.
# 5. Normalize features using StandardScaler so that all inputs are on the same scale.
# 6. Train a Multi-Layer Perceptron (MLP) classifier — a type of feed-forward neural network.
# 7. Evaluate performance using accuracy and confusion matrix.
#
# 🔹 MLP (Multi-Layer Perceptron):
# - It consists of input, hidden, and output layers.
# - Each neuron performs a weighted sum of inputs, applies an activation function, and passes it forward.
# - It learns patterns through backpropagation — adjusting weights to minimize prediction error.
#
# 🔹 Output:
# - Accuracy shows how well the model predicts correctly.
# - Confusion Matrix shows true positives, true negatives, false positives, and false negatives.
#
# 🔹 Common Improvements:
# - Tune hidden layers, neurons, learning rate, or increase max_iter for better convergence.
# - Handle class imbalance (if any) using class_weight or oversampling.
# --------------------------------------------------------------
