# pip install seaborn

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("diabetes.csv")
print(data.head())

x = data.drop("Outcome", axis=1)
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
acc = accuracy_score(y_test, y_pred_knn)
print(acc)
error_rate = 1 - acc
print(error_rate)
print(precision_score(y_test, y_pred_knn))
print(recall_score(y_test, y_pred_knn))

tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred No (0)', 'Pred Yes (1)'],
            yticklabels=['Actual No (0)', 'Actual Yes (1)'])
plt.title("Confusion Matrix - KNN on Diabetes Dataset")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# 🧠 THEORY & CONCEPTS
#
# 🔹 Goal:
# Predict whether a patient has diabetes (Outcome = 1) or not (Outcome = 0)
# based on medical features such as Glucose, BMI, Insulin, etc.
#
# 🔹 Dataset:
# diabetes.csv — contains numerical data about patient health.
# 'Outcome' is the target variable (0 = No Diabetes, 1 = Diabetes).
#
# 🔹 Steps:
# 1. Read the dataset using pandas.
# 2. Split the data into features (X) and target (y).
# 3. Divide the data into training and testing sets (80%-20%).
# 4. Normalize the features using StandardScaler for better distance comparison.
# 5. Use the K-Nearest Neighbors (KNN) algorithm for classification.
# 6. Evaluate the model using accuracy, confusion matrix, precision, recall, and specificity.
# 7. Visualize results using a heatmap.
#
# 🔹 KNN Algorithm:
# - KNN is a **non-parametric, instance-based** algorithm.
# - It classifies a new data point based on the majority label among its 'k' nearest neighbors.
# - Uses **Euclidean distance** to measure closeness between points.
# - Here, k=5 means the model looks at 5 nearest neighbors to make a decision.
#
# 🔹 Metrics:
# - **Accuracy**: Overall correctness of the model.
# - **Precision**: Out of all predicted positives, how many are truly positive.
# - **Recall (Sensitivity)**: Out of all actual positives, how many were correctly predicted.
# - **Specificity**: Out of all actual negatives, how many were correctly identified.
# - **Confusion Matrix**: Shows True Positives, True Negatives, False Positives, False Negatives.
#
# 🔹 Output:
# The confusion matrix heatmap shows the model’s performance visually.
# High accuracy and balanced precision/recall indicate a reliable model.
# --------------------------------------------------------------
