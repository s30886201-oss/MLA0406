# Experiment 19: Neural Network Analysis for Circular Data Class

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate circular dataset
X, y = make_circles(
    n_samples=500,
    noise=0.05,
    factor=0.5,
    random_state=42
)

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.50,
    random_state=42,
    stratify=y
)

# 3. Create Neural Network
# 2 hidden layers, 3 neurons each
# Linear activation function
model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',
    solver='sgd',
    learning_rate_init=0.03,
    max_iter=1000,
    random_state=42
)

# 4. Train the Neural Network
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Experiment 19")
print("-----------------------------")
print("Learning Rate       : 0.03")
print("Activation Function : Linear")
print("Hidden Layers       : 2")
print("Hidden Neurons      : 3")
print("-----------------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Plot original circular data
plt.figure(figsize=(7, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap='viridis',
    edgecolors='k'
)

plt.title("Original Circular Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Class")
plt.grid(True)
plt.show()

# 8. Plot test data with predictions
plt.figure(figsize=(7, 6))

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred,
    cmap='viridis',
    edgecolors='k'
)

plt.title("Neural Network Predictions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Predicted Class")
plt.grid(True)
plt.show()
