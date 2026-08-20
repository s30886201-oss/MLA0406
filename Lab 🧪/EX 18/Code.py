# Experiment 18: Neural Network Analysis for Two-Class Classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# 1. Create two-class sample dataset
# --------------------------------------------------

np.random.seed(42)

# Class 0
class0 = np.random.randn(100, 2) + [-2, -2]

# Class 1
class1 = np.random.randn(100, 2) + [2, 2]

# Combine data
X = np.vstack((class0, class1))

# Target labels
y = np.hstack((
    np.zeros(100),
    np.ones(100)
))

# --------------------------------------------------
# 2. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. Create Neural Network
# --------------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(3, 3),   # 2 hidden layers, 3 neurons each
    activation='identity',       # Linear activation
    learning_rate_init=0.03,    # Learning rate
    solver='sgd',
    max_iter=1000,
    random_state=42
)

# --------------------------------------------------
# 4. Train the Neural Network
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 6. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Neural Network Configuration")
print("--------------------------------")
print("Number of Classes     : 2")
print("Learning Rate         : 0.03")
print("Activation Function   : Linear")
print("Hidden Layers         : 2")
print("Neurons per Layer     : 3")
print("--------------------------------")

print("Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 7. Plot Decision Boundary
# --------------------------------------------------

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0"
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1"
)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Neural Network - Two Class Classification")
plt.legend()
plt.show()
