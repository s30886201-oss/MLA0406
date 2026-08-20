# Experiment 20: Neural Network Analysis for Multi-Class Classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# 1. Generate Multi-Class Dataset
# --------------------------------------------------

X, y = make_classification(
    n_samples=600,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# --------------------------------------------------
# 2. Split Dataset into Training and Testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 3. Create Neural Network
# --------------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(2, 2),   # 2 hidden layers, 2 neurons each
    activation='identity',       # Linear activation
    learning_rate_init=0.01,     # Learning rate
    max_iter=2000,
    random_state=42
)

# --------------------------------------------------
# 4. Train the Model
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 6. Calculate Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Neural Network Multi-Class Classification")
print("-----------------------------------------")
print("Learning Rate       :", 0.01)
print("Activation Function : Linear")
print("Hidden Layers       :", 2)
print("Neurons per Layer   :", 2)
print("Number of Classes   :", 3)
print("Accuracy            :", accuracy)

# --------------------------------------------------
# 7. Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 8. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# --------------------------------------------------
# 9. Plot Dataset
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap='viridis',
    edgecolors='k'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Multi-Class Dataset")
plt.colorbar(label="Class")
plt.show()

# --------------------------------------------------
# 10. Plot Confusion Matrix
# --------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])
plt.show()
