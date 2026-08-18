import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual and Predicted values
actual = np.array([
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Dog',
    'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

predicted = np.array([
    'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog',
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Create Confusion Matrix
conf_matrix = confusion_matrix(actual, predicted)

# Draw Heatmap
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='g',
    xticklabels=['Dog', 'Not Dog'],
    yticklabels=['Dog', 'Not Dog'],
    cmap='RdPu'
)

plt.xlabel("Actual", fontsize=16)
plt.ylabel("Predicted", fontsize=16)
plt.title("Confusion Matrix", fontsize=20)

plt.show()
