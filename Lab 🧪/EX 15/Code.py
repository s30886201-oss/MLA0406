import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load image from Google Drive
# Change this path according to your image location
img = cv2.imread('/content/drive/MyDrive/dog.jpeg')

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image. Please check the file path.")

else:
    # Convert image from BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image pixels to float32
    pixels = np.float32(rgb_img.reshape((-1, 3)))

    # Define termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # Number of clusters
    K = 3

    # Apply K-Means clustering
    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert cluster centers to uint8
    centers = np.uint8(centers)

    # Create segmented image
    segmented_img = centers[labels.flatten()].reshape(rgb_img.shape)

    # Display original and segmented images
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title('Original Image')
    plt.axis('off')

    # Segmented image
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img)
    plt.title('Segmented Image (K-Means)')
    plt.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Show images
    plt.show()
