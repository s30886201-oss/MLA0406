import numpy as np
import cv2
from matplotlib import pyplot as plt
from google.colab import drive

# Mount Google Drive
try:
    drive.mount('/content/drive')
except:
    print("Google Drive is already mounted.")

# Load image from Google Drive
img = cv2.imread('/content/drive/MyDrive/dog.jpeg')

# Check if image was loaded successfully
if img is None:
    print("Error: Could not load image. Please check the file path.")

else:
    # Convert BGR to RGB
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Create kernel
    kernel = np.ones((2, 2), np.uint8)

    # Apply morphological closing
    closing = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Apply dilation
    sure_bg = cv2.dilate(
        closing,
        kernel,
        iterations=3
    )

    # Display results
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    # Grayscale image
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    # Otsu threshold
    plt.subplot(2, 3, 3)
    plt.imshow(thresh, cmap="gray")
    plt.title("Otsu's Threshold")
    plt.axis("off")

    # Morphological closing
    plt.subplot(2, 3, 4)
    plt.imshow(closing, cmap="gray")
    plt.title("MorphologyEx: Closing (2x2)")
    plt.axis("off")

    # Dilation
    plt.subplot(2, 3, 5)
    plt.imshow(sure_bg, cmap="gray")
    plt.title("Dilation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Save dilation image
    plt.imsave("dilation.png", sure_bg, cmap="gray")

    print("Dilation image saved as dilation.png")
