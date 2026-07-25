from PIL import Image
import numpy as np

# Load image
img = Image.open("leaf.jpg")

# Convert image to RGB
img = img.convert("RGB")

# Convert to NumPy array
pixels = np.array(img)

# Calculate average RGB values
avg_red = np.mean(pixels[:, :, 0])
avg_green = np.mean(pixels[:, :, 1])
avg_blue = np.mean(pixels[:, :, 2])

print("Average Red   :", round(avg_red, 2))
print("Average Green :", round(avg_green, 2))
print("Average Blue  :", round(avg_blue, 2))

# Identify dominant color
if avg_green > avg_red and avg_green > avg_blue:
    print("Leaf Color: GREEN")
elif avg_red > avg_green and avg_red > avg_blue:
    print("Leaf Color: BROWN/RED")
elif avg_blue > avg_red and avg_blue > avg_green:
    print("Leaf Color: BLUE")
else:
    print("Leaf Color: MIXED")
