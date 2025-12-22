
#Experiment 2: Image Transformations 
#Develop a Python program using OpenCV to implement geometric transformations including 
#scaling, translation, and rotation. Display the original and transformed images. 


import cv2
import numpy as np

# Read the image
image = cv2.imread("house.jpg")

# Check if image is loaded
if image is None:
    print("Error: Image not found")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# ------------------- Scaling -------------------
scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5)

# ------------------- Translation -------------------
# Translation matrix
tx, ty = 100, 50
translation_matrix = np.float32([[1, 0, tx],
                                 [0, 1, ty]])
translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

# ------------------- Rotation -------------------
# Rotation matrix
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# ------------------- Display Images -------------------
cv2.imshow("Original Image", image)
cv2.imshow("Scaled Image", scaled_image)
cv2.imshow("Translated Image", translated_image)
cv2.imshow("Rotated Image", rotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
