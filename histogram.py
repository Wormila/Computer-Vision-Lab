
#Experiment 3: Histogram Analysis 
#Write a program to compute and plot the histogram of a grayscale image and perform histogram 
#equalization. 


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
image = cv2.imread("house.jpg", cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    print("Error: Image not found")
    exit()

# ------------------- Histogram Calculation -------------------
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# ------------------- Histogram Equalization -------------------
equalized_image = cv2.equalizeHist(image)

# Histogram of equalized image
hist_eq = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# ------------------- Display Images -------------------
cv2.imshow("Original Grayscale Image", image)
cv2.imshow("Histogram Equalized Image", equalized_image)

# ------------------- Plot Histograms -------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image Histogram")
plt.plot(hist)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.title("Equalized Image Histogram")
plt.plot(hist_eq)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
