
#Experiment 5: Image Enhancement 
#Develop a Python program using OpenCV to enhance the quality of a digital image by applying 
#spatial domain filtering techniques such as smoothing and sharpening 


import cv2
import numpy as np
# Read the image
image = cv2.imread("house.jpg")
# Check if image is loaded
if image is None:
    print("Error: Image not found")
    exit()
# ------------------- Smoothing Filters -------------------
# Using Gaussian Blur
smoothed_image = cv2.GaussianBlur(image, (7,7), 0)
# ------------------- Sharpening Filter -------------------
# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
# ------------------- Display Images -------------------
cv2.imshow("Original Image", image)
cv2.imshow("Smoothed Image", smoothed_image)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
