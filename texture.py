#HEISNAM KAVIT SINGH
#Redg. No. NDU202400083 MTECH-AI NIELIT IMPHAL
#Experiment 8: Texture Segmentation using Gabor Filters. 
#Implement texture segmentation using a bank of Gabor filters in OpenCV 


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("house.jpg")

if image is None:
    print("Error: Image not found")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ------------------- Gabor Filter Bank -------------------

gabor_responses = []

# Gabor filter parameters
ksize = 31
sigma = 4.0
lambd = 10.0
gamma = 0.5
psi = 0

# Different orientations
orientations = [0, 45, 90, 135]

for theta in orientations:
    theta_rad = theta * np.pi / 180
    kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta_rad, lambd, gamma, psi, ktype=cv2.CV_32F
    )
    
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    gabor_responses.append(filtered)

# ------------------- Combine Responses -------------------

# Take maximum response across all orientations
texture_response = np.max(gabor_responses, axis=0)

# ------------------- Threshold for Segmentation -------------------

_, segmented = cv2.threshold(
    texture_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# ------------------- Display Results -------------------

cv2.imshow("Original Image", image)
cv2.imshow("Texture Response", texture_response)
cv2.imshow("Texture Segmented Image", segmented)

cv2.waitKey(0)
cv2.destroyAllWindows()
