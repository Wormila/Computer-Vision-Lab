
#Experiment 6: Convolution, Filtering, and Motion Blur in Digital Images 
#Develop a Python program using OpenCV to apply convolution-based filtering techniques and 
#simulate motion blur. 


import cv2
import numpy as np
# -----------------------------------------
# READ IMAGE
# -----------------------------------------
image = cv2.imread("house.jpg")
if image is None:
    print("Error: Image not found")
    exit()
# =================================================
# 1. SMOOTHING FILTER (AVERAGE FILTER)
# =================================================
kernel_smooth = np.ones((5, 5), np.float32) / 25
smoothed = cv2.filter2D(image, -1, kernel_smooth)
# =================================================
# 2. SHARPENING FILTER
# =================================================
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, kernel_sharpen)
# =================================================
# 3. MOTION BLUR
# =================================================
size = 15  # Length of motion
kernel_motion = np.zeros((size, size))
kernel_motion[int((size-1)/2), :] = np.ones(size)
kernel_motion = kernel_motion / size
motion_blur = cv2.filter2D(image, -1, kernel_motion)
# =================================================
# DISPLAY RESULTS
# =================================================
cv2.imshow("Original Image", image)
cv2.imshow("Smoothed Image (Average Filter)", smoothed)
cv2.imshow("Sharpened Image", sharpened)
cv2.imshow("Motion Blur Image", motion_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
