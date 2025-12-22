
#Experiment 9: Image Segmentation 
#Design and implement an image segmentation framework using both classical (thresholding, region
#based) and learning-based approaches. Compare their performance on natural images. 


import cv2
import numpy as np
# ------------------- Read Image -------------------
image = cv2.imread("house.jpg")
if image is None:
    print("Error: Image not found")
    exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 1. CLASSICAL METHOD: THRESHOLDING
_, thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
# 2. CLASSICAL METHOD: REGION-BASED (WATERSHED)
# Noise removal
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(
    dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
watershed_image = image.copy()
markers = cv2.watershed(watershed_image, markers)
watershed_image[markers == -1] = [0, 0, 255]
# 3. LEARNING-BASED METHOD: K-MEANS CLUSTERING
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
k = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(
    pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)
centers = np.uint8(centers)
segmented_kmeans = centers[labels.flatten()]
segmented_kmeans = segmented_kmeans.reshape(image.shape)
# DISPLAY ALL RESULTS
cv2.imshow("Original Image", image)
cv2.imshow("Thresholding Segmentation", thresh)
cv2.imshow("Region-Based Segmentation (Watershed)", watershed_image)
cv2.imshow("Learning-Based Segmentation (K-Means)", segmented_kmeans)
cv2.waitKey(0)
cv2.destroyAllWindows()
