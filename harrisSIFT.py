
#Experiment 7: Corner Detection 
#Design and implement a corner detection algorithm using Harris / SIFT Techniques 


import cv2
import numpy as np

# -----------------------------------------
# READ IMAGE
# -----------------------------------------
image = cv2.imread("house.jpg")

if image is None:
    print("Error: Image not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# =================================================
# 1. HARRIS CORNER DETECTION
# =================================================
harris_response = cv2.cornerHarris(
    gray,
    blockSize=2,
    ksize=3,
    k=0.04
)

# Dilate for marking corners
harris_response = cv2.dilate(harris_response, None)

harris_result = image.copy()
harris_result[harris_response > 0.01 * harris_response.max()] = [0, 0, 255]

# =================================================
# 2. SIFT CORNER (KEYPOINT) DETECTION
# =================================================
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
)

sift_result = cv2.drawKeypoints(
    image, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# =================================================
# DISPLAY RESULTS
# =================================================
cv2.imshow("Original Image", image)
cv2.imshow("Harris Corner Detection", harris_result)
cv2.imshow("SIFT Keypoint Detection", sift_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
