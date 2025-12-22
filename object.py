
#Experiment 10: Object Detection  
#Implement and compare object detection using both traditional computer vision methods and 
#modern deep learning techniques


import cv2
import numpy as np

# ------------------- Read Image -------------------
image = cv2.imread("house.jpg")

if image is None:
    print("Error: Image not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =================================================
# 1. TRADITIONAL OBJECT DETECTION (HAAR CASCADE)
# =================================================

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

haar_image = image.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(haar_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# =================================================
# 2. DEEP LEARNING OBJECT DETECTION (SSD + MOBILENET)
# =================================================

# Load model files
prototxt = r"C:\Users\kavit\OneDrive\Desktop\wormi\Exp 10\MobileNetSSD_deploy.prototxt"
model = r"C:\Users\kavit\OneDrive\Desktop\wormi\Exp 10\MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)


net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Class labels
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    0.007843, (300, 300), 127.5
)

net.setInput(blob)
detections = net.forward()

dnn_image = image.copy()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = f"{CLASSES[idx]}: {confidence:.2f}"
        cv2.rectangle(dnn_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(
            dnn_image, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

# =================================================
# DISPLAY RESULTS
# =================================================

cv2.imshow("Original Image", image)
cv2.imshow("Traditional Object Detection (Haar Cascade)", haar_image)
cv2.imshow("Deep Learning Object Detection (SSD + MobileNet)", dnn_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
