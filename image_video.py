
#Experiment 1: Image /Video Reading and Display 
#Develop a Python program using OpenCV to demonstrate image acquisition and video acquisition 
#by reading images in multiple modes and capturing live video from a camera, displaying and storing 
#the video stream. 


import cv2

# -----------------------------------------
# 1. IMAGE READING IN MULTIPLE MODES
# -----------------------------------------
# Image path
image_path = "house.jpg"

# Read in color (BGR)
image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Read in grayscale
image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Read as is (unchanged, including alpha channel if exists)
image_unchanged = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check if image loaded
if image_color is None:
    print("Error: Image not found!")
    exit()

# Display images
cv2.imshow("Original Color Image", image_color)
cv2.imshow("Grayscale Image", image_gray)
cv2.imshow("Unchanged Image", image_unchanged)

cv2.waitKey(10000)  # Hold for some seconds
cv2.destroyAllWindows()

# -----------------------------------------
# 2. VIDEO CAPTURE FROM CAMERA
# -----------------------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera

# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define codec and create VideoWriter object to save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

print("Press 'q' to stop video capture...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Live Video', frame)

    # Write the frame into the file
    out.write(frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
