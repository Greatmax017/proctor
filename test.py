import cv2

# Attempt to open the default camera
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
else:
    print("Camera is available")
    cap.release()