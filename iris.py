import cv2
import dlib
import numpy as np
from collections import deque

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize variables
infraction_count = 0
max_infractions = 3
gaze_history = deque(maxlen=10)  # Store gaze direction for the last 10 frames
consistent_gaze_threshold = 8  # Number of consistent gazes to trigger an infraction

def get_eye_region(eye_points, frame):
    left = np.min(eye_points[:, 0])
    top = np.min(eye_points[:, 1])
    right = np.max(eye_points[:, 0])
    bottom = np.max(eye_points[:, 1])
    eye_region = frame[top:bottom, left:right]
    eye_region = cv2.resize(eye_region, (100, 50))
    return eye_region, (left, top, right, bottom)

def detect_iris(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def detect_gaze(iris_position, eye_region_width):
    if iris_position:
        iris_x, _ = iris_position
        if iris_x < eye_region_width * 0.3:
            return "left"
        elif iris_x > eye_region_width * 0.7:
            return "right"
    return "center"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    current_gaze = "center"

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        left_eye = shape[36:42]
        right_eye = shape[42:48]

        for eye in [left_eye, right_eye]:
            eye_region, (left, top, right, bottom) = get_eye_region(eye, frame)
            iris_position = detect_iris(eye_region)
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            if iris_position:
                iris_x, iris_y = iris_position
                iris_x = int(iris_x * (right - left) / 100) + left
                iris_y = int(iris_y * (bottom - top) / 50) + top
                cv2.circle(frame, (iris_x, iris_y), 3, (0, 0, 255), -1)

            gaze = detect_gaze(iris_position, eye_region.shape[1])
            if gaze != "center":
                current_gaze = gaze

    gaze_history.append(current_gaze)
    
    # Always display current gaze direction
    cv2.putText(frame, f"Gaze: {current_gaze.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Always display infraction count
    cv2.putText(frame, f"Infractions: {infraction_count}/{max_infractions}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Check if gaze has been consistent for the last few frames
    if len(gaze_history) == gaze_history.maxlen:
        if gaze_history.count("left") >= consistent_gaze_threshold or gaze_history.count("right") >= consistent_gaze_threshold:
            infraction_count += 1
            cv2.putText(frame, f"Alert: Looking {current_gaze.upper()} for too long!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            gaze_history.clear()  # Reset gaze history after an infraction

    if infraction_count >= max_infractions:
        cv2.putText(frame, "Exam terminated!", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Exam Monitoring", frame)
        cv2.waitKey(5000)
        break

    cv2.imshow("Exam Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()