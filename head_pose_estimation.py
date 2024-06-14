import cv2
import dlib
import numpy as np
import pyaudio
import struct
import math
import subprocess

try:
    # Load the detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Define the 3D model points of facial landmarks
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("Could not grab frame")
        exit()

    # Camera internals
    size = frame.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def get_eye_coordinates(landmarks, eye_indices):
        return np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in eye_indices])

    # Simple eye aspect ratio calculation
    def eye_aspect_ratio(eye):
        vertical_1 = np.linalg.norm(eye[1] - eye[5])
        vertical_2 = np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    # Indices for the left and right eye points based on the dlib facial landmark detector
    left_eye_indices = list(range(36, 42))
    right_eye_indices = list(range(42, 48))

    # Helper function to convert rotation vectors to Euler angles
    def rotation_vector_to_euler_angles(rotation_vector):
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0
        return np.rad2deg(np.array([x, y, z]))

    # Voice detection setup
    CHUNK = 2048  # Increase buffer size to reduce overflow
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    THRESHOLD = 50  # Adjust the threshold as needed

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    def detect_voice():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate the root mean square (RMS) of the audio data
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms > 0:
                # Calculate the sound pressure level (SPL) in dB relative to 20 μPa
                spl = 20 * math.log10(rms / (2 ** 15)) + 20 * math.log10(20e-6 / 2e-5)
                return spl
            else:
                return 0
        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                print("Input overflowed")
            else:
                raise
        return 0

    def system_alert(message):
        script = f'display alert "{message}"'
        subprocess.run(["osascript", "-e", script])

    infractions = 0
    blanked = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        spl = detect_voice()
        if spl > 50:  # Adjust this threshold as needed
            cv2.putText(frame, f"Voice audible to person beside (SPL: {spl:.2f} dB)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            infractions += 1
            system_alert(f"Please keep quiet. {infractions}/4 committed")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            image_points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),    # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
            ], dtype="double")

            # Solve the PnP problem
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            if not success:
                print("Could not solve PnP problem")
                continue

            # Get Euler angles
            euler_angles = rotation_vector_to_euler_angles(rotation_vector)

            # Determine the direction based on Euler angles
            direction = "Facing: Center"
            if euler_angles[1] < -10:
                direction = "Facing: Left"
            elif euler_angles[1] > 10:
                direction = "Facing: Right"
            if euler_angles[0] > 10:
                direction = "Facing: Up"

            # Gaze tracking based on eye aspect ratio (simplified approach)
            left_eye = get_eye_coordinates(landmarks, left_eye_indices)
            right_eye = get_eye_coordinates(landmarks, right_eye_indices)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            gaze_direction = "Looking: Forward"
            if left_ear < 0.2 and right_ear < 0.2:
                gaze_direction = "Eyes Closed"
            elif left_ear < 0.25:
                gaze_direction = "Looking: Left"
            elif right_ear < 0.25:
                gaze_direction = "Looking: Right"

            # Trigger system alert if looking left or right
            if (direction == "Facing: Left" or direction == "Facing: Right") and not blanked:
                blanked = True
                infractions += 1
                system_alert(f"Please keep facing forward. {infractions}/4 committed")
                blanked = False

            # Project a 3D point to the image plane to draw a line
            nose_end_point3D = np.array([(0.0, 0.0, 1000.0)])
            nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Convert image points from numpy array to integer tuples
            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                        # Draw the line from the nose tip to the projected point
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Display the direction and gaze direction on the frame
            cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, gaze_direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Voice detection
            if spl > 50:  # Adjust this threshold as needed
                cv2.putText(frame, "Talking Detected", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                infractions += 1
                system_alert(f"Please keep quiet. {infractions}/4 committed")

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    cap.release()
    stream.stop_stream()
    stream.close()
    p.terminate()
    cv2.destroyAllWindows()