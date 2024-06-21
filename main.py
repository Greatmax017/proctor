import cv2
import dlib
import numpy as np
import pyaudio
import math
import json
import subprocess
import time

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# load face detector model
def load_face_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(config['face_predictor_path'])
    return detector, predictor


# Initialize system camera in videocapture object, 0 is the default camera
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Could not open camera")
    return cap

# Initialize audio stream
def initialize_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=config['audio_channels'],
                    rate=config['audio_rate'],
                    input=True,
                    frames_per_buffer=config['audio_chunk'])
    return p, stream

# Get camera matrix
def get_camera_matrix(frame):
    size = frame.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    return camera_matrix

# detect voice
def detect_voice(stream, chunk):
    try:
        data = stream.read(chunk, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Apply a simple noise reduction technique
        noise_threshold = config['noise_threshold']
        audio_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)
        
        # Add a small constant to avoid division by zero
        rms = np.sqrt(np.mean(audio_data ** 2) + 1e-6)
        spl = 20 * math.log10(rms / (2 ** 15) + 1e-6) + 94  # Adjusted calculation
        return max(spl, 0)  # Ensure non-negative SPL
    except IOError as e:
        if e.errno == pyaudio.paInputOverflowed:
            print("Input overflowed")
        else:
            raise
    return 0

# Convert rotation vector to euler angles
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

# Get head pose
def get_head_pose(landmarks, camera_matrix, dist_coeffs):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),
        (landmarks.part(8).x, landmarks.part(8).y),
        (landmarks.part(36).x, landmarks.part(36).y),
        (landmarks.part(45).x, landmarks.part(45).y),
        (landmarks.part(48).x, landmarks.part(48).y),
        (landmarks.part(54).x, landmarks.part(54).y)
    ], dtype="double")
    
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return None
    
    return rotation_vector_to_euler_angles(rotation_vector)

# Show alert message using AppleScript
# def show_alert(message):
#     script = f'display dialog "{message}" buttons {{"OK"}} default button "OK"'
#     subprocess.run(["osascript", "-e", script])

# Show alert message using OpenCV window for cross-platform compatibility
def show_alert(message):
    """
    Display an alert message using OpenCV window.
    """
    alert_window = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(alert_window, message, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Alert", alert_window)
    cv2.waitKey(5000)  # Display for 2 seconds
    cv2.destroyWindow("Alert")

# Main function
def main():
    detector, predictor = load_face_detector()
    cap = initialize_camera()
    p, stream = initialize_audio()
    
    ret, frame = cap.read()
    if not ret:
        raise IOError("Could not grab frame")
    
    camera_matrix = get_camera_matrix(frame)
    dist_coeffs = np.zeros((4, 1))
    
    infractions = 0
    head_turn_start_time = None
    last_alert_time = 0
    alert_cooldown = 5  # Cooldown period in seconds
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            current_time = time.time()
            
            spl = detect_voice(stream, config['audio_chunk'])
            if spl > config['voice_threshold']:
                cv2.putText(frame, f"Voice detected (SPL: {spl:.2f} dB)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if current_time - last_alert_time > alert_cooldown:
                    infractions += 1
                    show_alert(f"Please keep quiet. Infraction {infractions}/3 committed")
                    last_alert_time = current_time
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            for face in faces:
                landmarks = predictor(gray, face)
                euler_angles = get_head_pose(landmarks, camera_matrix, dist_coeffs)
                
                if euler_angles is not None:
                    pitch, yaw, roll = euler_angles
                    print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")
                    
                    direction = "Facing: Center"
                    if abs(yaw) > config['head_angle_threshold']:
                        direction = "Facing: Left" if yaw < 0 else "Facing: Right"
                        if head_turn_start_time is None:
                            head_turn_start_time = current_time
                        elif current_time - head_turn_start_time > config['head_turn_duration']:
                            if current_time - last_alert_time > alert_cooldown:
                                infractions += 1
                                show_alert(f"Please face forward. Infraction {infractions}/3 committed")
                                last_alert_time = current_time
                            head_turn_start_time = None
                    else:
                        head_turn_start_time = None
                    
                    cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    # Draw head pose arrow
                    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
                    arrow_length = 100
                    arrow_end = (int(nose_tip[0] + arrow_length * np.sin(np.deg2rad(yaw))),
                                 int(nose_tip[1] - arrow_length * np.sin(np.deg2rad(pitch))))
                    cv2.arrowedLine(frame, nose_tip, arrow_end, (0, 255, 0), 2)
            
            cv2.imshow("Frame", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or infractions >= 3:
                break
    
    finally:
        cap.release()
        stream.stop_stream()
        stream.close()
        p.terminate()
        cv2.destroyAllWindows()



# Run the main function
if __name__ == "__main__":
    main()