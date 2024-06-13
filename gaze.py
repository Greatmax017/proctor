while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract eye regions
        left_eye = get_eye_coordinates(landmarks, left_eye_indices)
        right_eye = get_eye_coordinates(landmarks, right_eye_indices)

        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Determine if eyes are looking left, right, or center based on aspect ratio
        # This is a simplified approach and may need more complex logic for robust applications
        if left_ear < 0.2 or right_ear < 0.2:
            gaze_direction = "Looking Away"
        else:
            gaze_direction = "Looking Forward"

        cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()