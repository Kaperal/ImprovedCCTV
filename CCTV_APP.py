import cv2
from gun_and_human_detection import detect_objects
from face_recognition_module import recognize_faces


def main():
    cap = cv2.VideoCapture(0)  # Use your CCTV camera feed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Object detection (guns, humans)
        detections = detect_objects(frame)

        # Step 2: Check for guns or humans
        if 'gun' in detections:
            print("Gun detected! Triggering alert...")
            # Call alert system function

        if 'human' in detections:
            print("Human detected, starting face recognition...")
            faces = recognize_faces(frame)
            for face in faces:
                if face['recognized']:
                    print(f"Recognized: {face['name']}")
                else:
                    print("Unknown face detected, storing image...")
                    # Save or alert

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
