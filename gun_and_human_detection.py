import cv2
from ultralytics import YOLO

# Load YOLOv8 model (use 'yolov8n.pt' for a lightweight model or your custom model)
model = YOLO('yolov8n.pt')  # Replace with 'your_custom_model.pt' if you have one

# Define the classes we're interested in (for the pre-trained COCO model)
# COCO class IDs: '0' is for 'person', and a custom ID should be used for 'gun' if trained
TARGET_CLASSES = {'person': 0, 'gun': 1}  # You may need to adjust the IDs if using a custom model


def detect_objects(frame):
    """
    Detects guns and humans in the given frame.

    Args:
        frame (numpy.ndarray): The input image/frame from the camera.

    Returns:
        list: A list of detected objects with their labels.
    """
    # Perform object detection
    results = model(frame)[0]

    # Extract the detections
    detected_objects = []
    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()

        # Check if the detected class is one of our target classes
        if class_id in TARGET_CLASSES.values():
            label = list(TARGET_CLASSES.keys())[list(TARGET_CLASSES.values()).index(int(class_id))]
            detected_objects.append({
                'label': label,
                'confidence': score,
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            })

            # Draw bounding box and label on the frame
            color = (0, 255, 0) if label == 'person' else (0, 0, 255)  # Green for person, Red for gun
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return detected_objects, frame


def main():
    # Open the camera feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect guns and humans
        detections, annotated_frame = detect_objects(frame)

        # Print detected objects to console
        for detection in detections:
            print(f"Detected: {detection['label']} with confidence {detection['confidence']:.2f}")

        # Display the annotated frame
        cv2.imshow('Gun and Human Detection', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
