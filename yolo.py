import supergradients as sg
import cv2

#Retrieve the model
model = sg.models.get("yolo_nas_l", pretrained_weights="coco");
# Read video
cap = cv2.VideoCapture("in.avi")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect humans in the frame
    detections = sg.detect_objects(frame, model)

    # Process detection results
    for detection in detections:
        if detection["class"] == "person":  # Check if the detected object is a person
            x, y, w, h = detection["bbox"]
            confidence = detection["confidence"]

            # Draw bounding box around the detected person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

    # Display the output frame
    cv2.imshow("Human Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
