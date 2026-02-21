from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano model (fastest)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Optional: reduce resolution for better FPS
cap.set(3, 640)  # width
cap.set(4, 480)  # height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()