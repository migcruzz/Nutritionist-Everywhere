from ultralytics import YOLO
import cv2
from pathlib import Path

model_path = Path("/yolo8n.pt")
model = YOLO(str(model_path))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Fruits360", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
