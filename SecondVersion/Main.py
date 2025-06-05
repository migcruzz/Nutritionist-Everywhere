import threading
from difflib import get_close_matches

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

model_path = ("TrainedModel/best.pt")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLO(model_path).to(device)

food_df = pd.read_csv("FruitsDatasetDetails/food.csv")
food_nutrient_df = pd.read_csv("FruitsDatasetDetails/food_nutrient.csv")
nutrient_df = pd.read_csv("FruitsDatasetDetails/nutrient.csv")
descriptions = food_df["description"].dropna().str.lower().unique().tolist()


def get_best_match(label):
    label_lower = label.lower()
    matches = get_close_matches(label_lower, descriptions, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    fallback = [desc for desc in descriptions if label_lower in desc]
    return fallback[0] if fallback else None


def get_nutrition_lines(label):
    match = get_best_match(label)
    if not match:
        return []
    filtered = food_df[food_df["description"].str.lower() == match.lower()]
    if filtered.empty:
        return []
    match_row = filtered.iloc[0]
    fdc_id = match_row["fdc_id"]
    nutrients = food_nutrient_df[food_nutrient_df["fdc_id"] == fdc_id]
    nutrients = nutrients.merge(nutrient_df, left_on="nutrient_id", right_on="id")
    lines = [f"{match_row['description']}"]
    for _, row in nutrients.iterrows():
        name = row["name"]
        amount = row["amount"]
        unit = row["unit_name"]
        if pd.notna(amount):
            lines.append(f"{name}: {amount} {unit}")
    return lines[:6]


ip_camera_url = "http://192.168.1.66:4747/video"
cap = cv2.VideoCapture(ip_camera_url)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

latest_frame = None
lock = threading.Lock()
running = True


def is_inside_roi(box, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    roi_x_min = 0
    roi_x_max = int(frame_width * 0.5)
    roi_y_min = int(frame_height * 0.5)
    roi_y_max = frame_height
    x1, y1, x2, y2 = box
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2
    return roi_x_min <= box_center_x <= roi_x_max and roi_y_min <= box_center_y <= roi_y_max


def inference_thread():
    global latest_frame
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_height, frame_width = frame.shape[:2]

        roi_x_min = 0
        roi_x_max = int(frame_width * 0.5)
        roi_y_min = int(frame_height * 0.5)
        roi_y_max = frame_height

        roi = frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max]

        results = model(roi, verbose=False)
        boxes = results[0].boxes
        annotated = frame.copy()

        if boxes is not None and boxes.cls.numel() > 0:
            for i in range(len(boxes.cls)):
                cls_idx = int(boxes.cls[i])
                label = results[0].names[cls_idx]
                xyxy = boxes.xyxy[i].int().tolist()

                x1, y1, x2, y2 = xyxy
                x1 += roi_x_min
                x2 += roi_x_min
                y1 += roi_y_min
                y2 += roi_y_min

                lines = get_nutrition_lines(label)
                for j, line in enumerate(lines):
                    y_offset = y1 + 25 + j * 22
                    cv2.putText(
                        annotated, line, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
                    )

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
                )

        cv2.rectangle(
            annotated,
            (roi_x_min, roi_y_min),
            (roi_x_max, roi_y_max),
            (0, 255, 255),
            2
        )

        with lock:
            latest_frame = annotated


threading.Thread(target=inference_thread, daemon=True).start()

while True:
    with lock:
        display_frame = latest_frame if latest_frame is not None else None
    if display_frame is not None:
        cv2.imshow("YOLO Nutrition Overlay", display_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
