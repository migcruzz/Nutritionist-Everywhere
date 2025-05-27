

from ultralytics import YOLO
import pandas as pd
import os

DATA_PATH = "/kaggle/input/food-dataset-yolo/data.yaml"

model = YOLO("yolov8n.pt")

results = model.train(
    data=DATA_PATH,
    epochs=300,
    imgsz=640,
    batch=50,
    save_period=50,
    name="food_yolo_v8_t4x2",
    project="/kaggle/working",
    device=[0, 1]
)

run_dir = "/kaggle/working/food_yolo_v8_t4x2"
metrics_path = os.path.join(run_dir, "results.csv")
report_path = os.path.join(run_dir, "training_report.xlsx")

if os.path.exists(metrics_path):
    pd.read_csv(metrics_path).to_excel(report_path, index=False)
    print("Report saved to:", report_path)

weights_path = os.path.join(run_dir, "weights", "best.pt")
print("Best weights saved to:", weights_path)
