from pathlib import Path
from ultralytics import YOLO

base_path = Path("./datasets/food2022").resolve()
yaml_path = base_path / "food2022.yaml"

assert yaml_path.exists(), f"YAML file not found: {yaml_path}"

model = YOLO("yolov8n.pt")

model.train(
    data=str(yaml_path),
    epochs=5,
    imgsz=640,
    batch=32,
    workers=8,
    device="mps",
    amp=False,
    cache=True,
    rect=True,
    name="food2022_yolov8_mps"
)
