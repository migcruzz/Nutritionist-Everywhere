from pathlib import Path

from ultralytics import YOLO

PLATFORM = "mac"
BASE_PATH = Path("./datasets/food2022").resolve()
YAML_PATH = BASE_PATH / "food2022.yaml"
MODEL_NAME = "yolov8n.pt"
RUN_NAME = f"food2022_yolov8_{PLATFORM}"

EPOCHS = 10
BATCH = 8
IMGSZ = 640
WORKERS = 3
DEVICE = "mps"


def train():
    assert YAML_PATH.exists(), f"YAML file not found: {YAML_PATH}"
    print(f"Platform: {PLATFORM.upper()} | Device: {DEVICE.upper()}")
    model = YOLO(MODEL_NAME)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        amp=False,
        cache=True,
        rect=True,
        name=RUN_NAME,
        verbose=True
    )


if __name__ == "__main__":
    train()
