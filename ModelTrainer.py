from ultralytics import YOLO
from pathlib import Path
import torch

PLATFORM = "amd"
EXPORT_FORMAT = "onnx"

BASE_PATH = Path("./datasets/food2022").resolve()
YAML_PATH = BASE_PATH / "food2022.yaml"
MODEL_NAME = "yolov8n.pt"
RUN_NAME = f"food2022_yolov8_{PLATFORM}"

EPOCHS = 10
BATCH = 32
IMGSZ = 640
WORKERS = 4

def get_device():
    if PLATFORM == "apple":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available.")
        return "mps"
    elif PLATFORM == "nvidia":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif PLATFORM == "amd":
        print("AMD GPU not supported on Windows. Using CPU.")
        return "cpu"
    else:
        return "cpu"

def train():
    assert YAML_PATH.exists(), f"YAML file not found: {YAML_PATH}"
    device = get_device()
    print(f"Platform: {PLATFORM.upper()} | Device: {device.upper()}")
    model = YOLO(MODEL_NAME)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=device,
        amp=False,
        cache=True,
        rect=True,
        name=RUN_NAME
    )
    if EXPORT_FORMAT:
        model.export(format=EXPORT_FORMAT)

if __name__ == "__main__":
    train()
