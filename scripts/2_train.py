from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data_yaml = ROOT / "dataset" / "data_fixed.yaml"

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(data_yaml),
    epochs=80,
    imgsz=640,
    batch=8,            # CPU: keep smaller
    device="cpu",       # ✅ FIX HERE
    patience=15,
    project=str(ROOT / "runs"),
    name="food_detect_v1",
    workers=0           # Windows CPU: avoids dataloader issues
)

print("✅ Training completed")
