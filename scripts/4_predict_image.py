from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
weights = ROOT / "runs" / "food_detect_v1" / "weights" / "best.pt"

model = YOLO(str(weights))

# put a test image path here
img_path = str((ROOT / "dataset" / "test" / "images").glob("*").__iter__().__next__())

results = model.predict(source=img_path, conf=0.25, iou=0.45, save=True)
print("✅ Saved prediction image into runs/predict/")
