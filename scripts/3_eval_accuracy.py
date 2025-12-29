from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data_yaml = ROOT / "dataset" / "data_fixed.yaml"
weights = ROOT / "runs" / "food_detect_v1" / "weights" / "best.pt"

model = YOLO(str(weights))

metrics = model.val(data=str(data_yaml), split="test")  # test split
# metrics.box.map = mAP50-95, metrics.box.map50 = mAP50

print("\n==== ✅ MODEL ACCURACY (TEST SET) ====")
print(f"Precision:   {metrics.box.p.mean():.4f}")
print(f"Recall:      {metrics.box.r.mean():.4f}")
print(f"mAP@0.50:    {metrics.box.map50:.4f}")
print(f"mAP@0.50:95: {metrics.box.map:.4f}")
print("====================================\n")
