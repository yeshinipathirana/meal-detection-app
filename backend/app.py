from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "food_detect_v1" / "weights" / "best.pt"

if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"❌ Model not found at: {WEIGHTS}\nTrain first, then run the API."
    )

model = YOLO(str(WEIGHTS))

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Send image as form-data key = image"}), 400

    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Run YOLO prediction
    res = model.predict(img, conf=0.25, iou=0.45, verbose=False)[0]

    detections = []
    names = model.names

    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
            detections.append({
                "class_id": int(cls_id),
                "class_name": names[int(cls_id)],
                "confidence": float(c),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            })

    return jsonify({
        "count": len(detections),
        "detections": detections
    })

@app.route("/", methods=["GET"])
def home():
    return "✅ Food Detect API running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
