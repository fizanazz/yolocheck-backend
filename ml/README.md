# ML Model Weights

Place your trained YOLOv11 weights file here:

  ml/best.pt

## How to get best.pt

After training with ultralytics:
  runs/detect/train/weights/best.pt  →  copy to  ml/best.pt

Or set a different path in your .env:
  MODEL_PATH=path/to/your/best.pt

If best.pt is missing, the backend runs in STUB MODE:
  - Returns 2 fake detections per scan
  - Useful for testing the full flow without a trained model
  - Logs a warning: "Model file not found. YOLO inference will use stub mode."
