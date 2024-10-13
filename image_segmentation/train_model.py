from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n-seg.pt')

# Train the model using MPS
results = model.train(data="config.yaml", epochs=100, imgsz=640, device='mps', workers=4)
