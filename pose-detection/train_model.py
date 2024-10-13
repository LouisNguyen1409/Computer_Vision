from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

model.train(data='config.yaml', epochs=100, imgsz=640, device='mps')