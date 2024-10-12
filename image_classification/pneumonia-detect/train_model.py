import numpy as np
from ultralytics import YOLO
import torch

# train model
model = YOLO("yolov8n-cls.pt")
results = model.train(data="./pneumonia-dataset/", epochs=100, imgsz=64, device='mps')
