from ultralytics import YOLO
import torch
import numpy as np

model = YOLO("./runs/classify/train4/weights/best.pt")
results = model.predict('./weather-dataset/train/sunrise/sunrise1.jpg')

names_dict = results[0].names

probs = results[0].probs.numpy().data
print(names_dict)
print(probs)
print(names_dict[np.argmax(probs)])

