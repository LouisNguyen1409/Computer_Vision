from ultralytics import YOLO
import torch
import numpy as np

model = YOLO("./runs/classify/train/weights/last.pt")
results = model.predict('./ship-dataset/test/ship/1__20170613_180813_1017__-122.32435112918907_37.72793175152951.png')

names_dict = results[0].names

probs = results[0].probs.numpy().data
print(names_dict)
print(probs)
print(names_dict[np.argmax(probs)])

