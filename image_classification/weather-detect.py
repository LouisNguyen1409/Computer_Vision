import os
import shutil
import random
import numpy as np
from ultralytics import YOLO
import torch

# prepare data
source_dir = "./weather-dataset"
train_dir = "./weather-dataset/train"
val_dir = "./weather-dataset/val"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
categories = ['cloudy', 'rain', 'shine', 'sunrise']

for category in categories:
	os.makedirs(os.path.join(train_dir, category), exist_ok=True)
	os.makedirs(os.path.join(val_dir, category), exist_ok=True)

	images = []
	for file in os.listdir(source_dir):
		if file.startswith(category):
			images.append(file)

	random.shuffle(images)
	split_point = int(len(images) * 0.2)

	for i, image in enumerate(images):
		if i < split_point:
			dst = os.path.join(val_dir, category, image)
		else:
			dst = os.path.join(train_dir, category, image)
		shutil.copy2(os.path.join(source_dir, image), dst)
		os.remove(os.path.join(source_dir, image))


# train model
model = YOLO("yolov8n-cls.pt")
results = model.train(data="./weather-dataset/", epochs=100, imgsz=64, device='mps')