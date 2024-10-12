import os
import shutil
import random
import numpy as np
from ultralytics import YOLO
import torch

source_dir = "./ship-dataset"
train_dir = "./ship-dataset/train"
val_dir = "./ship-dataset/val"
test_dir = "./ship-dataset/test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

categories = ['no_ship', 'ship']

for category in categories:
	os.makedirs(os.path.join(train_dir, category), exist_ok=True)
	os.makedirs(os.path.join(val_dir, category), exist_ok=True)
	os.makedirs(os.path.join(test_dir, category), exist_ok=True)

	images = []
	for file in os.listdir(source_dir):
		if category == 'no_ship' and file.startswith('0__'):
			images.append(file)
		elif category == 'ship' and file.startswith('1__'):
			images.append(file)

	random.shuffle(images)
	val_split = int(len(images) * 0.2)
	test_split = int(len(images) * 0.1)

	for i, image in enumerate(images):
		if i < val_split:
			dst = os.path.join(val_dir, category, image)
		elif i < val_split + test_split:
			dst = os.path.join(test_dir, category, image)
		else:
			dst = os.path.join(train_dir, category, image)
		shutil.copy2(os.path.join(source_dir, image), dst)
		os.remove(os.path.join(source_dir, image))

# train model
model = YOLO("yolov8n-cls.pt")
results = model.train(data="./ship-dataset/", epochs=100, imgsz=64, device='mps')
