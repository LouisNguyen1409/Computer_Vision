import os
from skimage.io import imread
from skimage.transform import resize

# prepare data
input_dir = './parking-dataset/'
categories = os.listdir(input_dir)

data = []
labels = []
IMG_SIZE = 15

for category in categories:
	for file in os.listdir(os.path.join(input_dir, category)):
		img_path = os.path.join(input_dir, category, file)
		img = imread(img_path)
		img = resize(img, (IMG_SIZE, IMG_SIZE))
		data.append(img.flatten())
		labels.append(category)

# train / test split

# train classifier

# test performance