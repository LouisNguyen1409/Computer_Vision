import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# prepare data
input_dir = './parking-dataset/'
categories = os.listdir(input_dir)

data = []
labels = []
IMG_SIZE = 15

for idx, category in enumerate(categories):
	for file in os.listdir(os.path.join(input_dir, category)):
		img_path = os.path.join(input_dir, category, file)
		img = imread(img_path)
		img = resize(img, (IMG_SIZE, IMG_SIZE))
		data.append(img.flatten())
		labels.append(idx)

data = np.array(data)
labels = np.array(labels)
print("Finished preparing data")

# train / test split
# test_size = 0.2 means 20% of the data will be used for testing
# stratify = labels means the distribution of labels in the train and test set will be the same
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

# train classifier
classifier = SVC()

# we train 12 models with different pairs of hyperparameters
# then we choose the best model
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

# GridSearchCV will train 12 models
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters)
grid_search.fit(x_train, y_train)

print("Finished training classifier")

# test performance

# get the best model from 12 models
best_estimator = grid_search.best_estimator_

y_pred = best_estimator.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy: {}%".format(accuracy * 100))

# save the best model
pickle.dump(best_estimator, open('parking-detect-model.p', 'wb'))