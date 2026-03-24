from cProfile import label
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



training_dataset_path = "data\\preprocessed\\training"
test_dataset_path = "data\\preprocessed\\test"
validation_dataset_path = "data\\preprocessed\\validation"
labels = ['dark_forest', 'desert', 'mountains', 'plains', 'savanna', 'swamp']
image_size = (64,20)


# Load training set, resize, scale ###############################

def load_images_from_folder(folder_path, image_size):
    X = []
    y = []
    for picture in os.listdir(folder_path):
        picture_path = os.path.join(folder_path, picture)
        label = picture.split("_", 1)[1]
        label = label.rsplit(".", 1)[0]
        if os.path.isfile(picture_path):
            try:
                image = Image.open(picture_path).convert('RGB')
                image = image.resize(image_size)
                X.append(np.array(image) / 255.0)
                y.append(label)
            except Exception as e:
                print(f"Error processing {picture_path}: {e}")
    return np.array(X), np.array(y)


X_train, y_train = load_images_from_folder(training_dataset_path, image_size)
X_test, y_test = load_images_from_folder(test_dataset_path, image_size)
X_val, y_val = load_images_from_folder(validation_dataset_path, image_size)
print("X shape:", X_train.shape)
print("y shape:", y_train.shape)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
print("X_flat shape:", X_train.shape)


X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.20,random_state=77,stratify=y_train)
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])


# Test with built in function
model = SVC(kernel='poly')  # tried 'rbf', 'linear' too, worse accuracy
ovr = OneVsRestClassifier(model)
# fit model
ovr.fit(X_train, y_train)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy(library):", accuracy_score(y_test, y_pred))