from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
import random
import cv2
from sklearn.model_selection import train_test_split
import create_feature_label_vectors as compact_dataset

##Biomes List

biomes = [
        "plains",
        "desert",
        "mountains",
        "swamp",
        "dark_forest",
        "savanna"
    ]

##Create feature and label matrices for training and test sets
X_train, y_train = compact_dataset.create_compact_matrix_for_all_features(biomes, "train")
X_test, y_test = compact_dataset.create_compact_matrix_for_all_features(biomes, "test")


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 3. Evaluate
y_pred = rf.predict(X_test)


## Fit the Model
