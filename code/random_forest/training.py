from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np

# 1. Load the pre-extracted features
X_train = np.load('model_matrices/training/X_train.npy')
y_train = np.load('model_matrices/training/y_train.npy')


X_test = np.load('model_matrices/test/X_test.npy')
y_test = np.load('model_matrices/test/y_test.npy')

X_val = np.load('model_matrices/validation/X_validation.npy')
y_val = np.load('model_matrices/validation/y_validation.npy')

print("Training matrices shape")
print(X_train.shape)
print(y_train.shape)

print("Test matrices shape")

print(X_test.shape)
print(y_test.shape)


print("Validation matrices shape")
print(X_val.shape)
print(y_val.shape)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
    ))
])

# 3. Train
print("Training Random Forest...")

pipeline.fit(X_train, y_train)

train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
val_accuracy = pipeline.score(X_val, y_val)

print(f"Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
print(f"Validation accuracy: {val_accuracy:.3f}")


##3. "Confusion Matrix"
y_pred = pipeline.predict(X_val)
biomes = ["plains", "desert", "mountains", "swamp", "dark_forest", "savanna"]
print("\nDetailed Report:")
print(classification_report(y_val, y_pred, target_names=biomes))
