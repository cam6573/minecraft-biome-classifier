import os
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import seaborn as sns
import matplotlib.pyplot as plt


training_dataset_path = "data\\preprocessed\\training"
test_dataset_path = "data\\preprocessed\\test"
validation_dataset_path = "data\\preprocessed\\validation"
labels = ['dark_forest', 'desert', 'mountains', 'plains', 'savanna', 'swamp']
image_size = (64,64)




# Load data sets, resize and extract features

def extract_hog(img):
    features , hog_image = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        channel_axis=-1,
        visualize=True
    )
    return features, hog_image


def extract_color_features(image):
    img = np.array(image)
    # RGB channels
    hist_r = np.histogram(img[:, :, 0], bins=64, range=(0, 256))[0]
    hist_g = np.histogram(img[:, :, 1], bins=64, range=(0, 256))[0]
    hist_b = np.histogram(img[:, :, 2], bins=64, range=(0, 256))[0]

    features = np.concatenate([hist_r, hist_g, hist_b])

    return features

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
                image = np.array(image)
                color_features = extract_color_features(image)
                HOG_features, _ = extract_hog(image)
                features = np.concatenate((color_features, HOG_features))
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Error processing {picture_path}: {e}")
    return np.array(X), np.array(y)




X_train, y_train = load_images_from_folder(training_dataset_path, image_size)
X_test, y_test = load_images_from_folder(test_dataset_path, image_size)
X_val, y_val = load_images_from_folder(validation_dataset_path, image_size)


X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)


scaler = StandardScaler()

# Fit on training data
X_train = scaler.fit_transform(X_train)

# Apply same transformation to validation and test
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)



print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])
print("Validation set size:", X_val.shape[0])




#train svm model
model = SVC(kernel='rbf', C=10)
# fit model
model.fit(X_train, y_train)



# model evaluation 

def plot_confusion_matrix(y_test,y_pred_test,biomes):
    matrix = confusion_matrix(y_test, y_pred_test)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]


    # Build the plot
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)


    # Add labels to the plot
    tick_marks = np.arange(len(biomes))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, biomes, rotation=25)
    plt.yticks(tick_marks2, biomes, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for SVM Model')
    plt.savefig('resources/svm_confusion_matrix.png')


#validation accuracy
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))


cm_val = confusion_matrix(y_val, y_pred, labels=labels)
cm_val_df = pd.DataFrame(cm_val, index=labels, columns=labels)
print("Validation Confusion Matrix:")
print(cm_val_df)


#test accuracy
y_pred_test = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
cm_test = confusion_matrix(y_test, y_pred_test, labels=labels)
cm_test_df = pd.DataFrame(cm_test, index=labels, columns=labels)
print("Test Confusion Matrix:")
print(cm_test_df)
plot_confusion_matrix(y_test, y_pred_test, labels)
