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



dataset_path = "data\\preprocessed"
labels = os.listdir(dataset_path)
print("Labels:", labels)
image_size = (64,20)
X = []
y = []

# Testing #################################
#single image
#image_path = r"C:\Users\zangr\OneDrive\Desktop\MinecraftBiomes\minecraft-biome-classifier\data\dark_forest\biome_29_0.jpg"
#image_size = (16,5) 
#could do (64,64), but would not preserve original image dimensions

'''image = Image.open(image_path).convert('RGB')
image = image.resize(image_size)
image.show()

image = Image.open(image_path).convert('P')
image = image.resize(image_size)
image.show()'''

#image = image.resize(image_size)
#image = image.crop((0, 20, 128, 40)) #(left, top, right, bottom)
#print("Image shape:", image.size)

#image.show()

#labels = ['swamp', 'plains']
# Load dataset, resize, scale ###############################

# Loop through each label and process all the images
for label in labels:
    i = 0
    label_path = os.path.join(dataset_path, label)
    for picture in os.listdir(label_path):
        picture_path = os.path.join(label_path, picture)
        if os.path.isfile(picture_path):
            i=i+1
            if i >= 600: 
                break
            try:
                image = Image.open(picture_path).convert('RGB')
                image = image.resize(image_size)
                X.append(np.array(image) / 255.0)
                y.append(label)
            except Exception as e:
                print(f"Error processing {picture_path}: {e}")

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_flat = X.reshape(X.shape[0], -1)
print("X_flat shape:", X_flat.shape)


X_train,X_test,y_train,y_test=train_test_split(X_flat,y,test_size=0.20,random_state=77,stratify=y)
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

'''
class SVM:
    """
    A simple linear Support Vector Machine (SVM) implemented from scratch using NumPy.
    Uses gradient descent to optimize the hinge loss with L2 regularization.
    """

    def __init__(self, learning_rate=0.001, lambda_rate=0.01, n_iter=1000, multi_class=False):
        self.learning_rate = learning_rate
        self.lambda_rate = lambda_rate
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.loss_history = []
        self.multi_class = multi_class
        self.classes = None

    def compute_loss(self, X, y):
        """
        Compute hinge loss with L2 regularization.
        """
        distances = 1 - y * (np.dot(X, self.w) + self.b)
        distances = np.maximum(0, distances)  # hinge loss part
        hinge_loss = np.mean(distances)
        reg_loss = self.lambda_rate * (np.dot(self.w, self.w))
        return reg_loss + hinge_loss

    def fit(self, X, y):
        if self.multi_class:
            # Implement one-vs-rest strategy for multi-class classification
            self.classes = np.unique(y)
            print("Classes found:", self.classes)
            self.w = np.zeros((len(self.classes), X.shape[1]))
            self.b = np.zeros(len(self.classes))

            for cls in enumerate(self.classes):
                print(f"Training binary classifier for class: {cls}")
                y_binary = np.where(y == cls, 1, -1)
                self.binary_fit(X, y_binary) #idx?
        else:
            self.binary_fit(X, y)

    def binary_fit(self, X, y):
        """
        Train the SVM model using gradient descent.
        """
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_rate * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_rate * self.w - (y[idx] * x_i)
                    db = -y[idx]

                # Parameter updates
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            # Record loss
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)

            # Print progress every 10 iterations
            if i > 0 and (i + 1) % 10 == 0:
                print(f'iteration: {i + 1}, Loss: {loss:.4f}')

            # Early stopping on convergence
            if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < 1e-6:
                print(f'Converged at iteration: {i + 1} with loss {loss:.4f}')
                break

    def predict(self, X):
        """
        Predict class labels for input data.
        """
        pred = np.sign(np.dot(X, self.w) + self.b)
        return np.sign(pred)

svm = SVM(learning_rate=0.001, lambda_rate=0.05, n_iter=20, multi_class=True)
svm.fit(X_train, y_train)
'''

'''y_pred = svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)


# Create a DataFrame for visualization
loss_df = pd.DataFrame({
    'Iteration': range(1, len(svm.loss_history) + 1),
    'Loss': svm.loss_history
})

sns.lineplot(data=loss_df, x='Iteration', y='Loss', marker='o', color='royalblue')
plt.title("SVM Training Loss Convergence", fontsize=14, fontweight='bold')
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.tight_layout()
plt.show()




'''

# Test with built in function
model = SVC(kernel='poly')  # tried 'rbf', 'linear' too, worse accuracy
ovr = OneVsRestClassifier(model)
# fit model
ovr.fit(X_train, y_train)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy(library):", accuracy_score(y_test, y_pred))