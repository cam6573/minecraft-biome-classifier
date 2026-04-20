import os
import sys
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline   
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
        sys.path.append(current_dir)

import load_data

def plot_conf_matrix(y_true, y_pred, biomes):
        matrix = confusion_matrix(y_true, y_pred)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 6))
        sns.heatmap(matrix, annot=True, cmap='Blues', xticklabels=biomes, yticklabels=biomes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('AdaBoost Confusion Matrix')
        plt.savefig('code/adaboost/evaluation/confusion_matrix.png')

def train():
        X_train, y_train, X_val, y_val, X_test, y_test = load_data.loadData()

        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                                  n_estimators=100,
                                                  learning_rate=0.5,
                                                  random_state=42
                                        ))
        ])

        print("Training AdaBoost model...")
        pipeline.fit(X_train, y_train)
        y_pred_test = pipeline.predict(X_test)

        train_acc = pipeline.score(X_train, y_train)
        val_acc = pipeline.score(X_val, y_val)
        test_acc = pipeline.score(X_test, y_test)

        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        biomes = ["plains", "desert", "mountains", "swamp", "dark_forest", "savanna"]

        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred_test, labels=[0,1,2,3,4,5], target_names=biomes, zero_division=0))
        os.makedirs('code/adaboost/evaluation', exist_ok=True)
        plot_conf_matrix(y_test, y_pred_test, biomes)

if __name__ == "__main__":
        train()