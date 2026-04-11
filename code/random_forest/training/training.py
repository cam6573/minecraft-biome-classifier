import os
import sys

from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import load_data
  

def plot_accuracy_comparison(train_acc,test_acc,val_acc):

    data = [
    ["Training", train_acc],
    ["Validation",val_acc],
    ["Test",test_acc]
    ]

    headers = ["Dataset", "Accuracy"]

    # 3. Initialize a plot
    plt.title('Accuracy comparison across datasets')
    fig, ax = plt.subplots()

    ax.axis('off')

    ax.table(cellText=data, colLabels=headers, loc='center')

    plt.savefig('code/random_forest/evaluation/accuracy_table.png')



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
    plt.title('Confusion Matrix for Random Forest Model')
    plt.savefig('code/random_forest/evaluation/confusion_matrix.png')

def training_process():
    X_train,y_train,X_test,y_test,X_val,y_val = load_data.load_data()
    
    with open('code/random_forest/training/best_params.json','r') as j:
        best_parameters = json.load(j)
    

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=best_parameters['n_estimators'],
            max_depth=best_parameters['max_depth'],
            min_samples_split=best_parameters['min_samples_split'],
            min_samples_leaf=best_parameters['min_samples_leaf'],
            max_features=best_parameters['max_features'],
            class_weight='balanced_subsample',
        ))
    ])

    # 3. Train
    print("Training Random Forest...")
    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)
    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    val_accuracy = pipeline.score(X_val, y_val)

    plot_accuracy_comparison(train_accuracy,test_accuracy,val_accuracy)

    biomes = ["plains", "desert", "mountains", "swamp", "dark_forest", "savanna"]
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=biomes))
    plot_confusion_matrix(y_test,y_pred_test,biomes)



if __name__ == "__main__":
    training_process()