from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import numpy as np
import load_data
import json


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
    train_accuracy = pipeline.score(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    val_accuracy = pipeline.score(X_val, y_val)

    print(f"Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}")
    print(f"Validation accuracy: {val_accuracy:.3f}")
   
    y_pred = pipeline.predict(X_val)
    biomes = ["plains", "desert", "mountains", "swamp", "dark_forest", "savanna"]
    print("\nDetailed Report:")
    print(classification_report(y_val, y_pred, target_names=biomes))

training_process()