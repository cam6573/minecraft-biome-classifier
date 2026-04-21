
import os
import sys

from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import numpy as np
import json


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import load_data

param_grid = {
    'n_estimators': [100, 300],       
    'max_depth': [None, 10, 25],      
    'min_samples_leaf': [2, 10],       
    'max_features': ['sqrt'],
    'min_samples_split': [2, 5,10],   
}

def training_model_with_best_hyper_parameters():
    X_train,y_train,X_test,y_test,X_val,y_val = load_data.load_data()
    ##Preprocessing
    scaler = StandardScaler()
    
    ##Scale Features 
    training_set_scaled = scaler.fit_transform(X_train)
    test_set_scaled = scaler.transform(X_test)
    validation_set_scaled = scaler.transform(X_val)

    X_search = np.vstack((training_set_scaled, validation_set_scaled))
    y_search = np.concatenate((y_train, y_val))
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    pds = PredefinedSplit(test_fold=split_index)


    ##Start GridSearch
    model = RandomForestClassifier(random_state=45,class_weight='balanced_subsample')
    gridSearchRF = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv = pds
    )

    #Train using best parameters
    print("Running Grid Search...")
    gridSearchRF.fit(X_search, y_search)
    
    ##Print best parameters for model
    print(f"\nBest Params found with Grid Searc: {gridSearchRF.best_params_}")
    ##save best parameters to a json file
    with open('code/random_forest/training/best_params.json', "w") as f:
        json.dump(gridSearchRF.best_params_,f, indent=4)
    
    return gridSearchRF.best_params_

if __name__ == "__main__":
    training_model_with_best_hyper_parameters()