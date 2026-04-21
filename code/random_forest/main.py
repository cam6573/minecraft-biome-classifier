
import os
from feature_extraction import extract_features
from training import training,tuning

def main():
    ##Extract features
    if os.path.isdir('code/random_forest/model_matrices/'):
        print("Feature matrix and label vector already exists")
    else:
        extract_features.feature_label_vectors_for_datasets()

    ## Find best hyperparameters
    tuning.training_model_with_best_hyper_parameters()

    ##Train model with best hyperparameters
    training.training_process()

main()