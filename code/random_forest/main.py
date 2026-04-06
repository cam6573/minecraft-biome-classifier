
from feature_extraction import extract_features
from training import load_data,training,tuning
def main():
    ##Extract features
    extract_features.feature_label_vectors_for_datasets()
    ##Load Data
    load_data.load_data()
    ## Find best hyperparameters
    tuning.training_model_with_best_hyper_parameters()
    
    ##Train model with best hyperparameters
    training.training_process()
    pass