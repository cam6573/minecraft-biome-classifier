
from glob import glob
import os

import numpy as np

HOG_PROCESSED_DIR_FEATURES_PATH = "data/processed/HOG/HOG_vectors"
COLOR_PROCESSED_DIR_FEATURES_PATH = "data/processed/HOG/COLOR_vectors"


def create_vectors(biomes,feature_type_path,data_type):
    X = []
    Y = []


    for i in range(len(biomes)):
        biome_name = biomes[i]

        input_dir = os.path.join(f"{feature_type_path}/{data_type}", biome_name)
        HOG_vectors_file = glob(os.path.join(input_dir, "*.npy"))
        HOG_vectors_file.sort()

        for v in HOG_vectors_file:
            vector = np.load(v)
            X.append(vector)
            Y.append(i)

    return np.array(X), np.array(Y)
    

def create_compact_matrix_for_all_features(biomes,data_type):
    X_HOG,Y_HOG = create_vectors(biomes,HOG_PROCESSED_DIR_FEATURES_PATH,data_type)
    X_COLOR,Y_color = create_vectors(biomes,COLOR_PROCESSED_DIR_FEATURES_PATH,data_type)
    if X_HOG.shape[0] != X_COLOR.shape[0]:
        raise ValueError(f"Sample mismatch! HOG has {X_HOG.shape[0]} but Color has {X_COLOR.shape[0]}. Check your folders!")
    
    X_combined = np.hstack((X_HOG, X_COLOR))
    
    return X_combined, Y_HOG

    