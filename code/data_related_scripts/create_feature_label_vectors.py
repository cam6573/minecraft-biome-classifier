
from glob import glob
import os

import numpy as np


HOG_PROCESSED_DIR_FEATURES_PATH = "data/processed/HOG/HOG_vectors"

def compile_dataset(biomes):
    X = []
    Y = []

    for i in range(len(biomes)):
        biome_name = biomes[i]

        input_dir = os.path.join(HOG_PROCESSED_DIR_FEATURES_PATH, biome_name)

        HOG_vectors_file = glob(os.path.join(input_dir, "*.npy"))

        print(f"Loading {len(HOG_vectors_file)} samples from {biome_name}...")
        for v in HOG_vectors_file:
            print("hello")
            vector = np.load(v)
            X.append(vector)
            Y.append(i)
        
    return np.array(X), np.array(Y)
    

def main():
    biomes = [
        "plains",
        "desert",
        "mountains",
        "swamp",
        "dark_forest",
        "savanna"
    ]

    X,Y =compile_dataset(biomes)
    print(X.shape)
    print(Y.shape)

if __name__ == "__main__":
    main()