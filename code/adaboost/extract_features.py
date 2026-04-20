import os
import numpy as np
from glob import glob
from skimage import io, color
from skimage.feature import hog

IN_PATH = "data/preprocessed"
OUT_PATH = "code/adaboost/data"

BIOMES = [
        "plains",
        "desert",
        "mountains",
        "swamp",
        "dark_forest",
        "savanna"
]

def extract_hog_features(image):
        gray = color.rgb2gray(image)
        features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm="L2-Hys")
        return features

def extract_color_features(image):
        hist_features = []
        for i in range(3):
                hist, _ = np.histogram(image[:, :, i].ravel(),
                                       bins=256, range=(0,256), density=True) 
                hist_features.append(hist)
        return np.concatenate(hist_features)

def extract_features(image):
        image = io.imread(image)
        hog_features = extract_hog_features(image)
        color_features = extract_color_features(image)
        return np.concatenate([hog_features, color_features])

def process_split(split):
        X, y = [], []
        input_dir = os.path.join(IN_PATH, split)
        for label, biome in enumerate(BIOMES):
                image_paths = glob(os.path.join(input_dir, f"*{biome}.jpg"))
                print(f"{split} -> {biome}: {len(image_paths)} images")

                for img_path in image_paths:
                        try:
                                features = extract_features(img_path)
                                X.append(features)
                                y.append(label)
                        except:
                                print(f"Error processing {img_path}")

        X = np.array(X)
        y = np.array(y)

        out_dir = os.path.join(OUT_PATH, split)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X.npy"), X)
        np.save(os.path.join(out_dir, "y.npy"), y)
        print(f"\nSaved {split} dataset: X={X.shape}, y={y.shape}\n")
# def build_dataset(split):
#         X, y = [], []
#         input_dir = os.path.join(IN_PATH, split)
#         for label, biome in enumerate(BIOMES):

#                 image_paths = glob(os.path.join(input_dir, f"*{biome}.jpg"))
#                 print(f"{split} -> {biome}: {len(image_paths)} images")

#                 for img_path in image_paths:
#                         try:
#                                 features = extract_features(img_path)
#                                 X.append(features)
#                                 y.append(label)
#                         except:
#                                 print(f"Error processing {img_path}")
#         return np.array(X), np.array(y)

def save_dataset():
        for split in ["training", "validation", "test"]:
                process_split(split)
        

if __name__ == "__main__":
        save_dataset()