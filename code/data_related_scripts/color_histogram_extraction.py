

from glob import glob
import os
import numpy as np


PREPROCCESSED_DIR_PATH = "data/preprocessed"
# COLOR_PROCESSED_DIR_IMAGES_PATH = "data/processed/color/images"
COLOR_DIR_FEATURES_PATH = "data/processed/color/COLOR_vectors"



def extract_color_features(image):
    return None

def files_for_color_histogram_biomes(biome_name,data_type):
    input_dir = os.path.join(f"{PREPROCCESSED_DIR_PATH}/{data_type}", biome_name)
    output_features_dir = os.path.join(f"{COLOR_DIR_FEATURES_PATH}/{data_type}", biome_name)

    os.makedirs(output_features_dir,exist_ok=True)

    biome = glob(os.path.join(input_dir, "*.jpg"))
    print(f"Processing {biome_name}...")
    count = 0
    
    for image_path in biome:
        
        try:
            feature_matrix = extract_color_features(image_path)

            if feature_matrix is None:
                print(f"image {image_path} could not be processed into a HOG")
            else:

                # Get's file name and File Stem
                filename = os.path.basename(image_path)
                file_stem = os.path.splitext(filename)[0]

                 # Save Vector Values
                vector_save_path = os.path.join(output_features_dir, file_stem + ".npy")
                np.save(vector_save_path, feature_matrix)
                count+=1
        except Exception as e:
            print(e)    

    

