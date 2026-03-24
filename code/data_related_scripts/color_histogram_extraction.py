

from glob import glob
import os


PREPROCCESSED_DIR_PATH = "data/preprocessed/"
COLOR_PROCESSED_DIR_IMAGES_PATH = "data/processed/color/images"
COLOR_PROCESSED_DIR_FEATURES_PATH = "data/processed/color/COLOR_vectors"

def extract_color_features(image):
    return None


def files_for_color_histogram_biomes(biome_name):
    input_dir = os.path.join(PREPROCCESSED_DIR_PATH, biome_name)
    output_image_dir = os.path.join(COLOR_PROCESSED_DIR_IMAGES_PATH, biome_name)
    output_features_dir = os.path.join(COLOR_PROCESSED_DIR_FEATURES_PATH, biome_name)

    os.makedirs(output_image_dir,exist_ok=True)
    os.makedirs(output_features_dir,exist_ok=True)

    biome = glob(os.path.join(input_dir, "*.jpg"))
    print(f"Processing {biome_name}...")
    count = 0
    
    for image in biome:
        
        try:
            print("hello")

        except Exception as e:
            print(e)    

    

