from skimage import io, color, exposure, img_as_ubyte
from skimage.feature import hog
import os
from glob import glob
import numpy as np


PREPROCCESSED_DIR_PATH = "data/preprocessed/"
HOG_PROCESSED_DIR_IMAGES_PATH = "data/processed/HOG/images"
HOG_PROCESSED_DIR_FEATURES_PATH = "data/processed/HOG/HOG_vectors"

def get_HOG(img):
    image = io.imread(img)

    gray_image = color.rgb2gray(image)

    features , hog_image = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )

    return features, hog_image


def make_HOGs_for_biomes(biome_name:str):

    input_dir = os.path.join(PREPROCCESSED_DIR_PATH, biome_name)
    output_image_dir = os.path.join(HOG_PROCESSED_DIR_IMAGES_PATH, biome_name)
    output_features_dir = os.path.join(HOG_PROCESSED_DIR_FEATURES_PATH, biome_name)

    os.makedirs(output_image_dir,exist_ok=True)
    os.makedirs(output_features_dir,exist_ok=True)

    biome = glob(os.path.join(input_dir, "*.jpg"))
    print(f"Processing {biome_name}...")
    count = 0
    for image_path in biome:
        try:
            # Get HOG
            features, HOG_image = get_HOG(image_path)
            
            if HOG_image is None:
                print(f"image {image_path} could not be processed into a HOG")
            else:
                # Turns HOG image 0-1 to 8-bit scale 0-255
                hog_image_rescaled = exposure.rescale_intensity(HOG_image, in_range='image')
                hog_image_uint8 = img_as_ubyte(hog_image_rescaled)

                # Get's file name and File Stem
                filename = os.path.basename(image_path)
                file_stem = os.path.splitext(filename)[0]

                # Saves Image
                save_path = os.path.join(output_image_dir, filename)
                io.imsave(save_path, hog_image_uint8)


                # Save Vector Values
                vector_save_path = os.path.join(output_features_dir, file_stem + ".npy")
                np.save(vector_save_path, features)
                count+=1
                

        except Exception as e:
            print(f"error processing {image_path}: {e}")
    print(f"{biome_name} Completed: {count}/{len(biome)}\n")

def main():
    biomes = [
        "plains",
        "desert",
        "mountains",
        "swamp",
        "dark_forest",
        "savanna"
    ]

    for biome_name in biomes:
        make_HOGs_for_biomes(biome_name)

if __name__ == "__main__":
    main()


