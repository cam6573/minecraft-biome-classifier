from skimage import io, color, exposure, img_as_ubyte
from skimage.transform import resize
from skimage.feature import hog
import os
from glob import glob
import numpy as np


PREPROCCESSED_DIR_PATH = "data/preprocessed/"
MATRIX_DIRECTORY_OUTPUT = "code/random_forest/model_matrices/"

def extract_hog(img):
    features , hog_image = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        channel_axis=-1,
        visualize=True
    )
    return features, hog_image

def extract_color_features(image=None, normalize=True):
        histograms = []
        for channel in range(3):
            histogram, _ = np.histogram(
                # Selects all pixels for a specific color channel (R, G, or B) using numpy's ellipsis notation
                # and flattens the 2D array of pixel values into a 1D array
                image[..., channel].ravel(),
                bins=256, # divides the range into equal-width bins
                range=(0, 256)
            )

            if normalize:
                histogram = histogram / histogram.sum()

            histograms.append(histogram)

        return np.concatenate(histograms)

def combine_features(img):
    image = io.imread(img)
    ##extracting hog features
    image_resized = resize(image, (256, 256))
    hog_features,_ = extract_hog(image_resized)

    ##extracting ccolor features
    image_255 = (image_resized * 255).astype(np.uint8)
    color_features = extract_color_features(image_255)

    combined_features = np.concatenate([hog_features, color_features])
    return combined_features



def create_feature_vectors(biome_list,folder:str):
    X = []
    Y =[]
    input_dir = os.path.join(PREPROCCESSED_DIR_PATH,folder)

    for i,biome_name in enumerate(biome_list):
        biome = glob(os.path.join(input_dir, f"*{biome_name}.jpg"))
        print(f"Processing {biome_name}...")
        for image_path in biome:
            try:
                # Get HOG and color features
                image_features = combine_features(image_path)
                if image_features is not None:
                    X.append(image_features)
                    Y.append(i)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return np.array(X), np.array(Y)

def feature_label_vectors_for_datasets():
    biomes = [
        "plains",
        "desert",
        "mountains",
        "swamp",
        "dark_forest",
        "savanna"
    ]
    print("Creating feature matrix and label vector for Training set...")
    X_train,y_train = create_feature_vectors(biomes,"training")
    print("Creating feature matrix and label vector for Test set...")
    X_test,y_test = create_feature_vectors(biomes,"test")
    print("Creating feature matrix and label vector for Validation set...")
    X_validation,y_validation = create_feature_vectors(biomes,"validation")

    os.makedirs(MATRIX_DIRECTORY_OUTPUT,exist_ok=True)

    for sub in ["training", "test", "validation"]:
        os.makedirs(os.path.join(MATRIX_DIRECTORY_OUTPUT, sub), exist_ok=True)

    #save matrices
    np.save(f'{MATRIX_DIRECTORY_OUTPUT}/training/X_train.npy', X_train)
    np.save(f'{MATRIX_DIRECTORY_OUTPUT}/training/y_train.npy', y_train)
    np.save(f'{MATRIX_DIRECTORY_OUTPUT}/test/X_test.npy', X_test)
    np.save(f'{MATRIX_DIRECTORY_OUTPUT}/test/y_test.npy', y_test)
    np.save(f'{MATRIX_DIRECTORY_OUTPUT}/validation/X_validation.npy', X_validation)
    np.save(f'{MATRIX_DIRECTORY_OUTPUT}/validation/y_validation.npy', y_validation)

if __name__ == "__main__":
    
    feature_label_vectors_for_datasets()

