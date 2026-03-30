import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from glob import glob


class_names = ['dark_forest', 'desert', 'mountains', 'plains', 'savanna', 'swamp']

test_images_path = "data/processed/CNN/test/"

total_prediction = 0
accurate_predictions = 0

def main():
    model = tf.keras.models.load_model("minecraft_biome_cnn.keras")

    for biome in class_names:

        biome_path = os.path.join(test_images_path,biome)
        images = glob(os.path.join(biome_path, "*.jpg"))
        for img_path in images:
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            if predicted_class == biome:
                accurate_predictions+= 1
            total_prediction += 1

    print(f"Total Predictions: {total_prediction}")
    print(f"Accurate Predictions: {accurate_predictions}")
    print(f"Accuracy: {accurate_predictions/total_prediction}")


def test_given_model(model):
    class_names = ['dark_forest', 'desert', 'mountains', 'plains', 'savanna', 'swamp']

    test_images_path = "data/processed/CNN/test/"

    total_prediction = 0
    accurate_predictions = 0
    for biome in class_names:

        biome_path = os.path.join(test_images_path,biome)
        images = glob(os.path.join(biome_path, "*.jpg"))
        for img_path in images:
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions[0])]
            if predicted_class == biome:
                accurate_predictions+= 1
            total_prediction += 1

    print(f"Total Predictions: {total_prediction}")
    print(f"Accurate Predictions: {accurate_predictions}")
    print(f"Accuracy: {accurate_predictions/total_prediction}")

    return accurate_predictions, total_prediction





if __name__ == "__main__":
    main()


