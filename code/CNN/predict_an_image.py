import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

model = tf.keras.models.load_model("minecraft_biome_cnn.keras")

class_names = ['dark_forest', 'desert', 'mountains', 'plains', 'savanna', 'swamp']

img_path = "data/processed/CNN/test/swamp/2_swamp.jpg"

img = load_img(img_path, target_size=(128, 128))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions[0])]

print("Predicted biome:", predicted_class)
print("Confidence:", np.max(predictions[0]))
print("All scores:", predictions[0])