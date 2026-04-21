import tensorflow as tf
from CNN import get_datasets
from hyperparamter_tunning_script import evaluate_model

model = tf.keras.models.load_model("./models/CNN/minecraft_biome_cnn.keras")
_, _, class_names = get_datasets(batch_size=32)
metrics = evaluate_model(model=model, batch_size=32, class_names=class_names)

