import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 6


TRAIN_DIR = "data/processed/CNN/training"
VALIDATION_DIR = "data/processed/CNN/validation"


def train_CNN(
        save: bool = False,
        batch_size : int = 32,
        epochs : int =10,
        nodes : list= [32, 64, 128, 128]     
        ):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size = (IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        layers.Conv2D(nodes[0], (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(nodes[1], (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(nodes[2], (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(nodes[3], activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Show model structure
    model.summary()

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    if save:
        model.save("./models/CNN/minecraft_biome_cnn.keras")
    
    return model


def main():
    train_CNN(True)


if __name__ == "__main__":
    main()