import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import keras_tuner as kt

IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 6

TRAIN_DIR = "data/processed/CNN/training"
VALIDATION_DIR = "data/processed/CNN/validation"


def get_datasets(batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size
    )

    print(f"Classes: {train_ds.class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def create_model(nodes, dropout_rate):
    model = keras.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Rescaling(1. / 255),

        layers.RandomFlip("horizontal"), 

        layers.Conv2D(nodes[0], (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(nodes[1], (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(nodes[2], (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(nodes[3], activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_model(hp):
    nodes = [
        hp.Choice("conv1_nodes", values=[32, 64, 128]),
        hp.Choice("conv2_nodes", values=[64, 128, 256]),
        hp.Choice("conv3_nodes", values=[128, 256, 512]),
        hp.Choice("dense_nodes", values=[64, 128, 256, 512]),
    ]

    dropout_rate = hp.Choice("dropout_rate", values=[0.1, 0.2, 0.3, 0.4, 0.5])

    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model = create_model(nodes, dropout_rate)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def tune_best_hyperparameters(batch_size=32, epochs=15, max_trials=20):
    train_ds, val_ds = get_datasets(batch_size=batch_size)

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        min_delta=0.001,
    )

    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=1,
        directory="code/CNN/keras_tuner_dir",
        project_name="minecraft_biome_nodes",
        overwrite=True
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    best_nodes = [
        best_hp.get("conv1_nodes"),
        best_hp.get("conv2_nodes"),
        best_hp.get("conv3_nodes"),
        best_hp.get("dense_nodes"),
    ]
    best_dropout = best_hp.get("dropout_rate")

    print("Best hyperparameters found:")
    print(f"nodes: {best_nodes}")
    print(f"dropout_rate: {best_dropout}")

    return best_nodes, best_dropout


def train_CNN(
    save: bool = False,
    batch_size: int = 32,
    epochs: int = 40,
    nodes: list = None,
    dropout_rate: float = 0.2
):
    if nodes is None:
        nodes = [64, 128, 256, 128]

    train_ds, val_ds = get_datasets(batch_size=batch_size)

    model = create_model(nodes, dropout_rate)
    model.summary()

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        min_delta=0.001,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    best_val_accuracy = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    if save:
        model.save("./models/CNN/minecraft_biome_cnn.keras")

    return model, history


def main():
    best_nodes, best_dropout = tune_best_hyperparameters(
        batch_size=32,
        epochs=10,
        max_trials=5
    )

    print(f"Best Nodes: {best_nodes}")
    print(f"Best Dropout: {best_dropout}")

    model, history = train_CNN(
        save=True,
        batch_size=32,
        epochs=20,
        nodes=best_nodes,
        dropout_rate=best_dropout
    )


if __name__ == "__main__":
    main()