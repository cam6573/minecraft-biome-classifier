from CNN import train_CNN
from test_all_image import test_given_model
import matplotlib.pyplot as plt


# WARNING: THIS WILL TAKE A SOLID 30-60 MINUTES TO RUN


node_configurations = [
    [16, 64, 64, 64],
    [32, 64, 128, 128],
    [32, 64, 128, 256],
    [64, 128, 256, 128]
]

epoch_configuration = [
    5,
    10,
    15,
    20
]

batch_size_configuation = [
    16,
    32,
    64
]

def find_best_node_configuration():
    labels = []
    val_accuracies = []
    best_accuracy = 0.0
    best_conf = []

    for node_conf in node_configurations:
        model, history = train_CNN(nodes=node_conf)

        val_accuracy = max(history.history["val_accuracy"])

        val_accuracies.append(val_accuracy)
        labels.append(str(node_conf))

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            best_conf = node_conf


    plt.figure(figsize=(10, 5))
    plt.plot(labels, val_accuracies, marker='o')
    plt.xlabel("Node Configuration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for Different Node Configurations")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("resources/CNN/node_config_plot.png")
    plt.show()

    return best_conf, best_accuracy


def find_best_epoch(best_conf: list):
    labels = []
    val_accuracies = []
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in epoch_configuration:
        model, history = train_CNN(
            nodes=best_conf,
            epochs=epoch
        )
      
        val_accuracy = max(history.history["val_accuracy"])

        val_accuracies.append(val_accuracy)
        labels.append(str(epoch))

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch

    plt.figure(figsize=(10, 5))
    plt.plot(labels, val_accuracies, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for Different Epoch Configurations")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("resources/CNN/epoch_config_plot.png")
    plt.show()

    return best_epoch, best_accuracy

def find_best_batch_size(best_conf: list, best_epoch: int):
    labels = []
    val_accuracies = []
    best_accuracy = 0.0
    best_batch = 0

    for batch_size in batch_size_configuation:
        model, history = train_CNN(
            nodes=best_conf,
            epochs=best_epoch,
            batch_size=batch_size
        )
        
        val_accuracy = max(history.history["val_accuracy"])

        val_accuracies.append(val_accuracy)
        labels.append(str(batch_size))

        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            best_batch = batch_size

    plt.figure(figsize=(10, 5))
    plt.plot(labels, val_accuracies, marker='o')
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for Different Batch Size")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("resources/CNN/batch_size_config_plot.png")
    plt.show()

    return best_batch, best_accuracy

def plot_history(history):
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, marker='o', label="Training Accuracy")
    plt.plot(epochs, val_acc, marker='o', label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("resources/CNN/training_validation_accuracy.png")

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, marker='o', label="Training Loss")
    plt.plot(epochs, val_loss, marker='o', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("resources/CNN/training_validation_loss.png")
    plt.show()


def main():
    best_conf, best_conf_accuracy = find_best_node_configuration()
    print("Best node configuration:", best_conf)
    print("Accuracy:", best_conf_accuracy)

    best_epoch, best_epoch_accuracy = find_best_epoch(best_conf)
    print("Best epoch:", best_epoch)
    print("Accuracy:", best_epoch_accuracy)

    best_batch, best_batch_accuracy = find_best_batch_size(best_conf=best_conf, best_epoch=best_epoch)
    print("Best batch size:", best_batch)
    print("Accuracy:", best_batch_accuracy)

    final_model, final_history = train_CNN(
        save=True,
        nodes=best_conf,
        epochs=best_epoch,
        batch_size=best_batch
    )

    accurate_predictions, total_predictions = test_given_model(model=final_model)
    test_accuracy = accurate_predictions / total_predictions
    print(f"Final test accuracy: {test_accuracy}")

    plot_history(history=final_history)


if __name__ == "__main__":
    main()


# Best node configuration: [64, 128, 256, 128]
# Best Epoch configuration: 20
# Best Batch Size


