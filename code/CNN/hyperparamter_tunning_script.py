from CNN import train_CNN, tune_best_hyperparameters
from test_all_image import test_given_model
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no display needed
import matplotlib.pyplot as plt

# Total Predictions: 539 
# Accurate Predictions: 505 
# Accuracy: 0.9369202226345084 
# Best nodes: [32, 64, 128, 128] 
# Best dropout: 0.2 
# Best batch size: 16 
# Validation Accuracy: 0.9222221970558167 
# Final test accuracy: 0.9369202226345084

batch_size_configuation = [16, 32, 64, 128]


def find_best_batch_size(best_nodes, best_dropout):
    labels = []
    val_accuracies = []
    best_accuracy = 0.0
    best_batch = 0

    for batch_size in batch_size_configuation:
        model, history = train_CNN(
            batch_size=batch_size,
            epochs=20,
            nodes=best_nodes,
            dropout_rate=best_dropout
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
    best_nodes, best_dropout = tune_best_hyperparameters(
        batch_size=32,
        epochs=10,
        max_trials=5
    )

    best_batch, best_batch_accuracy = find_best_batch_size(
        best_nodes=best_nodes,
        best_dropout=best_dropout
    )

    final_model, final_history = train_CNN(
        save=True,
        batch_size=best_batch,
        epochs=20,
        nodes=best_nodes,
        dropout_rate=best_dropout
    )

    accurate_predictions, total_predictions = test_given_model(model=final_model)
    test_accuracy = accurate_predictions / total_predictions

    print("Best nodes:", best_nodes)
    print("Best dropout:", best_dropout)
    print("Best batch size:", best_batch)
    print("Validation Accuracy:", best_batch_accuracy)
    print(f"Final test accuracy: {test_accuracy}")

    plot_history(history=final_history)


if __name__ == "__main__":
    main()

