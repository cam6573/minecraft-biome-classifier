from CNN import train_CNN
from test_all_image import test_given_model
import matplotlib.pyplot as plt


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


def find_best_node_configuration():
    labels = []
    accuracies = []
    best_accuracy = 0.0
    best_conf = []

    for node_conf in node_configurations:
        model = train_CNN(nodes=node_conf)
        accurate_predictions, total_prediction = test_given_model(model=model)

        accuracy = accurate_predictions / total_prediction
        accuracies.append(accuracy)
        labels.append(str(node_conf))

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_conf = node_conf

    plt.figure(figsize=(10, 5))
    plt.plot(labels, accuracies, marker='o')
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
    accuracies = []
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in epoch_configuration:
        model = train_CNN(
            nodes=best_conf,
            epochs=epoch
        )
        accurate_predictions, total_prediction = test_given_model(model=model)

        accuracy = accurate_predictions / total_prediction
        accuracies.append(accuracy)
        labels.append(str(epoch))

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

    plt.figure(figsize=(10, 5))
    plt.plot(labels, accuracies, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for Different Epoch Configurations")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("resources/CNN/epoch_config_plot.png")
    plt.show()

    return best_epoch, best_accuracy


def main():
    best_conf, best_conf_accuracy = find_best_node_configuration()
    print("Best node configuration:", best_conf)
    print("Accuracy:", best_conf_accuracy)

    best_epoch, best_epoch_accuracy = find_best_epoch(best_conf)
    print("Best epoch:", best_epoch)
    print("Accuracy:", best_epoch_accuracy)


if __name__ == "__main__":
    main()


# Best node configuration: [64, 128, 256, 128]
# Best Epoch configuration: 20