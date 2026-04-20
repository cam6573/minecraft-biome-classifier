from CNN import train_CNN, tune_best_hyperparameters, get_datasets
from test_all_image import test_given_model
import matplotlib
import os
import numpy as np
matplotlib.use('Agg')  # non-interactive backend, no display needed
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# Total Predictions: 539 
# Accurate Predictions: 505 
# Accuracy: 0.9369202226345084 
# Best nodes: [32, 64, 128, 128] 
# Best dropout: 0.2 
# Best batch size: 16 
# Validation Accuracy: 0.9222221970558167 
# Final test accuracy: 0.9369202226345084

batch_size_configuation = [16, 32, 64, 128]
RESOURCES_DIR = "resources/CNN"
os.makedirs(RESOURCES_DIR, exist_ok=True)


def find_best_batch_size(best_nodes, best_dropout, best_lr):
    labels = []
    val_accuracies = []
    best_accuracy = 0.0
    best_batch = batch_size_configuation[0]

    for batch_size in batch_size_configuation:
        model, history, _ = train_CNN(
            batch_size=batch_size,
            epochs=20,
            nodes=best_nodes,
            dropout_rate=best_dropout,
            learning_rate=best_lr
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


# AI WAS USED FOR EVALUATE MODEL


# ---------------------------------------------------------------------------
# Full evaluation — confusion matrix, precision/recall/F1, ROC-AUC
# ---------------------------------------------------------------------------
 
def evaluate_model(model, batch_size, class_names):
    _, val_ds, _ = get_datasets(batch_size=batch_size)
 
    y_true_all = []
    y_prob_all = []
 
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        y_prob_all.append(probs)
        y_true_all.append(labels.numpy())
 
    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = np.argmax(y_prob, axis=1)
    n_classes = len(class_names)
 
    # ------------------------------------------------------------------ #
    # 1. Confusion matrix (counts + normalised side-by-side)
    # ------------------------------------------------------------------ #
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in zip(
        axes,
        [cm,      cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"],
        ["d",     ".2f"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax)
        ticks = np.arange(n_classes)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        thresh = data.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(
                    j, i,
                    format(data[i, j], fmt),
                    ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black",
                    fontsize=9,
                )
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
 
    plt.tight_layout()
    plt.savefig(f"{RESOURCES_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
 
    # ------------------------------------------------------------------ #
    # 2. Precision / Recall / F1
    # ------------------------------------------------------------------ #
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
    )
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
 
    metrics_per_class = {
        "Precision": [report[c]["precision"] for c in class_names],
        "Recall":    [report[c]["recall"]    for c in class_names],
        "F1-Score":  [report[c]["f1-score"]  for c in class_names],
    }
 
    x      = np.arange(n_classes)
    width  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
 
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, (metric, values) in enumerate(metrics_per_class.items()):
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[idx])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
 
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Per-Class Precision, Recall & F1-Score")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{RESOURCES_DIR}/precision_recall_f1.png", dpi=150)
    plt.close()
 
    # ------------------------------------------------------------------ #
    # 3. ROC curves (One-vs-Rest)
    # ------------------------------------------------------------------ #
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
 
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_bin[:, i], y_prob[:, i])
 
    macro_auc    = roc_auc_score(y_bin, y_prob, average="macro")
    weighted_auc = roc_auc_score(y_bin, y_prob, average="weighted")
 
    print(f"\nROC-AUC (macro):    {macro_auc:.4f}")
    print(f"ROC-AUC (weighted): {weighted_auc:.4f}")
 
    palette = plt.cm.tab10(np.linspace(0, 0.9, n_classes))
 
    plt.figure(figsize=(10, 7))
    for i, color in zip(range(n_classes), palette):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f"{class_names[i]}  (AUC = {roc_auc[i]:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC Curves — One-vs-Rest\n"
        f"Macro AUC = {macro_auc:.4f}  |  Weighted AUC = {weighted_auc:.4f}"
    )
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{RESOURCES_DIR}/roc_curves.png", dpi=150)
    plt.close()
 
    # ------------------------------------------------------------------ #
    # 4. Summary bar chart
    # ------------------------------------------------------------------ #
    overall_accuracy = report["accuracy"]
    macro_precision  = report["macro avg"]["precision"]
    macro_recall     = report["macro avg"]["recall"]
    macro_f1         = report["macro avg"]["f1-score"]
 
    summary_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "ROC-AUC\n(macro)"]
    summary_values = [overall_accuracy, macro_precision, macro_recall, macro_f1, macro_auc]
    bar_colors     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(summary_labels, summary_values, color=bar_colors)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=11)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Model Performance Summary")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{RESOURCES_DIR}/summary_metrics.png", dpi=150)
    plt.close()
 
    return {
        "accuracy":        overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
        "macro_f1":        macro_f1,
        "macro_auc":       macro_auc,
        "weighted_auc":    weighted_auc,
    }
 



def main():
    best_nodes, best_dropout, best_lr = tune_best_hyperparameters(
        batch_size=32,
        epochs=15,
        max_trials=20
    )

    best_batch, best_batch_accuracy = find_best_batch_size(
        best_nodes=best_nodes,
        best_dropout=best_dropout,
        best_lr=best_lr
    )

    final_model, final_history, class_names = train_CNN(
        save=True,
        batch_size=best_batch,
        epochs=20,
        nodes=best_nodes,
        dropout_rate=best_dropout,
        learning_rate=best_lr
    )

    plot_history(history=final_history)

    metrics = evaluate_model(
        model=final_model,
        batch_size=best_batch,
        class_names=class_names,
    )

    accurate_predictions, total_predictions = test_given_model(model=final_model)
    test_accuracy = accurate_predictions / total_predictions

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    print(f"  Best nodes          : {best_nodes}")
    print(f"  Best dropout        : {best_dropout}")
    print(f"  Best learning rate  : {best_lr}")
    print(f"  Best batch size     : {best_batch}")
    print(f"  Validation accuracy : {best_batch_accuracy:.4f}")
    print(f"  Test accuracy       : {test_accuracy:.4f}")
    print(f"  Macro Precision     : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall        : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score      : {metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC (macro)     : {metrics['macro_auc']:.4f}")
    print(f"  ROC-AUC (weighted)  : {metrics['weighted_auc']:.4f}")
    print("=" * 55)
    print(f"\n  Total predictions    : {total_predictions}")
    print(f"  Accurate predictions : {accurate_predictions}")
    print("=" * 55)



if __name__ == "__main__":
    main()

