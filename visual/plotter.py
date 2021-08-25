import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sn
import numpy as np


def plot_training_results(loss_data, train_data, test_data):

    loss_epochs, loss_values = zip(*loss_data)
    train_epochs, train_accuracies = zip(*train_data)
    test_epochs, test_accuracies = zip(*test_data)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout(pad=2)
    fig.suptitle('Model Training Results')

    ax1.plot(loss_epochs, loss_values, 'r-')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch #')
    ax1.set_ylabel('Loss')

    ax2.plot(train_epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(test_epochs, test_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Accuracies')
    ax2.set_xlabel('Epoch #')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.show()


def plot_class_images(class_data, plot_rows, plot_columns, class_labels=None):
    plot_idx = 0

    for target_class, class_image in class_data.items():
        plot_idx += 1
        ax = plt.subplot(plot_rows, plot_columns, plot_idx)
        plt.axis('off')
        class_title = target_class if class_labels is None else class_labels[target_class]
        ax.set_title(class_title)
        plt.imshow(class_image)

    plt.show()


def plot_class_images_with_quantity(class_data, plot_rows, plot_columns, class_labels=None):
    plot_idx = 0

    for target_class, (class_qty, class_image) in class_data.items():
        plot_idx += 1
        ax = plt.subplot(plot_rows, plot_columns, plot_idx)
        plt.axis('off')
        class_title = target_class if class_labels is None else class_labels[target_class]
        ax.set_title(f'{class_title} [{class_qty}]')
        plt.imshow(class_image)

    plt.show()


def plot_class_images_with_quantity_and_classification(class_data, classifications, plot_rows, plot_columns, class_labels=None):
    plot_idx = 0

    for target_class, (class_qty, class_image) in class_data.items():
        plot_idx += 1
        ax = plt.subplot(plot_rows, plot_columns, plot_idx)
        plt.axis('off')
        class_title = target_class if class_labels is None else class_labels[target_class]
        classification = classifications[target_class]
        classification_label = classification if class_labels is None else class_labels[classification]
        ax.set_title(f'{class_title} - {classification_label} [{class_qty}]')
        plt.imshow(class_image)

    plt.show()


def plot_kernel_activations(kernel_activations, plot_rows, plot_columns):
    plot_idx = 0

    for kernel in kernel_activations:
        print(kernel[0].shape)
        plot_idx += 1
        plt.subplot(plot_rows, plot_columns, plot_idx)
        plt.axis('off')
        plt.imshow(kernel)

    plt.show()


def plot_confusion_matrix(labels, predictions, class_labels=None):
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions)

    sn.set(font_scale=1.4)

    if class_labels is None:
        label_options = {}
    else:
        label_options = {
            'xticklabels': class_labels,
            'yticklabels': class_labels,
        }

    sn.heatmap(confusion_matrix, annot=True, annot_kws={'size': 16}, fmt='d', **label_options)

    plt.title('CNN Bias - Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
