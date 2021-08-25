from data.mnist_data import fetch_mnist
from data.file_manager import load_classification_data, load_training_results, save_classification_data, save_training_results, load_torch_data
from models.larger_cnn import BlockConfig, LargerCNN, SimpleBlock
from models.smaller_cnn import SmallerCNN, ConvLayerConfig
from noise import noise_analysis
from train.model_trainer import ModelTrainer
from train.training_config import TrainingConfig
from utils.cuda import get_device, has_cuda
from utils.argument_config import get_args
from utils.constants import FASHION_MNIST_LABELS
from visual import plotter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


def init_seed(seed: int):
    torch.manual_seed(seed)
    if has_cuda():
        torch.cuda.manual_seed(seed)


def get_class_labels(args):
    if args.fashion:
        return FASHION_MNIST_LABELS

    return None


def get_data(batch_size, test_batch_size, is_fashion):
    train_loader, test_loader = fetch_mnist(batch_size, test_batch_size, has_cuda(), is_fashion)
    return train_loader, test_loader


def get_larger_model_and_optimizer(learning_rate):
    block_config = BlockConfig(block=SimpleBlock, num_blocks=[2, 1, 3, 2], channels=[16, 32, 64, 128])
    cnn_model = LargerCNN(block_config=block_config, img_channels=1, num_classes=10)

    if has_cuda():
        cnn_model.cuda()

    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    return cnn_model, optimizer


def get_model_and_optimizer(learning_rate):
    conv1_layer = ConvLayerConfig(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
    conv2_layer = ConvLayerConfig(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
    cnn_model = SmallerCNN(input_shape=28, conv1_layer=conv1_layer, conv2_layer=conv2_layer)

    if has_cuda():
        cnn_model.cuda()

    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    return cnn_model, optimizer


def train_model(train_loader, test_loader, model, optimizer, epoch, args):
    training_config = TrainingConfig(
        epochs=args.epochs,
        loss_fnc=nn.CrossEntropyLoss().to(get_device()),
        save_path='models/cache' if args.save_model else None,
        save_name=args.model_name,
        starting_epoch=epoch,
        log_frequency=args.log_frequency,
    )

    loss_data, train_data, test_data = ModelTrainer.train(model, optimizer, train_loader, test_loader, training_config)

    training_results_filename = f'./data/trainingResults/{args.model_name}_training_results'

    # Save data
    if args.save_training:
        save_training_results(loss_data, train_data, test_data, training_results_filename)

    # Load data
    # loss_data, train_data, test_data = load_training_results(training_results_filename)

    # print(loss_data, train_data, test_data)
    if args.plot_data:
        plotter.plot_training_results(loss_data, train_data, test_data)


def load_model(learning_rate, load_model_path, large):
    if large:
        model, optimizer = get_larger_model_and_optimizer(learning_rate)
    else:
        model, optimizer = get_model_and_optimizer(learning_rate)

    epoch = 0
    if load_model_path:
        epoch = load_torch_data(load_model_path, model, optimizer)

    return model, optimizer, epoch


def show_model_summary_mnist(model):
    summary(model, (1, 28, 28))


def show_sample_images(test_loader, args):
    data, targets = test_loader.dataset.data, test_loader.dataset.targets

    sample_noise_stimulus = noise_analysis.get_sample_noise_stimulus(data, targets, args.gamma)

    plotter.plot_class_images(sample_noise_stimulus, 2, 5, get_class_labels(args))


def classification_analysis(test_loader, model, args):
    data, targets = test_loader.dataset.data, test_loader.dataset.targets

    target_classes = targets.unique(sorted=True).tolist()

    if args.load_class_data:
        class_data = load_classification_data(args.load_class_data)
    else:
        class_data = noise_analysis.get_classification_images(
            model, args.wn_num_batches, args.wn_batch_size, (data[0].shape), target_classes)

        # Save data
        classification_data_filepath = f'./data/classificationImages/{args.model_name}_{args.wn_batch_size}x{args.wn_num_batches}'
        save_classification_data(class_data, classification_data_filepath)

    classifications = noise_analysis.classify_classification_images(model, class_data)

    plotter.plot_class_images_with_quantity_and_classification(
        class_data, classifications, 2, 5, get_class_labels(args))

    average_noise_maps = np.empty((len(target_classes), data[0].shape[0], data[0].shape[1]))
    for target_class in target_classes:
        num_samples, average_noise_map = class_data[target_class]
        average_noise_maps[target_class] = average_noise_map

    output_indices = noise_analysis.classify_using_average_noise_map(average_noise_maps, data)
    plotter.plot_confusion_matrix(targets, output_indices, get_class_labels(args))


def spike_triggered_analysis(test_loader, model):
    kernel_activations = noise_analysis.get_kernel_activations(model, test_loader)
    for layer_idx in kernel_activations:
        cols = 8
        rows = int(np.ceil(kernel_activations[layer_idx].shape[0] / cols).item())
        plotter.plot_kernel_activations(kernel_activations[layer_idx].detach().numpy(), rows, cols)


def main():
    args = get_args()

    if args.seed:
        init_seed(args.seed)

    train_loader, test_loader = get_data(args.batch_size, args.test_batch_size, args.fashion)
    model, optimizer, epoch = load_model(args.learning_rate, args.load_model, args.large)

    if args.show_samples:
        show_sample_images(test_loader, args)

    if args.train:
        train_model(train_loader, test_loader, model, optimizer, epoch, args)

    if args.classification_analysis:
        classification_analysis(test_loader, model, args)

    if args.spike_triggered_analysis:
        spike_triggered_analysis(test_loader, model)

    if args.summary:
        show_model_summary_mnist(model)


if __name__ == '__main__':
    main()
