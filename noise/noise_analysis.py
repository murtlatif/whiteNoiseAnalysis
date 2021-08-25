import torch
import random
import numpy as np


from utils.shape_calculator import shape_to_tuple
from train import model_runner


def generate_noise_2d(shape):
    noise = torch.rand(shape)
    return noise


def noise_stimulus(signal, noise, gamma):
    scaled_signal = gamma * signal
    scaled_noise = (1-gamma) * noise

    return scaled_signal + scaled_noise


def get_class_data(data, targets):
    target_classes = targets.unique()

    class_data = {}

    for target_class in target_classes:
        current_class_data = data[targets == target_class]
        class_data[target_class.item()] = current_class_data

    return class_data


def get_mean_images(data, targets):
    class_data = get_class_data(data, targets)

    for target_class in class_data:
        class_data[target_class] = class_data[target_class].mean(0)

    return class_data


def get_random_sample_per_class(data, targets):
    class_data = get_class_data(data, targets)

    for target_class in class_data:
        random_sample = random.choice(class_data[target_class])
        class_data[target_class] = random_sample

    return class_data


def get_sample_noise_stimulus(data, targets, gamma):
    class_data = get_random_sample_per_class(data, targets)

    for target_class in class_data:
        target_class_data = class_data[target_class].float() / 255
        noise = generate_noise_2d(target_class_data.shape)
        stimulus = noise_stimulus(target_class_data, noise, gamma)

        class_data[target_class] = stimulus

    return class_data


def get_classification_images(model, num_batches, batch_size, img_shape, classes):

    model.eval()

    noise = {}

    for batch_idx in range(num_batches):
        noise_shape = (batch_size, *img_shape)
        batch_noise = generate_noise_2d(noise_shape)

        output_indices, _ = model_runner.predict_batch(model, batch_noise)

        print('\rWhite Noise Batch {}/{} [{}/{} ({}%)]'.format(
            batch_idx + 1,
            num_batches,
            (batch_idx + 1) * batch_size,
            num_batches * batch_size,
            100 * (batch_idx + 1) / num_batches
        ), end='' if (batch_idx+1) < num_batches else None)

        for target_class in classes:
            class_noise = batch_noise[output_indices == target_class]
            num_samples = class_noise.shape[0]
            total_class_noise = class_noise.sum(0)

            if target_class not in noise:
                noise[target_class] = (num_samples, total_class_noise)
            else:
                if num_samples == 0:
                    continue

                current_samples, current_total_noise = noise[target_class]
                new_samples = num_samples + current_samples
                new_total_noise = total_class_noise + current_total_noise
                noise[target_class] = (new_samples, new_total_noise)

    for target_class, noise_data in noise.items():
        num_samples, total_noise = noise_data
        if (num_samples > 0):
            noise[target_class] = (num_samples, total_noise / num_samples)

    return noise


def classify_classification_images(model, classification_data):

    class_data = {}

    for target_class, (num_samples, classification_image) in classification_data.items():
        output_class, _ = model_runner.predict_one(model, classification_image)
        class_data[target_class] = output_class.item()

    return class_data


def classify_using_average_noise_map(average_noise_maps, data):

    data = data[:, None, ...]

    outputs = np.multiply(average_noise_maps, data).sum((2, 3))
    output_indices = np.argmax(outputs, 1)

    return output_indices


def get_kernel_activations(model, data):

    model.eval()

    _, outputs = model_runner.predict_batch(model, data)

    kernel_activations = []
    for output_idx in range(1, len(outputs)):
        output = outputs[output_idx]
        kernel_activation = output.squeeze(1).float().mean(0)
        print(output.shape, kernel_activation.shape)
        kernel_activations.append(kernel_activation)

    return kernel_activations
