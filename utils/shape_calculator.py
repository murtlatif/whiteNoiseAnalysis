import numpy as np
import torch


def shape_after_operation(old_input_shape, kernel_size, stride, padding):
    """
    Output shape = floor[(W - K + 2P)/S] + 1
    Output shape has a lower limit of 1.

    Where:
        W - Input Size,
        K - Kernel Size,
        P - Padding,
        S - Stride
    """
    return np.maximum(np.floor((np.subtract(old_input_shape, kernel_size) - (2*padding)) / stride) + 1, 1).astype(dtype=np.int32)


def shape_after_layer(old_input_shape, layer):
    kernel_size = layer.config['kernel_size']
    stride = layer.config['stride']
    padding = layer.config['padding']

    return shape_after_operation(old_input_shape, kernel_size, stride, padding)


def shape_to_tuple(input_shape):
    if type(input_shape) is tuple or type(input_shape) is torch.Size and len(input_shape) <= 2:
        if len(input_shape) == 1:
            input_shape = (input_shape, input_shape)

        return input_shape

    elif type(input_shape) is int:
        input_shape = (input_shape, input_shape)

        return input_shape

    else:
        raise ValueError('Invalid input shape provided.')
