from utils.shape_calculator import shape_after_layer, shape_after_operation, shape_to_tuple
from utils.config_interface import ConfigInterface

import torch.nn as nn


class ConvLayerConfig(ConfigInterface):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def is_valid(self):
        cfg = self.config

        required_fields = [
            'in_channels',
            'out_channels',
            'kernel_size',
        ]

        for field in required_fields:
            if field not in cfg:
                return False

        if cfg['in_channels'] <= 0:
            return False

        if cfg['out_channels'] <= 0:
            return False

        if cfg['kernel_size'] <= 0:
            return False

        if cfg['stride'] <= 0:
            return False

        if cfg['padding'] < 0:
            return False

        return True


class SmallerCNN(nn.Module):
    def __init__(self, input_shape, conv1_layer: ConvLayerConfig, conv2_layer: ConvLayerConfig):
        super().__init__()

        input_shape = shape_to_tuple(input_shape)

        # ConvLayerConfig verification
        assert conv1_layer.is_valid()
        assert conv2_layer.is_valid()

        self.conv1 = nn.Sequential(
            nn.Conv2d(**conv1_layer.config, bias=False),
            nn.BatchNorm2d(conv1_layer['out_channels']),
            nn.ReLU(inplace=True),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(**conv2_layer.config, bias=False),
            nn.BatchNorm2d(conv2_layer['out_channels']),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        conv1_shape = shape_after_layer(input_shape, conv1_layer)
        conv1_pooled_shape = shape_after_operation(conv1_shape, 2, 2, 0)
        conv2_shape = shape_after_layer(conv1_pooled_shape, conv2_layer)
        conv2_pooled_shape = shape_after_operation(conv2_shape, 2, 2, 0)

        self.fc1_input_size = conv2_pooled_shape.prod() * conv2_layer['out_channels']

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc1_input_size, 300, bias=False),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(300, 10),
        )

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_pool1 = self.pool1(x_conv1)
        x_conv2 = self.conv2(x_pool1)
        x = self.pool2(x_conv2)
        x = x.view(-1, self.fc1_input_size)
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2
