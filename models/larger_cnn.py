from utils.config_interface import ConfigInterface
import torch.nn as nn


class BlockConfig(ConfigInterface):
    def __init__(self, block, num_blocks, channels):
        super().__init__(
            block=block,
            num_blocks=num_blocks,
            channels=channels
        )

    def is_valid(self):

        block = self.config['block']
        num_blocks = self.config['num_blocks']
        channels = self.config['channels']

        if block is not SimpleBlock:
            print('a')
            return False

        if not isinstance(num_blocks, list) or len(num_blocks) != 4:
            print('b')
            return False

        if any(num_blocks_in_layer <= 0 for num_blocks_in_layer in num_blocks):
            print('c')
            return False

        if not isinstance(channels, list) or len(num_blocks) != 4:
            print('d')
            return False

        if any(channels_in_layer <= 0 for channels_in_layer in num_blocks):
            print('e')
            return False

        return True


class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class LargerCNN(nn.Module):
    def __init__(self, block_config: BlockConfig, img_channels, num_classes):
        super().__init__()

        print(f'Initialized Larger CNN with config: {block_config}')
        assert block_config.is_valid()

        self.in_channels = block_config['channels'][0]

        self.conv1 = nn.Conv2d(img_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        num_blocks = block_config['num_blocks']
        channels = block_config['channels']
        block = block_config['block']

        self.layer1 = self._make_layer(block, num_blocks[0], channels[0])
        self.layer2 = self._make_layer(block, num_blocks[1], channels[1])
        self.layer3 = self._make_layer(block, num_blocks[2], channels[2], stride=2)
        self.layer4 = self._make_layer(block, num_blocks[3], channels[3])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, num_blocks, out_channels, stride=1):
        layers = []

        downsample = None
        if self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Only the first block in a layer has stride/downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        for i in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels))

        self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_flat = x.view(x.shape[0], -1)
        x = self.fc(x_flat)

        return x
