"""
Convolutional Neural Network Architectures
ResNet implementation with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection
    Implements: F(x) + x where F(x) is the residual mapping
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize residual block

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for first convolution
            downsample: Downsampling layer for skip connection
        """
        super(ResidualBlock, self).__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample for skip connection if needed
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        identity = x

        # First conv + BN + ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv + BN
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling to skip connection if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for deeper ResNets (ResNet-50, 101, 152)
    Uses 1x1, 3x3, 1x1 convolutions to reduce parameters
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(Bottleneck, self).__init__()

        # 1x1 convolution to reduce dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution to expand dimensions
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with bottleneck residual connection"""
        identity = x

        # 1x1 conv
        out = F.relu(self.bn1(self.conv1(x)))

        # 3x3 conv
        out = F.relu(self.bn2(self.conv2(out)))

        # 1x1 conv
        out = self.bn3(self.conv3(out))

        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet Architecture
    Configurable depth with residual blocks
    """

    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3
    ):
        """
        Initialize ResNet

        Args:
            block: Residual block type (ResidualBlock or Bottleneck)
            layers: Number of blocks in each layer
            num_classes: Number of output classes
            in_channels: Number of input channels (3 for RGB)
        """
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        """Create a ResNet layer with multiple blocks"""
        downsample = None

        # Create downsampling layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []

        # First block (may downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classifier
        x = self.fc(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier (for transfer learning)"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def ResNet18(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """ResNet-18 model"""
    return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes, in_channels)


def ResNet34(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """ResNet-34 model"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes, in_channels)


def ResNet50(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """ResNet-50 model"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels)


def ResNet101(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """ResNet-101 model"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels)


def ResNet152(num_classes: int = 1000, in_channels: int = 3) -> ResNet:
    """ResNet-152 model"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels)


# Example usage
if __name__ == "__main__":
    # Create ResNet-18
    model = ResNet18(num_classes=10)

    # Dummy input
    x = torch.randn(2, 3, 224, 224)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
