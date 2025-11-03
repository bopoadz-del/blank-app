"""
Transfer Learning Utilities
Load pretrained models, freeze layers, discriminative learning rates, fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import Optional, List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransferLearning:
    """
    Transfer Learning utilities for loading and adapting pretrained models
    """

    @staticmethod
    def load_pretrained_resnet(
        model_name: str = 'resnet50',
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ) -> nn.Module:
        """
        Load pretrained ResNet model

        Args:
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', etc.)
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone layers

        Returns:
            Modified ResNet model
        """
        logger.info(f"Loading {model_name} (pretrained={pretrained})")

        # Load model
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if requested
        if freeze_backbone:
            logger.info("Freezing backbone layers")
            for param in model.parameters():
                param.requires_grad = False

        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        logger.info(f"Replaced final layer: {num_features} -> {num_classes}")

        return model

    @staticmethod
    def load_pretrained_vgg(
        model_name: str = 'vgg16',
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ) -> nn.Module:
        """
        Load pretrained VGG model

        Args:
            model_name: VGG variant ('vgg11', 'vgg13', 'vgg16', 'vgg19')
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone layers

        Returns:
            Modified VGG model
        """
        logger.info(f"Loading {model_name} (pretrained={pretrained})")

        # Load model
        if model_name == 'vgg11':
            model = models.vgg11_bn(pretrained=pretrained)
        elif model_name == 'vgg13':
            model = models.vgg13_bn(pretrained=pretrained)
        elif model_name == 'vgg16':
            model = models.vgg16_bn(pretrained=pretrained)
        elif model_name == 'vgg19':
            model = models.vgg19_bn(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if requested
        if freeze_backbone:
            logger.info("Freezing backbone layers")
            for param in model.features.parameters():
                param.requires_grad = False

        # Replace classifier
        num_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        logger.info(f"Replaced classifier: {num_features} -> {num_classes}")

        return model

    @staticmethod
    def load_pretrained_efficientnet(
        model_name: str = 'efficientnet_b0',
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ) -> nn.Module:
        """
        Load pretrained EfficientNet model

        Args:
            model_name: EfficientNet variant
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone layers

        Returns:
            Modified EfficientNet model
        """
        logger.info(f"Loading {model_name} (pretrained={pretrained})")

        # Load model
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(pretrained=pretrained)
        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Freeze backbone if requested
        if freeze_backbone:
            logger.info("Freezing backbone layers")
            for param in model.parameters():
                param.requires_grad = False

        # Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )

        logger.info(f"Replaced classifier: {num_features} -> {num_classes}")

        return model

    @staticmethod
    def freeze_layers(
        model: nn.Module,
        freeze_until: Optional[str] = None,
        freeze_batch_norm: bool = True
    ):
        """
        Freeze layers up to a specific layer

        Args:
            model: Model to freeze
            freeze_until: Layer name to freeze until (None = freeze all)
            freeze_batch_norm: Also freeze batch normalization layers
        """
        freeze = True

        for name, param in model.named_parameters():
            if freeze_until and freeze_until in name:
                freeze = False
                logger.info(f"Unfreezing from: {name}")

            param.requires_grad = not freeze

        # Handle batch normalization
        if freeze_batch_norm:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

        # Count frozen/unfrozen parameters
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Frozen parameters: {frozen_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    @staticmethod
    def get_discriminative_lr_params(
        model: nn.Module,
        base_lr: float = 0.001,
        layer_lr_decay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups with discriminative learning rates
        Later layers get higher learning rates

        Args:
            model: Model
            base_lr: Base learning rate for final layer
            layer_lr_decay: LR decay factor for earlier layers

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []
        seen_params = set()

        # Get layer groups (reversed so later layers come first)
        layer_groups = []
        current_group = []

        for name, param in reversed(list(model.named_parameters())):
            if param.requires_grad:
                current_group.append((name, param))

                # Start new group at layer boundaries
                if 'layer' in name or 'fc' in name or 'classifier' in name:
                    if current_group:
                        layer_groups.append(current_group)
                        current_group = []

        if current_group:
            layer_groups.append(current_group)

        # Create parameter groups with decreasing learning rates
        for i, group in enumerate(layer_groups):
            lr = base_lr * (layer_lr_decay ** i)
            params = [p for _, p in group]

            param_groups.append({
                'params': params,
                'lr': lr
            })

            logger.info(
                f"Layer group {i}: {len(params)} params, lr={lr:.6f}"
            )

        return param_groups


class FineTuner:
    """
    Fine-tuning utilities with gradual unfreezing
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None
    ):
        """
        Initialize fine-tuner

        Args:
            model: Model to fine-tune
            optimizer: Optimizer
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Store layer names
        self.layer_names = [name for name, _ in model.named_parameters()]

    def gradual_unfreeze(
        self,
        num_layers_per_epoch: int = 1,
        start_from_end: bool = True
    ):
        """
        Gradually unfreeze layers

        Args:
            num_layers_per_epoch: Number of layers to unfreeze per call
            start_from_end: Start unfreezing from the end (True) or beginning (False)
        """
        # Get currently frozen parameters
        frozen_params = [
            name for name, param in self.model.named_parameters()
            if not param.requires_grad
        ]

        if not frozen_params:
            logger.info("All layers already unfrozen")
            return

        # Determine which layers to unfreeze
        if start_from_end:
            layers_to_unfreeze = frozen_params[-num_layers_per_epoch:]
        else:
            layers_to_unfreeze = frozen_params[:num_layers_per_epoch]

        # Unfreeze layers
        for name, param in self.model.named_parameters():
            if name in layers_to_unfreeze:
                param.requires_grad = True
                logger.info(f"Unfroze: {name}")

        # Update optimizer parameter groups
        self._update_optimizer()

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True

        self._update_optimizer()
        logger.info("All layers unfrozen")

    def freeze_batch_norm(self):
        """Freeze all batch normalization layers"""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

        logger.info("Batch normalization layers frozen")

    def _update_optimizer(self):
        """Update optimizer with current trainable parameters"""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Update optimizer
        self.optimizer.param_groups[0]['params'] = trainable_params


class FeatureExtractor:
    """
    Extract features from pretrained models for downstream tasks
    """

    def __init__(
        self,
        model: nn.Module,
        layer_name: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize feature extractor

        Args:
            model: Pretrained model
            layer_name: Layer to extract features from (None = before classifier)
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.layer_name = layer_name

        # Hook for feature extraction
        self.features = None
        if layer_name:
            self._register_hook(layer_name)

    def _register_hook(self, layer_name: str):
        """Register forward hook for feature extraction"""
        def hook(module, input, output):
            self.features = output.detach()

        # Find and register hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(hook)
                logger.info(f"Registered hook at: {layer_name}")
                return

        raise ValueError(f"Layer not found: {layer_name}")

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features

        Args:
            x: Input tensor

        Returns:
            Feature tensor
        """
        x = x.to(self.device)

        if self.layer_name:
            # Use hook
            _ = self.model(x)
            return self.features
        else:
            # Use model's feature extraction if available
            if hasattr(self.model, 'extract_features'):
                return self.model.extract_features(x)
            else:
                # Default: remove classifier and extract
                if isinstance(self.model, models.ResNet):
                    x = self.model.conv1(x)
                    x = self.model.bn1(x)
                    x = self.model.relu(x)
                    x = self.model.maxpool(x)
                    x = self.model.layer1(x)
                    x = self.model.layer2(x)
                    x = self.model.layer3(x)
                    x = self.model.layer4(x)
                    x = self.model.avgpool(x)
                    x = torch.flatten(x, 1)
                    return x
                else:
                    raise NotImplementedError("Feature extraction not implemented for this model")


# Example usage
if __name__ == "__main__":
    # Load pretrained ResNet50
    model = TransferLearning.load_pretrained_resnet(
        model_name='resnet50',
        num_classes=10,
        pretrained=True,
        freeze_backbone=True
    )

    # Freeze layers up to layer3
    TransferLearning.freeze_layers(model, freeze_until='layer3')

    # Get discriminative learning rate parameters
    param_groups = TransferLearning.get_discriminative_lr_params(
        model,
        base_lr=0.001,
        layer_lr_decay=0.5
    )

    # Create optimizer with discriminative learning rates
    optimizer = optim.Adam(param_groups)

    # Create fine-tuner
    fine_tuner = FineTuner(model, optimizer)

    # Gradual unfreezing example
    print("\nGradual unfreezing example:")
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        fine_tuner.gradual_unfreeze(num_layers_per_epoch=2)

    # Feature extraction example
    print("\nFeature extraction example:")
    extractor = FeatureExtractor(model, layer_name='layer4')

    dummy_input = torch.randn(2, 3, 224, 224)
    features = extractor.extract(dummy_input)
    print(f"Extracted features shape: {features.shape}")
