# Computer Vision Training Framework

Comprehensive training framework for computer vision tasks including image classification, object detection, and segmentation with advanced data augmentation and evaluation metrics.

## 📋 Features

### Training Pipelines
- **Image Classification**: ResNet, EfficientNet, Vision Transformer
- **Object Detection**: YOLOv8, Faster R-CNN, RetinaNet
- **Instance Segmentation**: Mask R-CNN, YOLACT
- **Semantic Segmentation**: U-Net, DeepLabV3, SegFormer

### Data Augmentation
- **Geometric**: Rotation, flip, crop, resize, affine transformations
- **Color**: Brightness, contrast, saturation, hue adjustments
- **Advanced**: CutMix, MixUp, AutoAugment, RandAugment
- **Domain-Specific**: Blur, noise, weather effects

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Detection**: mAP, IoU, Precision-Recall curves
- **Segmentation**: Dice coefficient, IoU, Pixel accuracy

### Dataset Management
- **Formats**: COCO, Pascal VOC, YOLO, ImageNet
- **Annotation**: Integration with LabelImg, CVAT, Labelme
- **Validation**: Dataset integrity checks, class distribution analysis
- **Splitting**: Train/val/test splits with stratification

## 🚀 Quick Start

### Image Classification

```python
from classification.trainer import ClassificationTrainer
from classification.models import create_resnet50
from augmentation.transforms import get_classification_transforms

# Create model
model = create_resnet50(num_classes=10)

# Setup augmentation
train_transforms = get_classification_transforms(mode='train', image_size=224)
val_transforms = get_classification_transforms(mode='val', image_size=224)

# Create trainer
trainer = ClassificationTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001
)

# Train
trainer.train()
```

### Object Detection (YOLOv8)

```python
from detection.yolo_trainer import YOLOv8Trainer
from detection.datasets import COCODataset

# Load dataset
train_dataset = COCODataset(
    root='./data/coco',
    split='train',
    transforms=get_detection_transforms()
)

# Create trainer
trainer = YOLOv8Trainer(
    model_size='n',  # nano, small, medium, large, xlarge
    num_classes=80,
    img_size=640
)

# Train
trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=300,
    batch_size=16
)
```

### Semantic Segmentation

```python
from segmentation.trainer import SegmentationTrainer
from segmentation.models import create_unet

# Create U-Net model
model = create_unet(
    in_channels=3,
    num_classes=21,
    encoder='resnet50'
)

# Train
trainer = SegmentationTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    criterion='dice_ce',  # Dice + CrossEntropy
    epochs=100
)

trainer.train()
```

## 📁 Project Structure

```
computer-vision/
├── classification/
│   ├── trainer.py           # Image classification trainer
│   ├── models.py            # Classification models (ResNet, EfficientNet, ViT)
│   └── datasets.py          # ImageNet, CIFAR datasets
│
├── detection/
│   ├── yolo_trainer.py      # YOLOv8 training
│   ├── faster_rcnn.py       # Faster R-CNN training
│   ├── retinanet.py         # RetinaNet training
│   ├── datasets.py          # COCO, Pascal VOC datasets
│   └── anchors.py           # Anchor generation
│
├── segmentation/
│   ├── trainer.py           # Segmentation trainer
│   ├── models.py            # U-Net, DeepLab, SegFormer
│   ├── datasets.py          # Segmentation datasets
│   └── losses.py            # Dice, Focal, Lovasz losses
│
├── augmentation/
│   ├── transforms.py        # Augmentation pipelines
│   ├── geometric.py         # Geometric transforms
│   ├── color.py             # Color transforms
│   ├── advanced.py          # CutMix, MixUp, AutoAugment
│   └── albumentation.py     # Albumentations integration
│
├── preprocessing/
│   ├── normalize.py         # Normalization utilities
│   ├── resize.py            # Resizing strategies
│   └── format_conversion.py # Format converters
│
├── evaluation/
│   ├── classification_metrics.py  # Classification metrics
│   ├── detection_metrics.py       # mAP, IoU calculation
│   ├── segmentation_metrics.py    # Dice, IoU, pixel accuracy
│   └── visualization.py           # Result visualization
│
├── datasets/
│   ├── coco.py              # COCO dataset
│   ├── voc.py               # Pascal VOC dataset
│   ├── imagenet.py          # ImageNet dataset
│   ├── custom.py            # Custom dataset templates
│   └── annotations.py       # Annotation parsers
│
├── utils/
│   ├── dataset_manager.py   # Dataset organization
│   ├── annotation_tools.py  # Annotation integration
│   ├── validation.py        # Dataset validation
│   └── splitting.py         # Train/val/test splitting
│
└── examples/
    ├── train_classification.py
    ├── train_yolo.py
    ├── train_segmentation.py
    └── evaluate_model.py
```

## 🔧 Installation

```bash
# Install base dependencies
pip install torch torchvision opencv-python albumentations

# Install detection dependencies
pip install ultralytics  # YOLOv8
pip install pycocotools

# Install segmentation dependencies
pip install segmentation-models-pytorch

# Install annotation tools
pip install labelme cvat-sdk
```

## 📊 Supported Datasets

### Classification
- ImageNet
- CIFAR-10/100
- Custom image folders

### Object Detection
- COCO (Common Objects in Context)
- Pascal VOC
- YOLO format
- Custom annotations

### Segmentation
- Cityscapes
- ADE20K
- Pascal VOC Segmentation
- Custom masks

## 🎨 Data Augmentation Examples

### Basic Augmentation
```python
from augmentation.transforms import BasicAugmentation

augmentation = BasicAugmentation(
    rotation=15,
    horizontal_flip=True,
    vertical_flip=False,
    brightness=0.2,
    contrast=0.2
)

augmented_image = augmentation(image)
```

### Advanced Augmentation
```python
from augmentation.advanced import CutMix, MixUp, AutoAugment

# CutMix for object detection
cutmix = CutMix(alpha=1.0, prob=0.5)
mixed_images, mixed_targets = cutmix(images, targets)

# MixUp for classification
mixup = MixUp(alpha=0.2)
mixed_images, mixed_labels = mixup(images, labels)

# AutoAugment
auto_aug = AutoAugment(policy='imagenet')
augmented = auto_aug(image)
```

## 📈 Evaluation Metrics

### Object Detection Metrics
```python
from evaluation.detection_metrics import calculate_map, calculate_iou

# Calculate mAP
map_50 = calculate_map(predictions, ground_truth, iou_threshold=0.5)
map_75 = calculate_map(predictions, ground_truth, iou_threshold=0.75)

# Calculate IoU for single prediction
iou = calculate_iou(pred_box, gt_box)
```

### Segmentation Metrics
```python
from evaluation.segmentation_metrics import dice_coefficient, pixel_accuracy

# Dice coefficient
dice = dice_coefficient(pred_mask, gt_mask)

# Pixel accuracy
accuracy = pixel_accuracy(pred_mask, gt_mask)

# Mean IoU
miou = mean_iou(pred_mask, gt_mask, num_classes=21)
```

## 🗂️ Dataset Management

### Create Dataset from Annotations
```python
from utils.dataset_manager import DatasetManager

manager = DatasetManager(root_dir='./data')

# Parse annotations
manager.parse_coco_annotations('annotations.json')

# Validate dataset
validation_report = manager.validate_dataset()

# Split dataset
manager.split_dataset(train=0.7, val=0.15, test=0.15, stratify=True)

# Export to different format
manager.export_to_yolo('./output/yolo_format')
```

### Annotation Tools Integration
```python
from utils.annotation_tools import LabelmeParser, CVATParser

# Parse Labelme annotations
labelme_parser = LabelmeParser()
annotations = labelme_parser.parse_directory('./labelme_annotations')

# Convert to COCO format
coco_annotations = labelme_parser.to_coco_format(annotations)
```

## 🎯 Training Tips

### Image Classification
- Use ImageNet pretrained weights for transfer learning
- Apply progressive resizing (start small, increase size)
- Use label smoothing for better generalization
- Apply test-time augmentation (TTA) for inference

### Object Detection
- Use mosaic augmentation for small objects
- Warm up learning rate for first few epochs
- Use anchor-free detectors (YOLOv8, FCOS) for simplicity
- Apply NMS post-processing carefully

### Semantic Segmentation
- Use weighted loss for imbalanced classes
- Apply deep supervision for better gradients
- Use multi-scale training and inference
- Consider using auxiliary losses

## 📝 Model Zoo

### Classification Models
| Model | Top-1 Acc | Parameters | Speed (img/s) |
|-------|-----------|------------|---------------|
| ResNet-50 | 76.1% | 25.6M | 450 |
| EfficientNet-B0 | 77.3% | 5.3M | 380 |
| ViT-Base | 81.8% | 86M | 120 |

### Detection Models
| Model | mAP@50 | Parameters | FPS |
|-------|--------|------------|-----|
| YOLOv8n | 37.3% | 3.2M | 280 |
| YOLOv8s | 44.9% | 11.2M | 180 |
| Faster R-CNN | 42.0% | 41.8M | 25 |

### Segmentation Models
| Model | mIoU | Parameters | FPS |
|-------|------|------------|-----|
| U-Net | 72.1% | 31.0M | 45 |
| DeepLabV3+ | 79.2% | 41.3M | 30 |
| SegFormer | 81.5% | 47.2M | 35 |

## 🚀 Advanced Features

### Multi-GPU Training
```python
trainer = ClassificationTrainer(
    model=model,
    train_dataset=train_dataset,
    distributed=True,
    world_size=4  # 4 GPUs
)
```

### Mixed Precision Training
```python
trainer.train(
    epochs=100,
    mixed_precision=True  # Use FP16
)
```

### Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=100,
    steps_per_epoch=len(train_loader)
)
```

## 📊 Monitoring and Logging

### TensorBoard Integration
```python
trainer = ClassificationTrainer(
    model=model,
    train_dataset=train_dataset,
    log_dir='./runs/experiment_1'
)

# View logs
# tensorboard --logdir=./runs
```

### MLflow Integration
```python
import mlflow

with mlflow.start_run():
    trainer.train()
    mlflow.log_metrics({
        'accuracy': accuracy,
        'loss': loss
    })
```

## 🔬 Experimentation

### Hyperparameter Tuning
```python
from utils.tuning import grid_search

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'weight_decay': [1e-4, 1e-5, 1e-6]
}

best_params = grid_search(
    model_fn=create_resnet50,
    param_grid=param_grid,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Overfitting**
   - Increase data augmentation
   - Add dropout layers
   - Use weight decay

3. **Slow Training**
   - Use multi-GPU training
   - Optimize data loading (num_workers)
   - Use mixed precision

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Albumentations](https://albumentations.ai/)
- [PyTorch Lightning](https://lightning.ai/)

## 📄 License

MIT License

---

Built with ❤️ for Computer Vision Research and Production
