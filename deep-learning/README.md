# Deep Learning Framework

**Comprehensive PyTorch-based deep learning framework with implementations of major neural network architectures, training loops, transfer learning, optimizers, and GPU optimization.**

## 🎯 Framework Overview

Complete implementations of:
1. **Neural Network Architectures**: CNN, RNN, LSTM, Transformer
2. **Training Loops**: Forward/backward pass, optimizer integration, loss computation
3. **Transfer Learning**: Pre-trained model loading, feature extraction, fine-tuning
4. **Model Fine-Tuning**: Layer freezing, discriminative learning rates, gradual unfreezing
5. **Gradient Descent**: SGD, Adam, AdamW, RMSprop with custom implementations
6. **Batch Processing**: Efficient batching, gradient accumulation, distributed training
7. **Data Loaders**: PyTorch DataLoader, custom samplers, augmentation pipelines
8. **GPU Utilization**: Multi-GPU training, memory optimization, profiling
9. **Mixed Precision**: FP16 training with automatic mixed precision (AMP)

## 📁 Project Structure

```
deep-learning/
├── architectures/
│   ├── __init__.py
│   ├── cnn.py              # CNN architectures (ResNet, VGG, EfficientNet)
│   ├── rnn.py              # RNN and LSTM implementations
│   ├── transformer.py      # Transformer architecture
│   └── custom_layers.py    # Custom layer implementations
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Main training loop
│   ├── losses.py           # Loss functions
│   └── metrics.py          # Evaluation metrics
├── transfer/
│   ├── __init__.py
│   ├── pretrained.py       # Load pretrained models
│   ├── fine_tuning.py      # Fine-tuning strategies
│   └── feature_extraction.py
├── optimizers/
│   ├── __init__.py
│   ├── sgd.py              # Custom SGD implementation
│   ├── adam.py             # Custom Adam/AdamW
│   └── lr_schedulers.py    # Learning rate schedules
├── dataloaders/
│   ├── __init__.py
│   ├── datasets.py         # Custom dataset classes
│   ├── transforms.py       # Data augmentation
│   └── samplers.py         # Custom samplers
├── utils/
│   ├── __init__.py
│   ├── gpu_utils.py        # GPU utilities
│   ├── mixed_precision.py  # AMP training
│   └── checkpointing.py    # Model checkpointing
├── examples/
│   ├── train_cnn.py        # CNN training example
│   ├── train_transformer.py
│   ├── transfer_learning.py
│   └── distributed_training.py
└── tests/
```

## 🏗️ 1. Neural Network Architectures

### Convolutional Neural Networks (CNNs)

#### ResNet Implementation

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet architecture"""

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
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
        x = self.fc(x)

        return x


def resnet50(num_classes=1000):
    """ResNet-50 model"""
    return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes)
```

### Recurrent Neural Networks (RNNs & LSTMs)

#### LSTM Implementation

```python
class LSTM(nn.Module):
    """LSTM for sequence modeling"""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out


class BiLSTM(nn.Module):
    """Bidirectional LSTM"""

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True
        )

        # *2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
```

### Transformer Architecture

#### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(query.device)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        x = torch.matmul(attention, V)

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        x = self.out_linear(x)

        return x, attention


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class Transformer(nn.Module):
    """Complete Transformer model"""

    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=512,
        num_classes=10,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, num_classes)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) * torch.sqrt(torch.FloatTensor([self.embedding.embedding_dim])).to(x.device)
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = self.dropout(x)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Global average pooling + classification
        x = x.mean(dim=1)
        x = self.fc(x)

        return x
```

## 🔄 2. Training Loops

### Complete Training Loop

```python
class Trainer:
    """Complete training pipeline with forward/backward pass"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        mixed_precision=False,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Mixed precision training
        self.mixed_precision = mixed_precision
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Mixed precision forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.gradient_accumulation_steps
            else:
                # Standard forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss, accuracy

    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.val_loader)

        return avg_loss, accuracy

    def fit(self, num_epochs, checkpoint_dir='checkpoints'):
        """Complete training loop"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_acc = 0

        for epoch in range(num_epochs):
            print(f'\nEpoch: {epoch+1}/{num_epochs}')

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')

            # Validate
            val_loss, val_acc = self.validate()
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, f'{checkpoint_dir}/best_model.pth')
                print(f'✓ Saved best model (Val Acc: {val_acc:.2f}%)')

        return best_val_acc
```

## 🔄 3. Transfer Learning & Fine-Tuning

### Transfer Learning Pipeline

```python
class TransferLearning:
    """Transfer learning utilities"""

    @staticmethod
    def load_pretrained(model_name, num_classes, pretrained=True):
        """Load pretrained model and modify classifier"""
        import torchvision.models as models

        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return model

    @staticmethod
    def freeze_layers(model, freeze_until='layer3'):
        """Freeze layers for feature extraction"""
        freeze = True
        for name, param in model.named_parameters():
            if freeze_until in name:
                freeze = False
            param.requires_grad = not freeze

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

        return model

    @staticmethod
    def gradual_unfreezing(model, optimizer, unfreeze_layers):
        """Gradually unfreeze layers during training"""
        for name, param in model.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True
                # Add to optimizer
                optimizer.add_param_group({'params': param})


class DiscriminativeLearningRates:
    """Different learning rates for different layers"""

    def __init__(self, model, base_lr=1e-3, layer_lr_decay=0.95):
        self.model = model
        self.base_lr = base_lr
        self.layer_lr_decay = layer_lr_decay

    def get_param_groups(self):
        """Create parameter groups with different learning rates"""
        param_groups = []

        # Get all named parameters
        params = list(self.model.named_parameters())

        # Group by layers
        layer_groups = {}
        for name, param in params:
            layer = name.split('.')[0]
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append(param)

        # Assign learning rates
        for i, (layer, params) in enumerate(layer_groups.items()):
            lr = self.base_lr * (self.layer_lr_decay ** (len(layer_groups) - i - 1))
            param_groups.append({'params': params, 'lr': lr})
            print(f'Layer {layer}: lr = {lr:.6f}')

        return param_groups
```

## 🔢 4. Custom Gradient Descent Implementation

### Custom Optimizers

```python
class SGD:
    """Stochastic Gradient Descent from scratch"""

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        """Update parameters"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # L2 regularization
            if self.weight_decay != 0:
                param.grad.add_(param.data, alpha=self.weight_decay)

            # Momentum
            self.velocity[i].mul_(self.momentum).add_(param.grad, alpha=1)

            # Update
            param.data.add_(self.velocity[i], alpha=-self.lr)

    def zero_grad(self):
        """Zero gradients"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class Adam:
    """Adam optimizer from scratch"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [torch.zeros_like(p) for p in self.params]  # First moment
        self.v = [torch.zeros_like(p) for p in self.params]  # Second moment
        self.t = 0

    def step(self):
        """Update parameters"""
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            # L2 regularization
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

## 📦 5. Data Loaders & Batch Processing

### Custom Dataset & DataLoader

```python
class CustomDataset(torch.utils.data.Dataset):
    """Custom PyTorch dataset"""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# Efficient data loading
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### Data Augmentation Pipeline

```python
import torchvision.transforms as transforms

# Training augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transform
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 🖥️ 6. GPU Utilization

### Multi-GPU Training

```python
# DataParallel (single-machine, multi-GPU)
model = nn.DataParallel(model)
model = model.cuda()

# DistributedDataParallel (multi-machine)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Wrap model
model = DDP(model.cuda(), device_ids=[rank])
```

### GPU Memory Optimization

```python
# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x

# Empty cache
torch.cuda.empty_cache()

# Monitor GPU memory
print(f'Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print(f'Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
```

## ⚡ 7. Mixed Precision Training

### Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 📊 Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 1. Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 2. Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# 3. Create model
model = resnet50(num_classes=10).to(device)

# 4. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 5. Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    mixed_precision=True,
    gradient_accumulation_steps=4
)

best_acc = trainer.fit(num_epochs=100)
print(f'Best validation accuracy: {best_acc:.2f}%')
```

## 📚 Key Features

✅ **Complete Architectures**: ResNet, LSTM, Transformer implementations
✅ **Training Loops**: Forward/backward pass, optimizer integration
✅ **Transfer Learning**: Pretrained models, fine-tuning strategies
✅ **Custom Optimizers**: SGD, Adam from scratch
✅ **Efficient Data Loading**: PyTorch DataLoader with optimizations
✅ **GPU Optimization**: Multi-GPU, memory management
✅ **Mixed Precision**: FP16 training with AMP
✅ **Gradient Accumulation**: Large effective batch sizes
✅ **Learning Rate Scheduling**: Cosine annealing, step decay
✅ **Checkpointing**: Save/load model states

## 🚀 Performance Tips

1. **Use pin_memory=True** for faster GPU transfer
2. **Enable mixed precision** for 2-3x speedup
3. **Use gradient accumulation** for large batch sizes
4. **Implement gradient checkpointing** for memory savings
5. **Profile with torch.profiler** to find bottlenecks
6. **Use DistributedDataParallel** for multi-GPU
7. **Freeze early layers** for faster fine-tuning
8. **Use cosine annealing** for better convergence

---

**Complete deep learning framework with all major architectures and training techniques** 🚀
