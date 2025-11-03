# Training Infrastructure

Advanced training utilities for deep learning with PyTorch.

## Features

### Trainer Module (`trainer.py`)
- **AdvancedTrainer**: Complete training loop with all modern features
  - Early stopping with configurable patience
  - Learning rate scheduling
  - Model checkpointing with versioning
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Gradient clipping
  - TensorBoard integration
  - Custom callbacks
  - Progress tracking

- **EarlyStopping**: Prevent overfitting
  - Monitors validation metrics
  - Configurable patience
  - Min/max mode support

- **ModelCheckpoint**: Save best models
  - Save based on metric
  - Keep top-k checkpoints
  - Automatic cleanup
  - Full state saving (model, optimizer, scheduler)

### Distributed Module (`distributed.py`)
- **MultiGPUTrainer**: DataParallel for multi-GPU
  - Simple multi-GPU training
  - Automatic model wrapping
  - Easy checkpoint management

- **DistributedTrainer**: DistributedDataParallel
  - Efficient multi-GPU and multi-node training
  - Process group initialization
  - Distributed data sampling
  - Better performance than DataParallel

- **GradientAccumulator**: Effective larger batch sizes
  - Accumulate gradients over multiple steps
  - Gradient clipping support
  - Memory-efficient training

## Usage

### Basic Training

```python
from trainer import AdvancedTrainer, EarlyStopping, ModelCheckpoint

# Create model, optimizer, criterion
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Setup early stopping
early_stopping = EarlyStopping(patience=5, mode='min')

# Setup checkpointing
checkpoint = ModelCheckpoint(
    save_dir='checkpoints',
    monitor='val_loss',
    mode='min',
    save_top_k=3
)

# Create trainer
trainer = AdvancedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    use_amp=True,
    gradient_accumulation_steps=4,
    early_stopping=early_stopping,
    checkpoint=checkpoint
)

# Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)
```

### Multi-GPU Training

```python
from distributed import MultiGPUTrainer

# Wrap model for multi-GPU
multi_gpu = MultiGPUTrainer(model, gpu_ids=[0, 1, 2, 3])

# Use multi_gpu.model in training
# Checkpoints are automatically unwrapped
multi_gpu.save_checkpoint('checkpoint.pt', epoch=10)
```

### Distributed Training

```python
from distributed import DistributedTrainer

# Initialize distributed trainer
trainer = DistributedTrainer(
    model=model,
    rank=rank,
    world_size=world_size
)

# Create distributed dataloader
train_loader = trainer.create_dataloader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

# Train as usual
# Checkpoints saved only on rank 0
trainer.save_checkpoint('checkpoint.pt')
```

## Requirements

- PyTorch >= 2.0.0
- TensorBoard (optional)
- NVIDIA GPU with CUDA (for AMP and multi-GPU)
