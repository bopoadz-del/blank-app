"""
Distributed and Multi-GPU Training
Support for DataParallel, DistributedDataParallel, and multi-GPU training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Callable, Dict, Any
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiGPUTrainer:
    """
    Multi-GPU training with DataParallel

    Simpler but less efficient than DistributedDataParallel.
    Good for single-node multi-GPU training.

    Features:
    - Automatic model wrapping
    - Batch splitting across GPUs
    - Easy to use
    """

    def __init__(
        self,
        model: nn.Module,
        gpu_ids: Optional[list] = None
    ):
        """
        Initialize multi-GPU trainer

        Args:
            model: Model to train
            gpu_ids: List of GPU IDs to use (None = all available)
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))

        if len(gpu_ids) == 0:
            raise ValueError("No GPUs specified")

        self.gpu_ids = gpu_ids
        self.device = f'cuda:{gpu_ids[0]}'

        # Move model to primary GPU
        model = model.to(self.device)

        # Wrap with DataParallel
        if len(gpu_ids) > 1:
            self.model = DP(model, device_ids=gpu_ids)
            logger.info(f"Using DataParallel with {len(gpu_ids)} GPUs: {gpu_ids}")
        else:
            self.model = model
            logger.info(f"Using single GPU: {gpu_ids[0]}")

        self.num_gpus = len(gpu_ids)

    def get_model(self) -> nn.Module:
        """Get underlying model (unwrapped)"""
        if isinstance(self.model, DP):
            return self.model.module
        return self.model

    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint (unwraps model if needed)"""
        state_dict = self.get_model().state_dict()
        checkpoint = {'model_state_dict': state_dict, **kwargs}
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.get_model().load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint


class DistributedTrainer:
    """
    Distributed training with DistributedDataParallel

    More efficient than DataParallel, supports multi-node training.

    Features:
    - Process group initialization
    - Distributed data sampling
    - Gradient synchronization
    - Multi-node support
    - Better performance than DataParallel
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        world_size: int,
        backend: str = 'nccl',
        init_method: str = 'env://'
    ):
        """
        Initialize distributed trainer

        Args:
            model: Model to train
            rank: Process rank (GPU ID)
            world_size: Total number of processes
            backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
            init_method: Initialization method
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size
            )
            logger.info(f"Process group initialized: rank={rank}, world_size={world_size}")

        # Set device
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(rank)

        # Move model to device
        model = model.to(self.device)

        # Wrap with DDP
        self.model = DDP(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            output_device=rank if torch.cuda.is_available() else None
        )

        logger.info(f"Rank {rank}: Model wrapped with DDP")

    def get_model(self) -> nn.Module:
        """Get underlying model (unwrapped)"""
        return self.model.module

    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """
        Create distributed dataloader

        Args:
            dataset: Dataset
            batch_size: Batch size per GPU
            shuffle: Shuffle data
            num_workers: Number of data loading workers
            **kwargs: Additional DataLoader arguments

        Returns:
            Distributed DataLoader
        """
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )

        return loader

    def save_checkpoint(self, path: str, **kwargs):
        """Save checkpoint (only on rank 0)"""
        if self.rank == 0:
            state_dict = self.get_model().state_dict()
            checkpoint = {'model_state_dict': state_dict, **kwargs}
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")

        # Wait for all processes
        dist.barrier()

    def load_checkpoint(self, path: str):
        """Load checkpoint (all ranks)"""
        # Map to current device
        map_location = {'cuda:0': f'cuda:{self.rank}'}
        checkpoint = torch.load(path, map_location=map_location)
        self.get_model().load_state_dict(checkpoint['model_state_dict'])

        if self.rank == 0:
            logger.info(f"Checkpoint loaded: {path}")

        dist.barrier()
        return checkpoint

    def barrier(self):
        """Synchronization barrier"""
        dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM):
        """All-reduce operation"""
        dist.all_reduce(tensor, op=op)
        return tensor

    def cleanup(self):
        """Cleanup distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(f"Rank {self.rank}: Process group destroyed")


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Setup distributed training environment

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(
        backend=backend,
        init_method='env://',
        rank=rank,
        world_size=world_size
    )


def cleanup_distributed():
    """Cleanup distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_distributed(
    train_fn: Callable,
    world_size: int,
    backend: str = 'nccl',
    **kwargs
):
    """
    Run distributed training

    Args:
        train_fn: Training function (receives rank as first argument)
        world_size: Number of processes
        backend: Communication backend
        **kwargs: Additional arguments for train_fn
    """
    mp.spawn(
        train_fn,
        args=(world_size, backend, kwargs),
        nprocs=world_size,
        join=True
    )


class GradientAccumulator:
    """
    Gradient accumulation for larger effective batch sizes

    Useful when:
    - GPU memory is limited
    - Want larger batch size than GPU can fit
    - Training large models
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        max_grad_norm: Optional[float] = 1.0
    ):
        """
        Initialize gradient accumulator

        Args:
            model: Model to train
            optimizer: Optimizer
            accumulation_steps: Accumulate gradients over n steps
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.step_count = 0

    def step(self, loss: torch.Tensor) -> bool:
        """
        Accumulate gradients and optionally update

        Args:
            loss: Loss tensor

        Returns:
            True if optimizer step was performed
        """
        # Scale loss
        loss = loss / self.accumulation_steps

        # Backward pass
        loss.backward()

        self.step_count += 1

        # Update weights after accumulation
        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            return True

        return False

    def reset(self):
        """Reset accumulator"""
        self.step_count = 0
        self.optimizer.zero_grad()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Distributed and Multi-GPU Training Test")
    print("=" * 70)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available, skipping multi-GPU tests")
        exit(0)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Test Multi-GPU (DataParallel)
    print("\n1. Multi-GPU with DataParallel")
    print("-" * 70)

    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    if num_gpus > 1:
        multi_gpu = MultiGPUTrainer(model, gpu_ids=[0, 1] if num_gpus > 1 else [0])
        print(f"Using {multi_gpu.num_gpus} GPUs")
        print(f"Device: {multi_gpu.device}")

        # Test forward pass
        dummy_input = torch.randn(32, 10).to(multi_gpu.device)
        output = multi_gpu.model(dummy_input)
        print(f"Output shape: {output.shape}")

        # Test checkpoint
        multi_gpu.save_checkpoint('test_checkpoint.pt', epoch=1)
        multi_gpu.load_checkpoint('test_checkpoint.pt')
        print("Checkpoint save/load successful")

    else:
        print("Only 1 GPU available, skipping DataParallel test")

    # Test Gradient Accumulation
    print("\n2. Gradient Accumulation")
    print("-" * 70)

    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    accumulator = GradientAccumulator(
        model=model,
        optimizer=optimizer,
        accumulation_steps=4
    )

    criterion = nn.CrossEntropyLoss()

    # Simulate training with small batches
    print("Simulating gradient accumulation...")
    for i in range(8):
        dummy_input = torch.randn(8, 10).cuda()
        dummy_target = torch.randint(0, 2, (8,)).cuda()

        output = model(dummy_input)
        loss = criterion(output, dummy_target)

        updated = accumulator.step(loss)

        if updated:
            print(f"Step {i+1}: Optimizer updated")
        else:
            print(f"Step {i+1}: Gradient accumulated")

    print("\n" + "=" * 70)
    print("Distributed training utilities tested successfully!")
