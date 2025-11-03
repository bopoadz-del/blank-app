"""
GPU Utilities and Multi-GPU Training
DataParallel, DistributedDataParallel, and distributed training setup
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, List
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get appropriate device (CPU or CUDA)

    Args:
        device_id: Specific GPU device ID (None = use default)

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cuda')

        logger.info(f"Using device: {device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        logger.info("CUDA not available, using CPU")

    return device


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Setup distributed training environment

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    logger.info(f"Distributed training initialized: rank {rank}/{world_size}")


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()
    logger.info("Distributed training cleaned up")


class MultiGPUTrainer:
    """
    Multi-GPU training with DataParallel or DistributedDataParallel
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = 'dataparallel',  # 'dataparallel' or 'distributed'
        device_ids: Optional[List[int]] = None,
        find_unused_parameters: bool = False
    ):
        """
        Initialize multi-GPU trainer

        Args:
            model: Model to train
            mode: Training mode ('dataparallel' or 'distributed')
            device_ids: List of GPU device IDs (None = use all)
            find_unused_parameters: Find unused parameters in DDP
        """
        self.mode = mode

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # Get device IDs
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids

        logger.info(f"Multi-GPU training mode: {mode}")
        logger.info(f"Using GPUs: {self.device_ids}")
        logger.info(f"Number of GPUs: {len(self.device_ids)}")

        # Setup model
        if mode == 'dataparallel':
            self.model = self._setup_dataparallel(model)
        elif mode == 'distributed':
            self.model = self._setup_distributed(model, find_unused_parameters)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _setup_dataparallel(self, model: nn.Module) -> nn.Module:
        """
        Setup DataParallel

        DataParallel:
        - Easier to use (single process)
        - Can cause GPU memory imbalance
        - Less efficient for multi-node
        """
        # Move model to primary GPU
        device = torch.device(f'cuda:{self.device_ids[0]}')
        model = model.to(device)

        # Wrap with DataParallel
        if len(self.device_ids) > 1:
            model = DataParallel(model, device_ids=self.device_ids)
            logger.info("Model wrapped with DataParallel")
        else:
            logger.info("Single GPU mode")

        return model

    def _setup_distributed(
        self,
        model: nn.Module,
        find_unused_parameters: bool = False
    ) -> nn.Module:
        """
        Setup DistributedDataParallel

        DistributedDataParallel:
        - More efficient (multi-process)
        - Better GPU memory balance
        - Supports multi-node training
        - Requires environment setup
        """
        # Get rank from environment
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        # Move model to device
        model = model.to(device)

        # Wrap with DistributedDataParallel
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters
        )

        logger.info(f"Model wrapped with DistributedDataParallel (rank {rank})")

        return model

    def get_model(self) -> nn.Module:
        """Get the underlying model"""
        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            return self.model.module
        return self.model


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for distributed training

    Args:
        dataset: Dataset
        batch_size: Batch size per GPU
        num_workers: Number of worker processes
        shuffle: Shuffle data
        pin_memory: Pin memory for faster transfer

    Returns:
        DataLoader with DistributedSampler
    """
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        shuffle=shuffle
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader


def synchronize():
    """
    Synchronize all processes in distributed training
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_world_size() -> int:
    """
    Get world size (total number of processes)

    Returns:
        World size
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """
    Get current process rank

    Returns:
        Rank
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """
    Check if current process is the main process

    Returns:
        True if main process
    """
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor across all processes

    Args:
        tensor: Tensor to reduce

    Returns:
        Reduced tensor
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    # Clone to avoid modifying original
    tensor = tensor.clone()

    # All-reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Average
    tensor /= get_world_size()

    return tensor


def gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensors from all processes

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes
    """
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]

    world_size = get_world_size()

    # Prepare tensor list
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    # All-gather
    dist.all_gather(tensor_list, tensor)

    return tensor_list


class GPUMemoryTracker:
    """
    Track GPU memory usage during training
    """

    def __init__(self, device: torch.device):
        """
        Initialize memory tracker

        Args:
            device: Device to track
        """
        self.device = device

        if not device.type == 'cuda':
            raise ValueError("Memory tracker only works with CUDA devices")

    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        torch.cuda.reset_peak_memory_stats(self.device)

    def get_memory_stats(self) -> dict:
        """
        Get current memory statistics

        Returns:
            Dictionary with memory stats
        """
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        max_reserved = torch.cuda.max_memory_reserved(self.device) / 1e9

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'max_reserved_gb': max_reserved
        }

    def print_memory_stats(self):
        """Print memory statistics"""
        stats = self.get_memory_stats()

        logger.info("GPU Memory Stats:")
        logger.info(f"  Allocated: {stats['allocated_gb']:.2f} GB")
        logger.info(f"  Reserved: {stats['reserved_gb']:.2f} GB")
        logger.info(f"  Max Allocated: {stats['max_allocated_gb']:.2f} GB")
        logger.info(f"  Max Reserved: {stats['max_reserved_gb']:.2f} GB")


# Example usage
if __name__ == "__main__":
    print("GPU Utilities Test")
    print("=" * 50)

    # Test device detection
    device = get_device()
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # Test memory tracker
        print("\nTesting GPU memory tracker:")
        tracker = GPUMemoryTracker(device)

        # Allocate some memory
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = x @ y

        tracker.print_memory_stats()

        # Test multi-GPU if available
        if torch.cuda.device_count() > 1:
            print("\nTesting DataParallel:")

            # Create simple model
            model = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )

            # Setup multi-GPU
            multi_gpu = MultiGPUTrainer(model, mode='dataparallel')

            print("DataParallel model created successfully")

    print("\nGPU utilities test completed!")
