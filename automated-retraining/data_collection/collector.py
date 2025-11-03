"""
Data Collection Service
Collects inference data from production for retraining
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import threading
import queue
import hashlib

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSample:
    """Single data sample for collection"""

    def __init__(
        self,
        image_path: str,
        prediction: Dict[str, Any],
        confidence: float,
        metadata: Optional[Dict] = None,
        timestamp: Optional[float] = None
    ):
        self.image_path = image_path
        self.prediction = prediction
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.sample_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique sample ID"""
        data = f"{self.image_path}{self.timestamp}".encode()
        return hashlib.md5(data).hexdigest()

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'sample_id': self.sample_id,
            'image_path': self.image_path,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DataSample':
        """Create from dictionary"""
        return cls(
            image_path=data['image_path'],
            prediction=data['prediction'],
            confidence=data['confidence'],
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp')
        )


class DataCollectionQueue:
    """Thread-safe queue for data collection"""

    def __init__(
        self,
        max_size: int = 10000,
        flush_interval: int = 60,
        flush_threshold: int = 100
    ):
        self.queue = queue.Queue(maxsize=max_size)
        self.flush_interval = flush_interval
        self.flush_threshold = flush_threshold
        self.stats = {
            'collected': 0,
            'flushed': 0,
            'dropped': 0
        }
        self._lock = threading.Lock()

    def put(self, sample: DataSample, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add sample to queue"""
        try:
            self.queue.put(sample, block=block, timeout=timeout)
            with self._lock:
                self.stats['collected'] += 1
            return True
        except queue.Full:
            with self._lock:
                self.stats['dropped'] += 1
            logger.warning(f"Queue full, dropped sample: {sample.sample_id}")
            return False

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[DataSample]:
        """Get sample from queue"""
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_batch(self, batch_size: int, timeout: float = 1.0) -> List[DataSample]:
        """Get batch of samples"""
        batch = []
        deadline = time.time() + timeout

        while len(batch) < batch_size and time.time() < deadline:
            remaining = deadline - time.time()
            sample = self.get(block=True, timeout=max(0.1, remaining))
            if sample:
                batch.append(sample)
            else:
                break

        return batch

    def size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()

    def is_full(self) -> bool:
        """Check if queue is full"""
        return self.queue.full()


class DataCollectionService:
    """
    Data collection service for production inference data
    Supports multiple storage backends and collection strategies
    """

    def __init__(
        self,
        storage_path: str,
        collection_strategy: str = "uncertainty",  # uncertainty, random, all
        confidence_threshold: float = 0.7,
        sample_rate: float = 0.1,
        redis_url: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        enable_deduplication: bool = True
    ):
        """
        Initialize data collection service

        Args:
            storage_path: Local storage path
            collection_strategy: Strategy for collecting samples
            confidence_threshold: Confidence threshold for uncertainty sampling
            sample_rate: Sample rate for random sampling
            redis_url: Redis URL for distributed queue
            s3_bucket: S3 bucket for cloud storage
            enable_deduplication: Remove duplicate samples
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.collection_strategy = collection_strategy
        self.confidence_threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.enable_deduplication = enable_deduplication

        # Initialize queue
        self.queue = DataCollectionQueue()

        # Initialize storage backends
        self.redis_client = self._init_redis(redis_url) if redis_url and REDIS_AVAILABLE else None
        self.s3_client = self._init_s3(s3_bucket) if s3_bucket and S3_AVAILABLE else None
        self.s3_bucket = s3_bucket

        # Deduplication
        self.seen_hashes = set()

        # Background worker
        self.running = False
        self.worker_thread = None

        logger.info("Data collection service initialized")
        logger.info(f"  Strategy: {collection_strategy}")
        logger.info(f"  Storage: {storage_path}")
        logger.info(f"  Redis: {'enabled' if self.redis_client else 'disabled'}")
        logger.info(f"  S3: {'enabled' if self.s3_client else 'disabled'}")

    def _init_redis(self, redis_url: str):
        """Initialize Redis client"""
        try:
            client = redis.from_url(redis_url)
            client.ping()
            logger.info("Redis connected")
            return client
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return None

    def _init_s3(self, bucket: str):
        """Initialize S3 client"""
        try:
            client = boto3.client('s3')
            client.head_bucket(Bucket=bucket)
            logger.info(f"S3 bucket connected: {bucket}")
            return client
        except Exception as e:
            logger.error(f"S3 connection failed: {e}")
            return None

    def should_collect(self, confidence: float) -> bool:
        """
        Determine if sample should be collected based on strategy

        Args:
            confidence: Prediction confidence

        Returns:
            True if sample should be collected
        """
        if self.collection_strategy == "all":
            return True

        elif self.collection_strategy == "uncertainty":
            # Collect low-confidence samples
            return confidence < self.confidence_threshold

        elif self.collection_strategy == "random":
            # Random sampling
            import random
            return random.random() < self.sample_rate

        elif self.collection_strategy == "boundary":
            # Collect samples near decision boundary
            return 0.4 < confidence < 0.6

        else:
            logger.warning(f"Unknown strategy: {self.collection_strategy}")
            return False

    def collect(
        self,
        image_path: str,
        prediction: Dict[str, Any],
        confidence: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Collect a data sample

        Args:
            image_path: Path to image
            prediction: Model prediction
            confidence: Prediction confidence
            metadata: Additional metadata

        Returns:
            True if collected
        """
        # Check collection criteria
        if not self.should_collect(confidence):
            return False

        # Create sample
        sample = DataSample(
            image_path=image_path,
            prediction=prediction,
            confidence=confidence,
            metadata=metadata
        )

        # Deduplication
        if self.enable_deduplication:
            image_hash = self._hash_file(image_path)
            if image_hash in self.seen_hashes:
                return False
            self.seen_hashes.add(image_hash)

        # Add to queue
        success = self.queue.put(sample, block=False)

        # Optionally publish to Redis
        if self.redis_client and success:
            try:
                self.redis_client.lpush('collection_queue', json.dumps(sample.to_dict()))
            except Exception as e:
                logger.error(f"Redis publish failed: {e}")

        return success

    def _hash_file(self, file_path: str) -> str:
        """Compute file hash for deduplication"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"File hash failed: {e}")
            return ""

    def start_worker(self):
        """Start background worker thread"""
        if self.running:
            logger.warning("Worker already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Background worker started")

    def stop_worker(self):
        """Stop background worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Background worker stopped")

    def _worker_loop(self):
        """Background worker loop for flushing data"""
        while self.running:
            try:
                # Get batch
                batch = self.queue.get_batch(batch_size=100, timeout=1.0)

                if batch:
                    self._flush_batch(batch)

                # Sleep briefly
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

    def _flush_batch(self, batch: List[DataSample]):
        """Flush batch to storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.storage_path / f"batch_{timestamp}.json"

        try:
            # Save locally
            data = [sample.to_dict() for sample in batch]
            with open(batch_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Flushed {len(batch)} samples to {batch_file}")

            # Upload to S3 if available
            if self.s3_client:
                try:
                    s3_key = f"collected_data/{timestamp}/batch.json"
                    self.s3_client.upload_file(
                        str(batch_file),
                        self.s3_bucket,
                        s3_key
                    )
                    logger.info(f"Uploaded to S3: {s3_key}")
                except Exception as e:
                    logger.error(f"S3 upload failed: {e}")

            # Update stats
            with self.queue._lock:
                self.queue.stats['flushed'] += len(batch)

        except Exception as e:
            logger.error(f"Flush failed: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """Get collection statistics"""
        with self.queue._lock:
            stats = self.queue.stats.copy()

        stats['queue_size'] = self.queue.size()
        stats['dedup_cache_size'] = len(self.seen_hashes)
        return stats

    def __enter__(self):
        """Context manager entry"""
        self.start_worker()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_worker()


class DataCollectionMonitor:
    """Monitor data collection metrics"""

    def __init__(self, service: DataCollectionService, check_interval: int = 60):
        self.service = service
        self.check_interval = check_interval
        self.history = deque(maxlen=1000)
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Collection monitor started")

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def _monitor_loop(self):
        """Monitor loop"""
        while self.running:
            try:
                stats = self.service.get_stats()
                stats['timestamp'] = time.time()
                self.history.append(stats)

                # Log stats
                logger.info(
                    f"Collection stats - "
                    f"Collected: {stats['collected']}, "
                    f"Flushed: {stats['flushed']}, "
                    f"Dropped: {stats['dropped']}, "
                    f"Queue: {stats['queue_size']}"
                )

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def get_history(self) -> List[dict]:
        """Get collection history"""
        return list(self.history)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Data Collection Service")
    parser.add_argument("--storage", default="./collected_data", help="Storage path")
    parser.add_argument("--strategy", default="uncertainty", choices=["all", "uncertainty", "random", "boundary"])
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--sample-rate", type=float, default=0.1, help="Sample rate")
    parser.add_argument("--redis", help="Redis URL")
    parser.add_argument("--s3-bucket", help="S3 bucket name")

    args = parser.parse_args()

    # Create service
    service = DataCollectionService(
        storage_path=args.storage,
        collection_strategy=args.strategy,
        confidence_threshold=args.threshold,
        sample_rate=args.sample_rate,
        redis_url=args.redis,
        s3_bucket=args.s3_bucket
    )

    # Start service
    with service:
        logger.info("Service running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
