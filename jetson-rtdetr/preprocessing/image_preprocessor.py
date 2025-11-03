"""
Image Preprocessing Pipeline for RT-DETR
Handles image loading, resizing, normalization, and batching
"""

import logging
from typing import Tuple, Optional, List, Union
from pathlib import Path
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Install with: pip install opencv-python")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Install with: pip install Pillow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocessing pipeline for RT-DETR inference
    Handles image loading, resizing, normalization
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        swap_rb: bool = True,  # BGR to RGB
        normalize: bool = True,
        keep_ratio: bool = True,  # Letterbox padding
        backend: str = "cv2"  # cv2 or pil
    ):
        """
        Initialize preprocessor

        Args:
            input_size: Target input size (height, width)
            mean: Mean values for normalization (RGB order)
            std: Std values for normalization (RGB order)
            swap_rb: Swap R and B channels (BGR to RGB)
            normalize: Apply normalization
            keep_ratio: Keep aspect ratio with letterbox padding
            backend: Image processing backend (cv2 or pil)
        """
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.swap_rb = swap_rb
        self.normalize = normalize
        self.keep_ratio = keep_ratio
        self.backend = backend.lower()

        if self.backend == "cv2" and not CV2_AVAILABLE:
            raise ImportError("OpenCV not available")
        if self.backend == "pil" and not PIL_AVAILABLE:
            raise ImportError("PIL not available")

        logger.info(f"Initialized ImagePreprocessor")
        logger.info(f"  Input size: {self.input_size}")
        logger.info(f"  Mean: {mean}")
        logger.info(f"  Std: {std}")
        logger.info(f"  Backend: {self.backend}")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (H, W, C) in RGB format
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.backend == "cv2":
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            if self.swap_rb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # PIL
            img = Image.open(image_path).convert('RGB')
            img = np.array(img)

        return img

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Resize image with optional letterbox padding

        Args:
            image: Input image (H, W, C)
            target_size: Target size (H, W), uses self.input_size if None

        Returns:
            Tuple of (resized image, metadata dict)
        """
        if target_size is None:
            target_size = self.input_size

        h, w = image.shape[:2]
        target_h, target_w = target_size

        metadata = {
            'original_size': (h, w),
            'target_size': target_size,
            'scale': None,
            'pad': (0, 0, 0, 0)  # top, bottom, left, right
        }

        if self.keep_ratio:
            # Calculate scale to fit image in target size
            scale = min(target_h / h, target_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)

            # Resize image
            if self.backend == "cv2":
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:  # PIL
                resized = np.array(
                    Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
                )

            # Calculate padding
            pad_h = target_h - new_h
            pad_w = target_w - new_w
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            # Apply letterbox padding
            if self.backend == "cv2":
                resized = cv2.copyMakeBorder(
                    resized, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
            else:  # PIL
                padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
                padded[top:top+new_h, left:left+new_w] = resized
                resized = padded

            metadata['scale'] = scale
            metadata['pad'] = (top, bottom, left, right)

        else:
            # Direct resize without keeping ratio
            if self.backend == "cv2":
                resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:  # PIL
                resized = np.array(
                    Image.fromarray(image).resize((target_w, target_h), Image.BILINEAR)
                )

            metadata['scale'] = (target_h / h, target_w / w)

        return resized, metadata

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image

        Args:
            image: Input image (H, W, C) in range [0, 255]

        Returns:
            Normalized image in range [0, 1] or standardized
        """
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        if self.normalize:
            # Standardize with mean and std
            image = (image - self.mean) / self.std

        return image

    def to_tensor(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to tensor format (C, H, W)

        Args:
            image: Input image (H, W, C)

        Returns:
            Tensor (C, H, W)
        """
        # Transpose from (H, W, C) to (C, H, W)
        tensor = np.transpose(image, (2, 0, 1))
        return tensor

    def preprocess(
        self,
        image: Union[str, np.ndarray],
        return_metadata: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Complete preprocessing pipeline

        Args:
            image: Image path or numpy array
            return_metadata: Return preprocessing metadata

        Returns:
            Preprocessed tensor (C, H, W) or tuple with metadata
        """
        # Load image if path provided
        if isinstance(image, str):
            image = self.load_image(image)

        # Resize
        resized, metadata = self.resize_image(image)

        # Normalize
        normalized = self.normalize_image(resized)

        # Convert to tensor
        tensor = self.to_tensor(normalized)

        if return_metadata:
            return tensor, metadata
        return tensor

    def preprocess_batch(
        self,
        images: List[Union[str, np.ndarray]],
        return_metadata: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[dict]]]:
        """
        Preprocess batch of images

        Args:
            images: List of image paths or numpy arrays
            return_metadata: Return preprocessing metadata

        Returns:
            Batch tensor (N, C, H, W) or tuple with metadata list
        """
        tensors = []
        metadata_list = []

        for image in images:
            if return_metadata:
                tensor, metadata = self.preprocess(image, return_metadata=True)
                metadata_list.append(metadata)
            else:
                tensor = self.preprocess(image, return_metadata=False)

            tensors.append(tensor)

        # Stack to batch
        batch = np.stack(tensors, axis=0)

        if return_metadata:
            return batch, metadata_list
        return batch

    def denormalize_image(self, tensor: np.ndarray) -> np.ndarray:
        """
        Denormalize tensor back to image

        Args:
            tensor: Normalized tensor (C, H, W) or (H, W, C)

        Returns:
            Image in range [0, 255]
        """
        # Handle different tensor formats
        if tensor.shape[0] == 3:  # (C, H, W)
            image = np.transpose(tensor, (1, 2, 0))
        else:  # (H, W, C)
            image = tensor.copy()

        if self.normalize:
            # Denormalize
            image = (image * self.std) + self.mean

        # Scale to [0, 255]
        image = (image * 255.0).clip(0, 255).astype(np.uint8)

        return image


class VideoPreprocessor:
    """
    Video preprocessing pipeline
    Handles video stream processing with frame batching
    """

    def __init__(
        self,
        video_path: str,
        preprocessor: ImagePreprocessor,
        batch_size: int = 1,
        skip_frames: int = 0
    ):
        """
        Initialize video preprocessor

        Args:
            video_path: Path to video file or camera index (0, 1, etc.)
            preprocessor: Image preprocessor instance
            batch_size: Number of frames per batch
            skip_frames: Number of frames to skip between reads
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for video processing")

        self.video_path = video_path
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.skip_frames = skip_frames

        # Open video capture
        if isinstance(video_path, int) or video_path.isdigit():
            self.cap = cv2.VideoCapture(int(video_path))
        else:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Opened video: {video_path}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps}")
        logger.info(f"  Frame count: {self.frame_count}")

    def __iter__(self):
        """Iterate over video frames"""
        return self

    def __next__(self) -> Tuple[np.ndarray, List[np.ndarray], List[dict]]:
        """
        Get next batch of frames

        Returns:
            Tuple of (preprocessed batch, original frames, metadata list)
        """
        frames = []
        originals = []
        metadata_list = []

        for _ in range(self.batch_size):
            # Read frame
            for _ in range(self.skip_frames + 1):
                ret, frame = self.cap.read()
                if not ret:
                    if len(frames) == 0:
                        raise StopIteration
                    break

            if not ret:
                break

            # Convert BGR to RGB if needed
            if self.preprocessor.swap_rb:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # Preprocess frame
            tensor, metadata = self.preprocessor.preprocess(frame_rgb, return_metadata=True)

            frames.append(tensor)
            originals.append(frame)  # Keep original in BGR for display
            metadata_list.append(metadata)

        if len(frames) == 0:
            raise StopIteration

        # Stack to batch
        batch = np.stack(frames, axis=0)

        return batch, originals, metadata_list

    def release(self):
        """Release video capture"""
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        """Cleanup"""
        self.release()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test image preprocessing")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--size", type=int, default=640, help="Input size")
    parser.add_argument("--output", help="Output path for visualization")

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = ImagePreprocessor(
        input_size=(args.size, args.size),
        keep_ratio=True,
        backend="cv2"
    )

    # Preprocess image
    tensor, metadata = preprocessor.preprocess(args.image, return_metadata=True)

    print(f"Input tensor shape: {tensor.shape}")
    print(f"Metadata: {metadata}")

    # Visualize if output specified
    if args.output:
        denorm = preprocessor.denormalize_image(tensor)
        cv2.imwrite(args.output, cv2.cvtColor(denorm, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to: {args.output}")
