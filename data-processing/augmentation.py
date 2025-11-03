"""
Data Augmentation Module

Comprehensive data augmentation techniques for:
- Image data (rotation, flipping, cropping, color jittering, noise)
- Text data (synonym replacement, back-translation, random insertion/deletion)
- Tabular data (noise injection, SMOTE-like, feature mixing)

Author: ML Framework Team
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Union
import random


# ============================================================================
# IMAGE AUGMENTATION
# ============================================================================

class ImageAugmenter:
    """
    Image augmentation with various transformations.

    Supports:
    - Geometric: rotation, flipping, cropping, scaling
    - Color: brightness, contrast, saturation, hue
    - Noise: Gaussian, salt & pepper
    - Advanced: cutout, mixup
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augmenter.

        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def rotate(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
        angle_range: Tuple[float, float] = (-30, 30)
    ) -> np.ndarray:
        """
        Rotate image by angle.

        Parameters:
        -----------
        image : np.ndarray
            Input image (H, W, C) or (H, W).
        angle : float, optional
            Rotation angle in degrees. If None, random angle from range.
        angle_range : tuple
            Range for random angle selection.

        Returns:
        --------
        rotated : np.ndarray
            Rotated image.
        """
        if angle is None:
            angle = np.random.uniform(*angle_range)

        # Simple rotation using affine transformation
        # For production, use cv2.warpAffine or similar
        angle_rad = np.deg2rad(angle)
        h, w = image.shape[:2]

        # Center of rotation
        cx, cy = w // 2, h // 2

        # Rotation matrix
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)

        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w]

        # Translate to origin
        x_centered = x - cx
        y_centered = y - cy

        # Rotate
        x_rot = cos_val * x_centered - sin_val * y_centered + cx
        y_rot = sin_val * x_centered + cos_val * y_centered + cy

        # Clip coordinates
        x_rot = np.clip(x_rot, 0, w - 1).astype(int)
        y_rot = np.clip(y_rot, 0, h - 1).astype(int)

        # Apply rotation
        if len(image.shape) == 3:
            rotated = image[y_rot, x_rot, :]
        else:
            rotated = image[y_rot, x_rot]

        return rotated

    def flip(
        self,
        image: np.ndarray,
        mode: str = 'horizontal'
    ) -> np.ndarray:
        """
        Flip image horizontally or vertically.

        Parameters:
        -----------
        image : np.ndarray
            Input image.
        mode : str
            'horizontal', 'vertical', or 'both'.

        Returns:
        --------
        flipped : np.ndarray
            Flipped image.
        """
        if mode == 'horizontal':
            return np.fliplr(image)
        elif mode == 'vertical':
            return np.flipud(image)
        elif mode == 'both':
            return np.flipud(np.fliplr(image))
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def crop(
        self,
        image: np.ndarray,
        crop_size: Optional[Tuple[int, int]] = None,
        crop_fraction: float = 0.8
    ) -> np.ndarray:
        """
        Random crop of image.

        Parameters:
        -----------
        image : np.ndarray
            Input image (H, W, C) or (H, W).
        crop_size : tuple, optional
            (height, width) of crop. If None, use crop_fraction.
        crop_fraction : float
            Fraction of image to crop (if crop_size is None).

        Returns:
        --------
        cropped : np.ndarray
            Cropped image.
        """
        h, w = image.shape[:2]

        if crop_size is None:
            crop_h = int(h * crop_fraction)
            crop_w = int(w * crop_fraction)
        else:
            crop_h, crop_w = crop_size

        # Random position
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)

        # Crop
        if len(image.shape) == 3:
            return image[top:top+crop_h, left:left+crop_w, :]
        else:
            return image[top:top+crop_h, left:left+crop_w]

    def adjust_brightness(
        self,
        image: np.ndarray,
        factor: Optional[float] = None,
        factor_range: Tuple[float, float] = (0.7, 1.3)
    ) -> np.ndarray:
        """
        Adjust image brightness.

        Parameters:
        -----------
        image : np.ndarray
            Input image (values in [0, 1] or [0, 255]).
        factor : float, optional
            Brightness factor. If None, random from range.
        factor_range : tuple
            Range for random factor.

        Returns:
        --------
        adjusted : np.ndarray
            Brightness-adjusted image.
        """
        if factor is None:
            factor = np.random.uniform(*factor_range)

        adjusted = image * factor
        return np.clip(adjusted, 0, image.max())

    def adjust_contrast(
        self,
        image: np.ndarray,
        factor: Optional[float] = None,
        factor_range: Tuple[float, float] = (0.7, 1.3)
    ) -> np.ndarray:
        """
        Adjust image contrast.

        Parameters:
        -----------
        image : np.ndarray
            Input image.
        factor : float, optional
            Contrast factor. If None, random from range.
        factor_range : tuple
            Range for random factor.

        Returns:
        --------
        adjusted : np.ndarray
            Contrast-adjusted image.
        """
        if factor is None:
            factor = np.random.uniform(*factor_range)

        # Adjust contrast relative to mean
        mean = image.mean()
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, image.max())

    def add_gaussian_noise(
        self,
        image: np.ndarray,
        mean: float = 0.0,
        std: float = 0.1
    ) -> np.ndarray:
        """
        Add Gaussian noise to image.

        Parameters:
        -----------
        image : np.ndarray
            Input image.
        mean : float
            Mean of Gaussian noise.
        std : float
            Standard deviation of Gaussian noise.

        Returns:
        --------
        noisy : np.ndarray
            Noisy image.
        """
        noise = np.random.normal(mean, std, image.shape)
        noisy = image + noise * image.max()
        return np.clip(noisy, 0, image.max())

    def add_salt_pepper_noise(
        self,
        image: np.ndarray,
        salt_prob: float = 0.01,
        pepper_prob: float = 0.01
    ) -> np.ndarray:
        """
        Add salt and pepper noise.

        Parameters:
        -----------
        image : np.ndarray
            Input image.
        salt_prob : float
            Probability of salt noise.
        pepper_prob : float
            Probability of pepper noise.

        Returns:
        --------
        noisy : np.ndarray
            Noisy image.
        """
        noisy = image.copy()

        # Salt noise (white)
        salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
        if len(image.shape) == 3:
            salt_mask = salt_mask[:, :, np.newaxis]
        noisy[salt_mask] = image.max()

        # Pepper noise (black)
        pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
        if len(image.shape) == 3:
            pepper_mask = pepper_mask[:, :, np.newaxis]
        noisy[pepper_mask] = 0

        return noisy

    def cutout(
        self,
        image: np.ndarray,
        n_holes: int = 1,
        hole_size: Optional[Tuple[int, int]] = None,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Apply cutout augmentation (random rectangular masks).

        Parameters:
        -----------
        image : np.ndarray
            Input image.
        n_holes : int
            Number of cutout holes.
        hole_size : tuple, optional
            (height, width) of holes. If None, use 1/8 of image size.
        fill_value : float
            Value to fill holes with.

        Returns:
        --------
        cutout_image : np.ndarray
            Image with cutout.
        """
        h, w = image.shape[:2]

        if hole_size is None:
            hole_h, hole_w = h // 8, w // 8
        else:
            hole_h, hole_w = hole_size

        cutout_image = image.copy()

        for _ in range(n_holes):
            # Random position
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)

            # Calculate hole boundaries
            y1 = np.clip(y - hole_h // 2, 0, h)
            y2 = np.clip(y + hole_h // 2, 0, h)
            x1 = np.clip(x - hole_w // 2, 0, w)
            x2 = np.clip(x + hole_w // 2, 0, w)

            # Apply cutout
            cutout_image[y1:y2, x1:x2] = fill_value

        return cutout_image

    def mixup(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, float]:
        """
        Apply mixup augmentation (linear interpolation of two images).

        Parameters:
        -----------
        image1 : np.ndarray
            First image.
        image2 : np.ndarray
            Second image.
        alpha : float
            Beta distribution parameter for mixing coefficient.

        Returns:
        --------
        mixed : np.ndarray
            Mixed image.
        lam : float
            Mixing coefficient (for label mixing).
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        mixed = lam * image1 + (1 - lam) * image2
        return mixed, lam

    def random_augment(
        self,
        image: np.ndarray,
        n_ops: int = 2,
        magnitude: float = 0.5
    ) -> np.ndarray:
        """
        Apply random augmentation operations.

        Parameters:
        -----------
        image : np.ndarray
            Input image.
        n_ops : int
            Number of operations to apply.
        magnitude : float
            Magnitude of augmentations (0 to 1).

        Returns:
        --------
        augmented : np.ndarray
            Augmented image.
        """
        augmented = image.copy()

        ops = [
            lambda x: self.rotate(x, angle_range=(-30*magnitude, 30*magnitude)),
            lambda x: self.flip(x, mode='horizontal'),
            lambda x: self.adjust_brightness(x, factor_range=(1-0.3*magnitude, 1+0.3*magnitude)),
            lambda x: self.adjust_contrast(x, factor_range=(1-0.3*magnitude, 1+0.3*magnitude)),
            lambda x: self.add_gaussian_noise(x, std=0.1*magnitude),
        ]

        selected_ops = random.sample(ops, min(n_ops, len(ops)))

        for op in selected_ops:
            augmented = op(augmented)

        return augmented


# ============================================================================
# TEXT AUGMENTATION
# ============================================================================

class TextAugmenter:
    """
    Text augmentation with various techniques.

    Supports:
    - Synonym replacement
    - Random insertion
    - Random swap
    - Random deletion
    - Back-translation (simulated)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augmenter.

        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        # Simple synonym dictionary (in production, use WordNet or similar)
        self.synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['poor', 'terrible', 'awful', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased'],
            'sad': ['unhappy', 'sorrowful', 'dejected', 'gloomy'],
        }

    def synonym_replacement(
        self,
        text: str,
        n_replacements: int = 2
    ) -> str:
        """
        Replace n random words with synonyms.

        Parameters:
        -----------
        text : str
            Input text.
        n_replacements : int
            Number of words to replace.

        Returns:
        --------
        augmented : str
            Augmented text.
        """
        words = text.split()

        # Find replaceable words
        replaceable_indices = [
            i for i, word in enumerate(words)
            if word.lower() in self.synonyms
        ]

        if not replaceable_indices:
            return text

        # Random selection
        n_replacements = min(n_replacements, len(replaceable_indices))
        indices_to_replace = random.sample(replaceable_indices, n_replacements)

        # Replace
        for idx in indices_to_replace:
            word = words[idx].lower()
            synonym = random.choice(self.synonyms[word])
            # Preserve case
            if words[idx][0].isupper():
                synonym = synonym.capitalize()
            words[idx] = synonym

        return ' '.join(words)

    def random_insertion(
        self,
        text: str,
        n_insertions: int = 2
    ) -> str:
        """
        Insert n random words from the text.

        Parameters:
        -----------
        text : str
            Input text.
        n_insertions : int
            Number of insertions.

        Returns:
        --------
        augmented : str
            Augmented text.
        """
        words = text.split()

        if len(words) < 2:
            return text

        for _ in range(n_insertions):
            # Random word
            word_to_insert = random.choice(words)
            # Random position
            position = random.randint(0, len(words))
            words.insert(position, word_to_insert)

        return ' '.join(words)

    def random_swap(
        self,
        text: str,
        n_swaps: int = 2
    ) -> str:
        """
        Swap positions of n random word pairs.

        Parameters:
        -----------
        text : str
            Input text.
        n_swaps : int
            Number of swaps.

        Returns:
        --------
        augmented : str
            Augmented text.
        """
        words = text.split()

        if len(words) < 2:
            return text

        for _ in range(n_swaps):
            # Random positions
            idx1, idx2 = random.sample(range(len(words)), 2)
            # Swap
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    def random_deletion(
        self,
        text: str,
        p: float = 0.1
    ) -> str:
        """
        Delete each word with probability p.

        Parameters:
        -----------
        text : str
            Input text.
        p : float
            Deletion probability.

        Returns:
        --------
        augmented : str
            Augmented text.
        """
        words = text.split()

        if len(words) == 1:
            return text

        # Keep words with probability 1-p
        kept_words = [word for word in words if random.random() > p]

        # Ensure at least one word remains
        if not kept_words:
            return random.choice(words)

        return ' '.join(kept_words)

    def back_translation(
        self,
        text: str
    ) -> str:
        """
        Simulate back-translation by applying multiple transformations.

        In production, use actual translation APIs (e.g., en -> de -> en).

        Parameters:
        -----------
        text : str
            Input text.

        Returns:
        --------
        augmented : str
            Augmented text.
        """
        # Simulate back-translation with multiple operations
        augmented = text
        augmented = self.synonym_replacement(augmented, n_replacements=2)
        augmented = self.random_swap(augmented, n_swaps=1)
        return augmented

    def random_augment(
        self,
        text: str,
        n_ops: int = 2
    ) -> str:
        """
        Apply n random augmentation operations.

        Parameters:
        -----------
        text : str
            Input text.
        n_ops : int
            Number of operations.

        Returns:
        --------
        augmented : str
            Augmented text.
        """
        ops = [
            lambda x: self.synonym_replacement(x, n_replacements=1),
            lambda x: self.random_insertion(x, n_insertions=1),
            lambda x: self.random_swap(x, n_swaps=1),
            lambda x: self.random_deletion(x, p=0.1),
        ]

        augmented = text
        selected_ops = random.sample(ops, min(n_ops, len(ops)))

        for op in selected_ops:
            augmented = op(augmented)

        return augmented


# ============================================================================
# TABULAR AUGMENTATION
# ============================================================================

class TabularAugmenter:
    """
    Tabular data augmentation.

    Supports:
    - Gaussian noise injection
    - Feature mixing (mixup)
    - SMOTE-like interpolation
    - Random feature perturbation
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize augmenter.

        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def add_noise(
        self,
        X: np.ndarray,
        noise_level: float = 0.05
    ) -> np.ndarray:
        """
        Add Gaussian noise to features.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        noise_level : float
            Noise level as fraction of feature std.

        Returns:
        --------
        X_noisy : np.ndarray
            Noisy feature matrix.
        """
        std = X.std(axis=0)
        noise = np.random.normal(0, std * noise_level, X.shape)
        return X + noise

    def mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation to tabular data.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Labels (n_samples,).
        alpha : float
            Beta distribution parameter.

        Returns:
        --------
        X_mixed : np.ndarray
            Mixed feature matrix.
        y_mixed : np.ndarray
            Mixed labels.
        """
        n_samples = X.shape[0]

        # Random pairing
        indices = np.random.permutation(n_samples)
        X_paired = X[indices]
        y_paired = y[indices]

        # Mixing coefficients
        if alpha > 0:
            lam = np.random.beta(alpha, alpha, n_samples)
        else:
            lam = np.ones(n_samples)

        # Mix
        lam = lam.reshape(-1, 1)
        X_mixed = lam * X + (1 - lam) * X_paired

        # Mix labels (for regression) or keep original (for classification)
        if y.dtype in [np.float32, np.float64]:
            y_mixed = lam.ravel() * y + (1 - lam.ravel()) * y_paired
        else:
            y_mixed = y.copy()

        return X_mixed, y_mixed

    def smote_interpolate(
        self,
        X: np.ndarray,
        n_samples: int = 100,
        k_neighbors: int = 5
    ) -> np.ndarray:
        """
        Generate synthetic samples using SMOTE-like interpolation.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        n_samples : int
            Number of synthetic samples to generate.
        k_neighbors : int
            Number of nearest neighbors to consider.

        Returns:
        --------
        X_synthetic : np.ndarray
            Synthetic samples.
        """
        n_samples_orig = X.shape[0]
        synthetic = []

        for _ in range(n_samples):
            # Random sample
            idx = np.random.randint(0, n_samples_orig)
            sample = X[idx]

            # Find k nearest neighbors (simple Euclidean distance)
            distances = np.linalg.norm(X - sample, axis=1)
            neighbor_indices = np.argsort(distances)[1:k_neighbors+1]

            # Random neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = X[neighbor_idx]

            # Interpolate
            alpha = np.random.rand()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic.append(synthetic_sample)

        return np.array(synthetic)

    def random_feature_perturbation(
        self,
        X: np.ndarray,
        n_features_perturb: int = 2,
        perturbation_strength: float = 0.1
    ) -> np.ndarray:
        """
        Randomly perturb n features for each sample.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        n_features_perturb : int
            Number of features to perturb per sample.
        perturbation_strength : float
            Perturbation strength as fraction of feature range.

        Returns:
        --------
        X_perturbed : np.ndarray
            Perturbed feature matrix.
        """
        X_perturbed = X.copy()
        n_samples, n_features = X.shape

        # Feature ranges
        feature_ranges = X.max(axis=0) - X.min(axis=0)

        for i in range(n_samples):
            # Random features to perturb
            features_to_perturb = np.random.choice(
                n_features,
                size=min(n_features_perturb, n_features),
                replace=False
            )

            # Perturb
            for j in features_to_perturb:
                perturbation = np.random.uniform(
                    -feature_ranges[j] * perturbation_strength,
                    feature_ranges[j] * perturbation_strength
                )
                X_perturbed[i, j] += perturbation

        return X_perturbed

    def augment_batch(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        augmentation_factor: int = 2,
        methods: List[str] = ['noise', 'mixup', 'smote']
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment batch with multiple methods.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray, optional
            Labels.
        augmentation_factor : int
            How many times to augment the data.
        methods : list
            Augmentation methods to use.

        Returns:
        --------
        X_augmented : np.ndarray
            Augmented features.
        y_augmented : np.ndarray, optional
            Augmented labels (if y provided).
        """
        X_augmented = [X]
        y_augmented = [y] if y is not None else None

        for _ in range(augmentation_factor - 1):
            method = random.choice(methods)

            if method == 'noise':
                X_aug = self.add_noise(X)
                y_aug = y.copy() if y is not None else None

            elif method == 'mixup' and y is not None:
                X_aug, y_aug = self.mixup(X, y)

            elif method == 'smote':
                X_aug = self.smote_interpolate(X, n_samples=X.shape[0])
                y_aug = y.copy() if y is not None else None

            elif method == 'perturbation':
                X_aug = self.random_feature_perturbation(X)
                y_aug = y.copy() if y is not None else None

            else:
                continue

            X_augmented.append(X_aug)
            if y_augmented is not None:
                y_augmented.append(y_aug)

        X_final = np.vstack(X_augmented)
        y_final = np.concatenate(y_augmented) if y_augmented else None

        return X_final, y_final


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA AUGMENTATION EXAMPLES")
    print("=" * 70)

    # Example 1: Image Augmentation
    print("\n1. Image Augmentation")
    print("-" * 70)

    # Create sample image
    image = np.random.rand(64, 64, 3)

    augmenter = ImageAugmenter(seed=42)

    # Various augmentations
    rotated = augmenter.rotate(image, angle=30)
    flipped = augmenter.flip(image, mode='horizontal')
    cropped = augmenter.crop(image, crop_fraction=0.8)
    bright = augmenter.adjust_brightness(image, factor=1.2)
    noisy = augmenter.add_gaussian_noise(image, std=0.1)
    cutout_img = augmenter.cutout(image, n_holes=2)

    # Random augmentation
    random_aug = augmenter.random_augment(image, n_ops=3, magnitude=0.7)

    print(f"Original shape: {image.shape}")
    print(f"Rotated shape:  {rotated.shape}")
    print(f"Cropped shape:  {cropped.shape}")
    print(f"Applied multiple augmentations successfully!")

    # Example 2: Text Augmentation
    print("\n2. Text Augmentation")
    print("-" * 70)

    text = "This is a good example of fast text processing with big data"

    text_augmenter = TextAugmenter(seed=42)

    synonym_text = text_augmenter.synonym_replacement(text, n_replacements=2)
    inserted_text = text_augmenter.random_insertion(text, n_insertions=2)
    swapped_text = text_augmenter.random_swap(text, n_swaps=2)
    deleted_text = text_augmenter.random_deletion(text, p=0.1)
    random_aug_text = text_augmenter.random_augment(text, n_ops=2)

    print(f"Original:  {text}")
    print(f"Synonym:   {synonym_text}")
    print(f"Inserted:  {inserted_text}")
    print(f"Swapped:   {swapped_text}")
    print(f"Deleted:   {deleted_text}")
    print(f"Random:    {random_aug_text}")

    # Example 3: Tabular Augmentation
    print("\n3. Tabular Augmentation")
    print("-" * 70)

    # Sample data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    tabular_augmenter = TabularAugmenter(seed=42)

    # Noise injection
    X_noisy = tabular_augmenter.add_noise(X, noise_level=0.05)

    # Mixup
    X_mixed, y_mixed = tabular_augmenter.mixup(X, y, alpha=0.2)

    # SMOTE interpolation
    X_synthetic = tabular_augmenter.smote_interpolate(X, n_samples=50, k_neighbors=5)

    # Feature perturbation
    X_perturbed = tabular_augmenter.random_feature_perturbation(X, n_features_perturb=2)

    # Batch augmentation
    X_aug, y_aug = tabular_augmenter.augment_batch(
        X, y,
        augmentation_factor=3,
        methods=['noise', 'mixup', 'smote']
    )

    print(f"Original data:     {X.shape}")
    print(f"Noisy data:        {X_noisy.shape}")
    print(f"Mixed data:        {X_mixed.shape}")
    print(f"Synthetic data:    {X_synthetic.shape}")
    print(f"Perturbed data:    {X_perturbed.shape}")
    print(f"Batch augmented:   {X_aug.shape}")

    print("\n" + "=" * 70)
    print("All augmentation examples completed successfully!")
    print("=" * 70)
