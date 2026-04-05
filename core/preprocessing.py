"""Image preprocessing pipeline for signature verification."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Resize, normalize, and optionally binarize signature images.

    All methods are stateless; create one instance and reuse it freely.

    Args:
        target_size: (width, height) to resize images to.
    """

    def __init__(self, target_size: tuple = (150, 150)):
        self.target_size = target_size  # (width, height) for cv2.resize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, image_path: str) -> np.ndarray | None:
        """Load an image from disk, resize and normalize to [0, 1].

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            float32 array of shape (H, W, 3) in [0, 1], or None on failure.
        """
        path = Path(image_path)
        if not path.exists():
            logger.warning("Image file not found: %s", image_path)
            return None
        if not path.is_file():
            logger.warning("Path is not a file: %s", image_path)
            return None
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}:
            logger.warning("Unsupported image format: %s", path.suffix)
            return None

        try:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("cv2 could not read image: %s", image_path)
                return None
            return self._resize_and_normalize(img)
        except (cv2.error, OSError, ValueError) as exc:
            logger.error("Failed to load image %s: %s", image_path, exc)
            return None

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Apply Otsu binarization to a normalized float32 image.

        Converts to uint8 grayscale first, applies Otsu threshold, then
        returns a float32 binary image (0.0 or 1.0) stacked to 3 channels.

        Args:
            image: float32 array of shape (H, W, 3) in [0, 1].

        Returns:
            float32 array of shape (H, W, 3) with binary values.
        """
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_float = (binary / 255.0).astype(np.float32)
        return np.stack([binary_float] * 3, axis=-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resize_and_normalize(self, img: np.ndarray) -> np.ndarray:
        """Resize to target_size and normalize pixel values to [0, 1]."""
        img = cv2.resize(img, self.target_size)
        return img.astype("float32") / 255.0
