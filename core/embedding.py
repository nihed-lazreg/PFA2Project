"""Embedding generator – converts signature images to L2-normalized vectors."""

import logging
from typing import List, Tuple

import numpy as np

from .model import SiameseModel
from .preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Encodes signature images into fixed-size L2-normalized embedding vectors.

    Combines :class:`ImagePreprocessor` and :class:`SiameseModel` into a
    single high-level API used by the service layer.

    Args:
        model:         A ready :class:`SiameseModel` instance.
        preprocessor:  An :class:`ImagePreprocessor` instance.
    """

    def __init__(self, model: SiameseModel, preprocessor: ImagePreprocessor):
        self._model = model
        self._preprocessor = preprocessor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, image_path: str) -> np.ndarray | None:
        """Encode a single signature image.

        Args:
            image_path: Path to the signature image file.

        Returns:
            1-D float32 array of shape ``(embedding_size,)``, or ``None``
            if the image could not be loaded or encoded.
        """
        img = self._preprocessor.load(image_path)
        if img is None:
            logger.warning("Could not load image for encoding: %s", image_path)
            return None
        batch = np.expand_dims(img, axis=0)
        try:
            embeddings = self._model.predict_batch(batch)
            return embeddings[0]
        except RuntimeError as exc:
            logger.error("Encoding failed: %s", exc)
            return None

    def encode_batch(
        self, image_paths: List[str]
    ) -> Tuple[np.ndarray | None, List[str]]:
        """Encode a list of signature images in a single forward pass.

        Args:
            image_paths: List of file paths to signature images.

        Returns:
            Tuple ``(embeddings, valid_paths)`` where:
              - ``embeddings`` has shape ``(N, embedding_size)`` (or ``None``
                if no images could be loaded).
              - ``valid_paths`` contains only the successfully loaded paths.
        """
        images: list = []
        valid_paths: list = []

        for path in image_paths:
            img = self._preprocessor.load(path)
            if img is not None:
                images.append(img)
                valid_paths.append(path)
            else:
                logger.warning("Skipping unreadable image: %s", path)

        if not images:
            logger.error("No valid images found in batch of %d paths.", len(image_paths))
            return None, []

        batch = np.array(images)
        try:
            embeddings = self._model.predict_batch(batch)
            return embeddings, valid_paths
        except RuntimeError as exc:
            logger.error("Batch encoding failed: %s", exc)
            return None, []
