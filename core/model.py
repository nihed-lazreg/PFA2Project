"""Siamese CNN model wrapper – inference only.

Training is kept in the legacy ``siamese_encoder.py`` script and is NOT
mixed with inference logic here.
"""

import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)


def _build_encoder(input_shape: tuple, embedding_size: int) -> tf.keras.Model:
    """Build the shared CNN encoder graph (architecture must match trained weights)."""
    model = models.Sequential(
        [
            layers.Conv2D(32, (5, 5), activation="relu", padding="same",
                          input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(embedding_size, activation=None),
            layers.Lambda(
                lambda x: tf.math.l2_normalize(x, axis=1),
                name="l2_normalize",
            ),
        ],
        name="encoder",
    )
    return model


class SiameseModel:
    """Thin wrapper around the trained Siamese CNN encoder.

    Responsibilities:
      - Load pre-trained weights from disk.
      - Expose a single :meth:`predict_batch` method for batch inference.

    This class intentionally has **no** training logic.

    Args:
        weights_path:    Path to the ``.weights.h5`` file.
        embedding_size:  Dimensionality of the output embeddings (default 128).
        input_size:      (height, width) of the input images (default 150×150).
    """

    def __init__(
        self,
        weights_path: str = "models/siamese_encoder_v2.weights.h5",
        embedding_size: int = 128,
        input_size: tuple = (150, 150),
    ):
        self.weights_path = weights_path
        self.embedding_size = embedding_size
        self.input_shape = (*input_size, 3)

        self._encoder = _build_encoder(self.input_shape, embedding_size)
        self._loaded = False
        self._load_weights()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if weights were loaded successfully."""
        return self._loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Run the encoder on a batch of preprocessed images.

        Args:
            images: float32 array of shape ``(N, H, W, 3)`` in [0, 1].

        Returns:
            L2-normalized embeddings of shape ``(N, embedding_size)``.

        Raises:
            RuntimeError: If weights have not been loaded.
        """
        if not self._loaded:
            raise RuntimeError(
                "Model weights not loaded. Run training first or check the weights path."
            )
        return self._encoder.predict(images, verbose=0)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        if not os.path.exists(self.weights_path):
            logger.error(
                "Model weights not found at '%s'. Run training first.",
                self.weights_path,
            )
            return
        try:
            self._encoder.load_weights(self.weights_path)
            logger.info("Model weights loaded from '%s'.", self.weights_path)
            self._loaded = True
        except Exception as exc:
            logger.error("Failed to load model weights: %s", exc)
