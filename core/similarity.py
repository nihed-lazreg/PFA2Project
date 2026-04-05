"""Similarity computation between L2-normalized embedding vectors."""

import numpy as np


class SimilarityEngine:
    """Compute cosine and Euclidean similarity/distance between embeddings.

    All methods are static – no state is required.
    The encoder already L2-normalizes outputs, so cosine similarity reduces
    to the dot product.  Explicit normalization is applied as a safety guard.
    """

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity ∈ [−1, 1].  1.0 = identical directions.

        Args:
            emb_a: 1-D float32 array.
            emb_b: 1-D float32 array of the same shape.

        Returns:
            Scalar float in [−1, 1].
        """
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(emb_a / norm_a, emb_b / norm_b))

    @staticmethod
    def cosine_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine distance ∈ [0, 2].  0.0 = identical.

        Args:
            emb_a: 1-D float32 array.
            emb_b: 1-D float32 array of the same shape.

        Returns:
            Scalar float in [0, 2].
        """
        return 1.0 - SimilarityEngine.cosine_similarity(emb_a, emb_b)

    @staticmethod
    def euclidean_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Euclidean distance between two embedding vectors.

        Args:
            emb_a: 1-D float32 array.
            emb_b: 1-D float32 array of the same shape.

        Returns:
            Non-negative scalar float.
        """
        return float(np.linalg.norm(emb_a - emb_b))

    @staticmethod
    def best_match_distance(
        query: np.ndarray, embeddings: np.ndarray
    ) -> tuple[float, float]:
        """Compute min and mean cosine distance between *query* and a set of embeddings.

        Args:
            query:      1-D float32 embedding of shape ``(D,)``.
            embeddings: 2-D float32 array of shape ``(N, D)``.

        Returns:
            Tuple ``(min_distance, mean_distance)`` using cosine distance.
        """
        # For L2-normalized vectors dot product = cosine similarity
        cos_sims = embeddings @ query  # (N,)
        sim_max = float(np.max(cos_sims))
        sim_mean = float(np.mean(cos_sims))
        return 1.0 - sim_max, 1.0 - sim_mean
