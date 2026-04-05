"""Embedding repository – stores and retrieves client signature embeddings.

Persistence is file-based (NumPy .npz) with a clear abstraction layer so
the storage backend can be swapped for a database without touching service code.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingRepository:
    """Persist and retrieve L2-normalized embedding matrices per client.

    Storage format: a single ``.npz`` file keyed by client ID.

    Args:
        db_path: Path to the ``.npz`` embeddings database file.
    """

    def __init__(self, db_path: str = "models/base_empreintes.npz"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Dict[client_id -> ndarray of shape (N, embedding_size)]
        self._store: Dict[str, np.ndarray] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_client(self, client_id: str) -> bool:
        """Return True if embeddings for the client exist."""
        return client_id in self._store

    def get(self, client_id: str) -> Optional[np.ndarray]:
        """Return embeddings for a client, or None if not found.

        Returns:
            float32 array of shape ``(N, embedding_size)`` or ``None``.
        """
        return self._store.get(client_id)

    def all_clients(self) -> List[str]:
        """Return a sorted list of all client IDs with stored embeddings."""
        return sorted(self._store.keys())

    def all_embeddings(self) -> Dict[str, np.ndarray]:
        """Return a copy of the full embedding store."""
        return dict(self._store)

    def count(self) -> int:
        """Return the number of clients with stored embeddings."""
        return len(self._store)

    def upsert(self, client_id: str, embeddings: np.ndarray) -> None:
        """Store or replace embeddings for a client and persist to disk.

        Args:
            client_id:  Client identifier string.
            embeddings: float32 array of shape ``(N, embedding_size)``.
        """
        self._store[client_id] = embeddings.astype(np.float32)
        self._save()
        logger.debug(
            "Embeddings upserted for client '%s': %d vectors.", client_id, len(embeddings)
        )

    def delete(self, client_id: str) -> bool:
        """Remove embeddings for a client.

        Returns:
            True if the client existed and was removed.
        """
        if client_id not in self._store:
            logger.warning("Cannot delete: no embeddings for client '%s'.", client_id)
            return False
        del self._store[client_id]
        self._save()
        logger.info("Embeddings deleted for client '%s'.", client_id)
        return True

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.debug("No embeddings database found at '%s'; starting fresh.", self._path)
            return
        try:
            data = np.load(str(self._path), allow_pickle=True)
            # Support both legacy format (clients dict) and new flat format
            if "clients" in data:
                raw = data["clients"].item()
                self._store = {
                    cid: np.array(info["empreintes"], dtype=np.float32)
                    for cid, info in raw.items()
                }
                logger.info(
                    "Migrated legacy embeddings DB: %d clients loaded.", len(self._store)
                )
            else:
                self._store = {
                    k: data[k].astype(np.float32)
                    for k in data.files
                }
                logger.info("Embeddings DB loaded: %d clients.", len(self._store))
        except Exception as exc:
            logger.error("Failed to load embeddings database: %s", exc)
            self._store = {}

    def _save(self) -> None:
        try:
            # np.savez appends ".npz" automatically; strip the extension to avoid
            # double-extension issues, then rename back to the desired path.
            stem = str(self._path.with_suffix(""))
            tmp_stem = stem + "_tmp"
            np.savez(tmp_stem, **self._store)
            tmp_path = Path(tmp_stem + ".npz")
            tmp_path.replace(self._path)
        except OSError as exc:
            logger.error("Failed to save embeddings database: %s", exc)
