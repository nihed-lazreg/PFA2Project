"""Client metadata repository – file-based, DB-ready interface."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_METADATA_FILE = "models/clients_metadata.json"


@dataclass
class ClientRecord:
    """Persistent metadata for an enrolled client."""

    client_id: str
    enrolled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    num_signatures: int = 0
    notes: str = ""


class ClientRepository:
    """Manage client metadata (CRUD) with file-based persistence.

    The repository is safe to instantiate multiple times; all instances
    share the same on-disk JSON file.

    Args:
        metadata_path: Path to the JSON file used for persistence.
    """

    def __init__(self, metadata_path: str = _METADATA_FILE):
        self._path = Path(metadata_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, ClientRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exists(self, client_id: str) -> bool:
        """Return True if the client is registered."""
        return client_id in self._records

    def get(self, client_id: str) -> Optional[ClientRecord]:
        """Retrieve a client record or None if not found."""
        return self._records.get(client_id)

    def list_clients(self) -> List[str]:
        """Return a sorted list of all registered client IDs."""
        return sorted(self._records.keys())

    def upsert(self, record: ClientRecord) -> None:
        """Insert or replace a client record and persist to disk.

        Args:
            record: :class:`ClientRecord` to save.
        """
        record.updated_at = datetime.now(timezone.utc).isoformat()
        self._records[record.client_id] = record
        self._save()
        logger.debug("Client record upserted: %s", record.client_id)

    def delete(self, client_id: str) -> bool:
        """Remove a client from the repository.

        Args:
            client_id: ID of the client to remove.

        Returns:
            True if the client existed and was removed, False otherwise.
        """
        if client_id not in self._records:
            logger.warning("Cannot delete: client '%s' not found.", client_id)
            return False
        del self._records[client_id]
        self._save()
        logger.info("Client '%s' deleted.", client_id)
        return True

    def count(self) -> int:
        """Return the number of registered clients."""
        return len(self._records)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.debug("No client metadata file found; starting fresh.")
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                raw: dict = json.load(fh)
            self._records = {
                cid: ClientRecord(**data) for cid, data in raw.items()
            }
            logger.info("Loaded %d client records.", len(self._records))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.error("Failed to load client metadata: %s – starting fresh.", exc)
            self._records = {}

    def _save(self) -> None:
        try:
            tmp = self._path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(
                    {cid: asdict(rec) for cid, rec in self._records.items()},
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )
            tmp.replace(self._path)
        except OSError as exc:
            logger.error("Failed to save client metadata: %s", exc)
