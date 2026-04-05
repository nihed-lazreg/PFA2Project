"""Structured audit logger for banking-grade traceability."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AuditLogger:
    """Append-only JSONL audit log for all system actions.

    Each event is written as a single JSON line to the audit log file.
    The file is never truncated; old records are never modified.

    Args:
        log_path: Path to the ``.jsonl`` audit log file.
    """

    def __init__(self, log_path: str = "logs/audit.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: str, details: Dict[str, Any]) -> None:
        """Write a structured audit event.

        Args:
            action:  Short uppercase string describing the action
                     (e.g. ``"ENROLL"``, ``"VERIFY"``, ``"IDENTIFY"``).
            details: Arbitrary key-value pairs to include in the event.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **details,
        }
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Audit log write failed: %s", exc)
