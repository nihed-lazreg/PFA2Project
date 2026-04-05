"""EnrollmentService – register new clients or update existing ones."""

import logging
from datetime import datetime, timezone
from typing import List

from config.settings import Settings
from core.embedding import EmbeddingGenerator
from storage.client_repository import ClientRecord, ClientRepository
from storage.embedding_repository import EmbeddingRepository
from services.audit import AuditLogger

logger = logging.getLogger(__name__)


class EnrollmentResult:
    """Result returned by :meth:`EnrollmentService.enroll`."""

    def __init__(
        self,
        success: bool,
        client_id: str,
        num_enrolled: int = 0,
        num_skipped: int = 0,
        message: str = "",
        is_update: bool = False,
    ):
        self.success = success
        self.client_id = client_id
        self.num_enrolled = num_enrolled
        self.num_skipped = num_skipped
        self.message = message
        self.is_update = is_update  # True if client already existed

    def __repr__(self) -> str:
        status = "UPDATE" if self.is_update else "NEW"
        return (
            f"EnrollmentResult(success={self.success}, client='{self.client_id}', "
            f"enrolled={self.num_enrolled}, skipped={self.num_skipped}, "
            f"type={status})"
        )


class EnrollmentService:
    """Register or update a client's signature embeddings.

    Args:
        embedding_generator: :class:`EmbeddingGenerator` instance.
        client_repo:         :class:`ClientRepository` instance.
        embedding_repo:      :class:`EmbeddingRepository` instance.
        audit_logger:        :class:`AuditLogger` instance.
        settings:            :class:`Settings` configuration.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        client_repo: ClientRepository,
        embedding_repo: EmbeddingRepository,
        audit_logger: AuditLogger,
        settings: Settings,
    ):
        self._gen = embedding_generator
        self._client_repo = client_repo
        self._embedding_repo = embedding_repo
        self._audit = audit_logger
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enroll(
        self,
        client_id: str,
        signature_paths: List[str],
        overwrite: bool = False,
        notes: str = "",
    ) -> EnrollmentResult:
        """Enroll a client with one or more signature images.

        Args:
            client_id:        Unique string identifier for the client.
            signature_paths:  List of paths to signature image files.
            overwrite:        If True, replace existing embeddings.
                              If False and the client exists, returns an error.
            notes:            Optional free-text notes to store with the record.

        Returns:
            :class:`EnrollmentResult` with outcome details.
        """
        if not client_id or not client_id.strip():
            msg = "Client ID must not be empty."
            logger.error(msg)
            self._audit.log("ENROLL_REJECTED", {"client_id": client_id, "reason": msg})
            return EnrollmentResult(False, client_id, message=msg)

        if not signature_paths:
            msg = "At least one signature path must be provided."
            logger.error(msg)
            self._audit.log("ENROLL_REJECTED", {"client_id": client_id, "reason": msg})
            return EnrollmentResult(False, client_id, message=msg)

        is_update = self._client_repo.exists(client_id)

        if is_update and not overwrite:
            msg = (
                f"Client '{client_id}' already enrolled. "
                "Pass overwrite=True to update."
            )
            logger.warning(msg)
            self._audit.log("ENROLL_REJECTED", {"client_id": client_id, "reason": msg})
            return EnrollmentResult(False, client_id, message=msg, is_update=True)

        # Encode signatures
        embeddings, valid_paths = self._gen.encode_batch(signature_paths)
        num_skipped = len(signature_paths) - len(valid_paths)

        if embeddings is None or len(embeddings) == 0:
            msg = "All signature images failed to encode. Check file paths and formats."
            logger.error(msg)
            self._audit.log(
                "ENROLL_FAILED",
                {"client_id": client_id, "paths": signature_paths},
            )
            return EnrollmentResult(False, client_id, message=msg)

        # Persist
        self._embedding_repo.upsert(client_id, embeddings)

        record = ClientRecord(
            client_id=client_id,
            num_signatures=len(embeddings),
            notes=notes,
        )
        self._client_repo.upsert(record)

        action = "ENROLL_UPDATE" if is_update else "ENROLL_NEW"
        self._audit.log(
            action,
            {
                "client_id": client_id,
                "num_enrolled": len(embeddings),
                "num_skipped": num_skipped,
                "valid_paths": valid_paths,
            },
        )

        logger.info(
            "Client '%s' enrolled: %d embeddings stored (%d skipped).",
            client_id, len(embeddings), num_skipped,
        )

        return EnrollmentResult(
            success=True,
            client_id=client_id,
            num_enrolled=len(embeddings),
            num_skipped=num_skipped,
            message="Enrollment successful.",
            is_update=is_update,
        )

    def delete_client(self, client_id: str) -> bool:
        """Remove all data for a client.

        Args:
            client_id: Client to remove.

        Returns:
            True if the client was found and removed.
        """
        emb_ok = self._embedding_repo.delete(client_id)
        meta_ok = self._client_repo.delete(client_id)
        removed = emb_ok or meta_ok
        if removed:
            self._audit.log("CLIENT_DELETED", {"client_id": client_id})
            logger.info("Client '%s' deleted.", client_id)
        else:
            logger.warning("Delete requested for unknown client '%s'.", client_id)
        return removed
