"""VerificationService – 1:1 signature authentication against a known client."""

import logging
from dataclasses import dataclass

from config.settings import Settings
from core.embedding import EmbeddingGenerator
from core.similarity import SimilarityEngine
from storage.embedding_repository import EmbeddingRepository
from services.audit import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of a 1:1 verification request."""

    is_authentic: bool
    client_id: str
    cosine_distance: float
    threshold_used: float
    confidence_pct: float  # 100 * cosine_similarity
    status: str            # "AUTHENTIC" | "REJECTED" | "ERROR"
    message: str = ""

    def __repr__(self) -> str:
        return (
            f"VerificationResult(status={self.status}, client='{self.client_id}', "
            f"dist={self.cosine_distance:.4f}, threshold={self.threshold_used})"
        )


class VerificationService:
    """Verify whether a signature belongs to a specific enrolled client.

    Performs 1:1 comparison: one signature against one client's stored
    embeddings.  The decision is based on the minimum cosine distance
    between the query embedding and any of the client's enrolled embeddings.

    Args:
        embedding_generator: :class:`EmbeddingGenerator` instance.
        embedding_repo:      :class:`EmbeddingRepository` instance.
        audit_logger:        :class:`AuditLogger` instance.
        settings:            :class:`Settings` configuration.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        embedding_repo: EmbeddingRepository,
        audit_logger: AuditLogger,
        settings: Settings,
    ):
        self._gen = embedding_generator
        self._embedding_repo = embedding_repo
        self._audit = audit_logger
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        client_id: str,
        signature_path: str,
        threshold: float | None = None,
    ) -> VerificationResult:
        """Verify a signature against a specific client.

        Args:
            client_id:      ID of the enrolled client to match against.
            signature_path: Path to the query signature image.
            threshold:      Cosine-distance acceptance threshold.
                            Defaults to ``settings.verification_threshold``.

        Returns:
            :class:`VerificationResult` with the authentication decision.
        """
        threshold = threshold if threshold is not None else self._settings.verification_threshold

        # Validate client
        client_embeddings = self._embedding_repo.get(client_id)
        if client_embeddings is None:
            msg = f"Client '{client_id}' not enrolled in the system."
            logger.warning(msg)
            self._audit.log(
                "VERIFY_ERROR",
                {"client_id": client_id, "signature": signature_path, "reason": msg},
            )
            return VerificationResult(
                is_authentic=False,
                client_id=client_id,
                cosine_distance=2.0,
                threshold_used=threshold,
                confidence_pct=0.0,
                status="ERROR",
                message=msg,
            )

        # Encode query signature
        query_emb = self._gen.encode(signature_path)
        if query_emb is None:
            msg = f"Could not encode query signature: '{signature_path}'."
            logger.error(msg)
            self._audit.log(
                "VERIFY_ERROR",
                {"client_id": client_id, "signature": signature_path, "reason": msg},
            )
            return VerificationResult(
                is_authentic=False,
                client_id=client_id,
                cosine_distance=2.0,
                threshold_used=threshold,
                confidence_pct=0.0,
                status="ERROR",
                message=msg,
            )

        dist_min, _dist_mean = SimilarityEngine.best_match_distance(
            query_emb, client_embeddings
        )
        confidence_pct = max(0.0, (1.0 - dist_min) * 100)

        is_authentic = dist_min < threshold
        status = "AUTHENTIC" if is_authentic else "REJECTED"

        self._audit.log(
            "VERIFY",
            {
                "client_id": client_id,
                "signature": signature_path,
                "cosine_distance": round(dist_min, 6),
                "threshold": threshold,
                "decision": status,
            },
        )

        logger.info(
            "Verification for client '%s': %s (dist=%.4f, threshold=%.4f).",
            client_id, status, dist_min, threshold,
        )

        return VerificationResult(
            is_authentic=is_authentic,
            client_id=client_id,
            cosine_distance=dist_min,
            threshold_used=threshold,
            confidence_pct=confidence_pct,
            status=status,
        )
