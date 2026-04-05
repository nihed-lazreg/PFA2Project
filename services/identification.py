"""IdentificationService – 1:N signature search across all enrolled clients."""

import logging
from dataclasses import dataclass, field
from typing import List

from config.settings import Settings
from core.embedding import EmbeddingGenerator
from core.similarity import SimilarityEngine
from storage.embedding_repository import EmbeddingRepository
from services.audit import AuditLogger

logger = logging.getLogger(__name__)


@dataclass
class ClientMatch:
    """A single candidate match from an identification search."""

    client_id: str
    cosine_distance: float
    cosine_distance_mean: float
    confidence_pct: float  # 100 * (1 - cosine_distance)


@dataclass
class IdentificationResult:
    """Outcome of a 1:N identification request."""

    status: str            # "IDENTIFIED" | "UNCERTAIN" | "UNKNOWN" | "ERROR"
    best_match: ClientMatch | None
    top_candidates: List[ClientMatch] = field(default_factory=list)
    threshold_used: float = 0.0
    message: str = ""

    def __repr__(self) -> str:
        best = self.best_match.client_id if self.best_match else "None"
        return (
            f"IdentificationResult(status={self.status}, best='{best}', "
            f"threshold={self.threshold_used})"
        )


class IdentificationService:
    """Search a query signature against all enrolled clients (1:N).

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

    def identify(
        self,
        signature_path: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> IdentificationResult:
        """Search all enrolled clients for the best signature match.

        Args:
            signature_path: Path to the query signature image.
            top_k:          Maximum candidates to return
                            (defaults to ``settings.identification_top_k``).
            threshold:      Cosine-distance threshold for "IDENTIFIED" status
                            (defaults to ``settings.identification_threshold``).

        Returns:
            :class:`IdentificationResult` with ranked candidates.
        """
        top_k = top_k if top_k is not None else self._settings.identification_top_k
        threshold = threshold if threshold is not None else self._settings.identification_threshold
        uncertain_threshold = threshold * self._settings.identification_uncertain_factor

        all_clients = self._embedding_repo.all_clients()
        if not all_clients:
            msg = "No clients enrolled in the system."
            logger.warning(msg)
            return IdentificationResult(
                status="ERROR",
                best_match=None,
                threshold_used=threshold,
                message=msg,
            )

        # Encode query signature
        query_emb = self._gen.encode(signature_path)
        if query_emb is None:
            msg = f"Could not encode query signature: '{signature_path}'."
            logger.error(msg)
            self._audit.log(
                "IDENTIFY_ERROR",
                {"signature": signature_path, "reason": msg},
            )
            return IdentificationResult(
                status="ERROR",
                best_match=None,
                threshold_used=threshold,
                message=msg,
            )

        # Score against every enrolled client
        candidates: List[ClientMatch] = []
        for client_id in all_clients:
            client_embeddings = self._embedding_repo.get(client_id)
            if client_embeddings is None:
                continue
            dist_min, dist_mean = SimilarityEngine.best_match_distance(
                query_emb, client_embeddings
            )
            candidates.append(
                ClientMatch(
                    client_id=client_id,
                    cosine_distance=dist_min,
                    cosine_distance_mean=dist_mean,
                    confidence_pct=max(0.0, (1.0 - dist_min) * 100),
                )
            )

        if not candidates:
            msg = "No embeddings available for any enrolled client."
            logger.error(msg)
            return IdentificationResult(
                status="ERROR",
                best_match=None,
                threshold_used=threshold,
                message=msg,
            )

        candidates.sort(key=lambda c: c.cosine_distance)
        best = candidates[0]
        top_candidates = candidates[:top_k]

        # Decision
        if best.cosine_distance < threshold:
            status = "IDENTIFIED"
        elif best.cosine_distance < uncertain_threshold:
            status = "UNCERTAIN"
        else:
            status = "UNKNOWN"

        self._audit.log(
            "IDENTIFY",
            {
                "signature": signature_path,
                "status": status,
                "best_client": best.client_id,
                "cosine_distance": round(best.cosine_distance, 6),
                "threshold": threshold,
                "num_clients_searched": len(candidates),
            },
        )

        logger.info(
            "Identification: %s (best='%s', dist=%.4f, threshold=%.4f).",
            status, best.client_id, best.cosine_distance, threshold,
        )

        return IdentificationResult(
            status=status,
            best_match=best,
            top_candidates=top_candidates,
            threshold_used=threshold,
        )
