"""System-wide configuration for the banking signature verification system."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_weights_path: str = "models/siamese_encoder_v2.weights.h5"
    embedding_size: int = 128
    input_image_size: tuple = (150, 150)

    # ── Storage ───────────────────────────────────────────────────────────────
    embeddings_db_path: str = "models/base_empreintes.npz"
    data_real_dir: str = "data/real"
    data_fake_dir: str = "data/fake"

    # ── Similarity thresholds (cosine distance: 0 = identical) ───────────────
    # Accept as authentic if cosine distance < threshold
    verification_threshold: float = 0.15
    # 1:N identification: accepted vs uncertain boundary
    identification_threshold: float = 0.15
    identification_uncertain_factor: float = 1.5  # threshold * factor = uncertain zone

    # ── Enrollment constraints ────────────────────────────────────────────────
    min_signatures_required: int = 1
    recommended_signatures: int = 5

    # ── Audit log ─────────────────────────────────────────────────────────────
    audit_log_path: str = "logs/audit.jsonl"

    # ── Supported image extensions ────────────────────────────────────────────
    supported_extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".bmp")

    # ── Identification ────────────────────────────────────────────────────────
    identification_top_k: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        """Override defaults from environment variables if present."""
        return cls(
            model_weights_path=os.getenv(
                "SIG_MODEL_PATH", cls.__dataclass_fields__["model_weights_path"].default
            ),
            embeddings_db_path=os.getenv(
                "SIG_EMBEDDINGS_DB", cls.__dataclass_fields__["embeddings_db_path"].default
            ),
            verification_threshold=float(
                os.getenv("SIG_VERIFY_THRESHOLD", cls.__dataclass_fields__["verification_threshold"].default)
            ),
            identification_threshold=float(
                os.getenv("SIG_IDENT_THRESHOLD", cls.__dataclass_fields__["identification_threshold"].default)
            ),
            audit_log_path=os.getenv(
                "SIG_AUDIT_LOG", cls.__dataclass_fields__["audit_log_path"].default
            ),
        )
