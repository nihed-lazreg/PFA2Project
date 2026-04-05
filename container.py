"""Factory for wiring together all system components.

Provides a single entry point to build the fully-configured service objects
without scattering construction logic across the codebase.
"""

import logging

from config.settings import Settings
from core.embedding import EmbeddingGenerator
from core.model import SiameseModel
from core.preprocessing import ImagePreprocessor
from services.audit import AuditLogger
from services.enrollment import EnrollmentService
from services.identification import IdentificationService
from services.verification import VerificationService
from storage.client_repository import ClientRepository
from storage.embedding_repository import EmbeddingRepository

logger = logging.getLogger(__name__)


def build_services(settings: Settings | None = None):
    """Build and return all service instances, wired together.

    Args:
        settings: Optional :class:`Settings` override.
                  If None, uses default Settings().

    Returns:
        Tuple ``(enrollment, verification, identification, audit_logger)``.

    Raises:
        RuntimeError: If the model weights cannot be loaded.
    """
    if settings is None:
        settings = Settings()

    audit = AuditLogger(log_path=settings.audit_log_path)

    model = SiameseModel(
        weights_path=settings.model_weights_path,
        embedding_size=settings.embedding_size,
        input_size=settings.input_image_size,
    )
    if not model.is_ready:
        msg = (
            f"Model weights not found at '{settings.model_weights_path}'. "
            "Run training with siamese_encoder.py first."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    preprocessor = ImagePreprocessor(target_size=settings.input_image_size)
    generator = EmbeddingGenerator(model=model, preprocessor=preprocessor)

    client_repo = ClientRepository()
    embedding_repo = EmbeddingRepository(db_path=settings.embeddings_db_path)

    enrollment = EnrollmentService(
        embedding_generator=generator,
        client_repo=client_repo,
        embedding_repo=embedding_repo,
        audit_logger=audit,
        settings=settings,
    )

    verification = VerificationService(
        embedding_generator=generator,
        embedding_repo=embedding_repo,
        audit_logger=audit,
        settings=settings,
    )

    identification = IdentificationService(
        embedding_generator=generator,
        embedding_repo=embedding_repo,
        audit_logger=audit,
        settings=settings,
    )

    logger.info(
        "Services initialized. Enrolled clients: %d", embedding_repo.count()
    )
    return enrollment, verification, identification, audit
