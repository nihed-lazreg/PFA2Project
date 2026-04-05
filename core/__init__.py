from .preprocessing import ImagePreprocessor
from .similarity import SimilarityEngine

# TensorFlow-dependent modules are imported lazily to avoid hard failures
# in environments where TF is not installed.
try:
    from .model import SiameseModel
    from .embedding import EmbeddingGenerator
except ImportError:
    pass

__all__ = [
    "ImagePreprocessor",
    "SiameseModel",
    "EmbeddingGenerator",
    "SimilarityEngine",
]
