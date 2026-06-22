"""FocalStackGeneration: focal-stack depth, AIF reconstruction, and reliability diagnostics."""

__version__ = "1.0.0"

__all__ = [
    "FocalStackGenerationPipeline",
    "FocalStackProcessor",
    "FocalEvidenceEncoder",
]


def __getattr__(name: str):
    """Lazily import heavy modules on demand."""

    if name == "FocalStackGenerationPipeline":
        from .pipelines import FocalStackGenerationPipeline

        return FocalStackGenerationPipeline
    if name in {"FocalStackProcessor", "FocalEvidenceEncoder"}:
        from .models import FocalEvidenceEncoder, FocalStackProcessor

        mapping = {
            "FocalStackProcessor": FocalStackProcessor,
            "FocalEvidenceEncoder": FocalEvidenceEncoder,
        }
        return mapping[name]

    raise AttributeError(f"module 'src' has no attribute '{name}'")
