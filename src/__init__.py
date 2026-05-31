"""FSDiffusion: Reliable Zero-Shot Focal-Stack Diffusion via Focal Evidence."""

__version__ = "1.0.0"

__all__ = [
    "FocalDiffusionPipeline",
    "FocalStackProcessor",
    "FocalEvidenceHead",
]


def __getattr__(name: str):
    """Lazily import heavy modules on demand."""

    if name == "FocalDiffusionPipeline":
        from .pipelines import FocalDiffusionPipeline

        return FocalDiffusionPipeline
    if name in {"FocalStackProcessor", "FocalEvidenceHead"}:
        from .models import FocalEvidenceHead, FocalStackProcessor

        mapping = {
            "FocalStackProcessor": FocalStackProcessor,
            "FocalEvidenceHead": FocalEvidenceHead,
        }
        return mapping[name]

    raise AttributeError(f"module 'src' has no attribute '{name}'")
