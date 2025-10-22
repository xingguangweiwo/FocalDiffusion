"""FocalDiffusion - Zero-shot depth and all-in-focus generation from focal stacks."""

__version__ = "1.0.0"

__all__ = [
    "FocalDiffusionPipeline",
    "FocalStackProcessor",
    "CameraInvariantEncoder",
]


def __getattr__(name: str):
    """Lazily import heavy modules on demand.

    Importing :mod:`src` previously pulled in ``torch`` and other optional
    dependencies immediately which prevented lightweight utilities (such as
    the YAML loader) from running in environments where those libraries are
    not installed.  The lazy loader keeps the public API intact while deferring
    the expensive imports until the corresponding attributes are actually
    requested.
    """

    if name == "FocalDiffusionPipeline":
        from .pipelines import FocalDiffusionPipeline

        return FocalDiffusionPipeline
    if name in {"FocalStackProcessor", "CameraInvariantEncoder"}:
        from .models import CameraInvariantEncoder, FocalStackProcessor

        mapping = {
            "FocalStackProcessor": FocalStackProcessor,
            "CameraInvariantEncoder": CameraInvariantEncoder,
        }
        return mapping[name]

    raise AttributeError(f"module 'src' has no attribute '{name}'")
