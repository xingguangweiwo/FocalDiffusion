"""FocalDiffusion pipelines."""

from .focal_diffusion_pipeline import (
    FocalDiffusionOutput,
    FocalDiffusionPipeline,
    FocalInjectedSD3Transformer,
)

__all__ = [
    "FocalDiffusionPipeline",
    "FocalDiffusionOutput",
    "FocalInjectedSD3Transformer",
    "load_pipeline",
    "save_pipeline",
]


def __getattr__(name: str):
    """Lazily load utility helpers that may import optional runtime dependencies."""

    if name in {"load_pipeline", "save_pipeline"}:
        from .pipeline_utils import load_pipeline, save_pipeline

        mapping = {
            "load_pipeline": load_pipeline,
            "save_pipeline": save_pipeline,
        }
        return mapping[name]

    raise AttributeError(f"module 'src.pipelines' has no attribute {name!r}")
