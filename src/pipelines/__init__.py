"""FocalStackGeneration pipelines."""

from .focal_stack_generation_pipeline import (
    FocalInjectedSD3Transformer,
    FocalStackGenerationOutput,
    FocalStackGenerationPipeline,
    FocalDiffusionOutput,
    FocalDiffusionPipeline,
)

__all__ = [
    "FocalStackGenerationPipeline",
    "FocalStackGenerationOutput",
    "FocalInjectedSD3Transformer",
    "FocalDiffusionOutput",
    "FocalDiffusionPipeline",
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
