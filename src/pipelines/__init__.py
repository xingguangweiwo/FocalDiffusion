"""FocalStackGeneration pipelines."""

from .focal_stack_generation_pipeline import (
    FocalInjectedSD3Transformer,
    FocalTraceOutput,
    FocalTracePipeline,
    FocalStackGenerationOutput,
    FocalStackGenerationPipeline,
    FocalDiffusionOutput,
    FocalDiffusionPipeline,
)

__all__ = [
    "FocalTracePipeline",
    "FocalTraceOutput",
    "FocalStackGenerationPipeline",
    "FocalStackGenerationOutput",
    "FocalInjectedSD3Transformer",
    "FocalDiffusionOutput",
    "FocalDiffusionPipeline",
    "load_pipeline",
    "save_pipeline",
    "migrate_checkpoint_schema",
    "migrate_config_schema",
]


def __getattr__(name: str):
    """Lazily load utility helpers that may import optional runtime dependencies."""

    if name in {"load_pipeline", "save_pipeline", "migrate_checkpoint_schema", "migrate_config_schema"}:
        from .pipeline_utils import load_pipeline, save_pipeline, migrate_checkpoint_schema, migrate_config_schema

        mapping = {
            "load_pipeline": load_pipeline,
            "save_pipeline": save_pipeline,
            "migrate_checkpoint_schema": migrate_checkpoint_schema,
            "migrate_config_schema": migrate_config_schema,
        }
        return mapping[name]

    raise AttributeError(f"module 'src.pipelines' has no attribute {name!r}")
