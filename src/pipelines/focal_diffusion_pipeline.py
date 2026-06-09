"""Deprecated compatibility imports for the renamed focal-stack generation pipeline."""

from .focal_stack_generation_pipeline import (  # noqa: F401
    FocalInjectedSD3Transformer,
    FocalStackGenerationOutput,
    FocalStackGenerationPipeline,
)

# Backward-compatible aliases for external scripts using pre-rename APIs.
FocalDiffusionOutput = FocalStackGenerationOutput
FocalDiffusionPipeline = FocalStackGenerationPipeline
