"""FocalStackGeneration: focal-stack depth, AIF reconstruction, and reliability diagnostics."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("focaldiffusion")
except PackageNotFoundError:  # editable/source checkout before installation
    __version__ = "0.1.0"

__all__ = [
    "FocalTracePipeline",
    "FocalTraceOutput",
    "FocusLikelihoodEstimator",
    "ReliabilityFusionHead",
    "FocalConsistencyEvaluator",
    "FocalConsistencyTrace",
    "JointReconstructionDecoder",
    "FocalStackGenerationPipeline",
    "FocalStackProcessor",
    "FocalEvidenceEncoder",
]


def __getattr__(name: str):
    """Lazily import heavy modules on demand."""

    if name in {"FocalTracePipeline", "FocalTraceOutput", "FocalStackGenerationPipeline"}:
        from .pipelines import FocalTraceOutput, FocalTracePipeline, FocalStackGenerationPipeline

        return {
            "FocalTracePipeline": FocalTracePipeline,
            "FocalTraceOutput": FocalTraceOutput,
            "FocalStackGenerationPipeline": FocalStackGenerationPipeline,
        }[name]
    if name in {"FocalStackProcessor", "FocalEvidenceEncoder", "FocusLikelihoodEstimator", "ReliabilityFusionHead", "FocalConsistencyEvaluator", "FocalConsistencyTrace", "JointReconstructionDecoder"}:
        from .models import (
            FocalConsistencyEvaluator,
            FocalConsistencyTrace,
            FocalEvidenceEncoder,
            FocalStackProcessor,
            FocusLikelihoodEstimator,
            JointReconstructionDecoder,
            ReliabilityFusionHead,
        )

        mapping = {
            "FocalStackProcessor": FocalStackProcessor,
            "FocalEvidenceEncoder": FocalEvidenceEncoder,
            "FocusLikelihoodEstimator": FocusLikelihoodEstimator,
            "ReliabilityFusionHead": ReliabilityFusionHead,
            "FocalConsistencyEvaluator": FocalConsistencyEvaluator,
            "FocalConsistencyTrace": FocalConsistencyTrace,
            "JointReconstructionDecoder": JointReconstructionDecoder,
        }
        return mapping[name]

    raise AttributeError(f"module 'src' has no attribute '{name}'")
