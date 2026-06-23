"""FocalStackGeneration model components exposed for training and inference."""

from .task_output_decoder import JointReconstructionDecoder, TaskOutputDecoder
from .focal_attention import FocalCrossAttention
from .focal_evidence_encoder import (
    FocusLikelihoodEstimator,
    ReliabilityFusionHead,
    build_reliability_features,
    FocalEvidenceEncoder,
    PhysicalEvidenceEstimator,
    build_physical_evidence_features,
    decode_metric_depth_from_focal_posterior,
)
from .focal_processor import FocalStackProcessor
from .verification_trace import FocalConsistencyTrace, PhysicalVerificationTrace
from .physics_modules import FocusMeasureBank, DefocusConsistencyVerifier, FocalConsistencyEvaluator, FocalPhysicalVerifier

__all__ = [
    "FocalCrossAttention",
    "FocalStackProcessor",
    "FocusLikelihoodEstimator",
    "FocalEvidenceEncoder",
    "ReliabilityFusionHead",
    "PhysicalEvidenceEstimator",
    "build_reliability_features",
    "build_physical_evidence_features",
    "decode_metric_depth_from_focal_posterior",
    "JointReconstructionDecoder",
    "TaskOutputDecoder",
    "FocalConsistencyTrace",
    "PhysicalVerificationTrace",
    "FocusMeasureBank",
    "DefocusConsistencyVerifier",
    "FocalConsistencyEvaluator",
    "FocalPhysicalVerifier",
]
