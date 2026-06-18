"""FocalStackGeneration model components exposed for training and inference."""

from .task_output_decoder import TaskOutputDecoder
from .focal_attention import FocalCrossAttention
from .focal_evidence_encoder import (
    FocalEvidenceEncoder,
    PhysicalEvidenceEstimator,
    build_physical_evidence_features,
    decode_metric_depth_from_focal_posterior,
)
from .focal_processor import FocalStackProcessor
from .verification_trace import PhysicalVerificationTrace
from .physics_modules import FocusMeasureBank, DefocusConsistencyVerifier, FocalPhysicalVerifier

__all__ = [
    "FocalCrossAttention",
    "FocalStackProcessor",
    "FocalEvidenceEncoder",
    "PhysicalEvidenceEstimator",
    "build_physical_evidence_features",
    "decode_metric_depth_from_focal_posterior",
    "TaskOutputDecoder",
    "PhysicalVerificationTrace",
    "FocusMeasureBank",
    "DefocusConsistencyVerifier",
    "FocalPhysicalVerifier",
]
