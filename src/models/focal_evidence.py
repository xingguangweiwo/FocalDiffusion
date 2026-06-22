"""Backward-compatible focal evidence exports for FocalTrace."""

from __future__ import annotations

from .focal_evidence_encoder import (
    FocusLikelihoodEstimator,
    ReliabilityFusionHead,
    build_reliability_features,
    FocalEvidenceEncoder,
    PhysicalEvidenceEstimator,
    build_physical_evidence_features,
    decode_metric_depth_from_focal_posterior,
)
from .verification_trace import FocalConsistencyTrace, PhysicalVerificationTrace

__all__ = [
    "FocusLikelihoodEstimator",
    "FocalEvidenceEncoder",
    "ReliabilityFusionHead",
    "PhysicalEvidenceEstimator",
    "FocalConsistencyTrace",
    "PhysicalVerificationTrace",
    "build_reliability_features",
    "build_physical_evidence_features",
    "decode_metric_depth_from_focal_posterior",
]
