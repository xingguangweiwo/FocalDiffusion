"""Backward-compatible focal evidence exports for FocalTrace."""

from __future__ import annotations

from .focal_evidence_encoder import (
    FocalEvidenceEncoder,
    PhysicalEvidenceEstimator,
    build_physical_evidence_features,
    decode_metric_depth_from_focal_posterior,
)
from .verification_trace import PhysicalVerificationTrace

__all__ = [
    "FocalEvidenceEncoder",
    "PhysicalEvidenceEstimator",
    "PhysicalVerificationTrace",
    "build_physical_evidence_features",
    "decode_metric_depth_from_focal_posterior",
]
