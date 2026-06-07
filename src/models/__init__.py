"""FSDiffusion model components exposed for training and inference."""

from .dual_decoder import DualOutputDecoder
from .focal_attention import FocalCrossAttention
from .focal_evidence import (
    FocalEvidenceHead,
    PhysicalSupportHead,
    build_support_inputs,
    expected_metric_depth_from_focus_posterior,
)
from .focal_processor import FocalStackProcessor

__all__ = [
    "FocalCrossAttention",
    "FocalStackProcessor",
    "FocalEvidenceHead",
    "PhysicalSupportHead",
    "build_support_inputs",
    "expected_metric_depth_from_focus_posterior",
    "DualOutputDecoder",
]
