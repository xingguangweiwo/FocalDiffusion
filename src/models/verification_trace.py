"""Typed containers for focal-stack consistency diagnostic outputs."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class FocalConsistencyTrace:
    """Focus and image-formation consistency diagnostics for a focal stack.

    These scores are self-consistency diagnostics, not ground-truth correctness
    labels. All score fields are tensors shaped ``[B, 1, H, W]`` except
    ``verdict_scores`` is retained as a compatibility alias for diagnostic
    logits shaped ``[B, 3, H, W]``; it is not a class-probability tensor.
    """

    focus_peak_confidence: torch.Tensor
    focus_peak_index: torch.Tensor
    focus_peak_coordinate: torch.Tensor | None
    focus_margin: torch.Tensor
    focus_entropy: torch.Tensor
    operator_agreement: torch.Tensor
    texture_confidence: torch.Tensor
    depth_focus_discrepancy: torch.Tensor
    stack_reprojection_residual: torch.Tensor
    focus_support: torch.Tensor
    generation_support: torch.Tensor
    conflict_score: torch.Tensor
    invalid_score: torch.Tensor
    verdict_scores: torch.Tensor
    focus_identifiability: torch.Tensor | None = None
    focus_depth_disagreement: torch.Tensor | None = None
    measurement_residual: torch.Tensor | None = None
    prior_disagreement: torch.Tensor | None = None
    model_mismatch_score: torch.Tensor | None = None
    abstention_evidence: torch.Tensor | None = None
    diagnostic_logits: torch.Tensor | None = None


# Backward-compatible alias for checkpoints and public imports.
PhysicalVerificationTrace = FocalConsistencyTrace
