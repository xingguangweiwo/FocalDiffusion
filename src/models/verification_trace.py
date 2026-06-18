"""Typed containers for physical verification trace outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PhysicalVerificationTrace:
    """Batch-first, per-pixel physical evidence summary for FocalTrace.

    All score fields are tensors shaped ``[B, 1, H, W]`` except
    ``verdict_logits``, which is shaped ``[B, 3, H, W]`` for conservative
    ``support``, ``conflict``, and ``invalid`` logits.
    """

    focus_peak: torch.Tensor
    focus_margin: torch.Tensor
    focus_entropy: torch.Tensor
    operator_agreement: torch.Tensor
    texture_confidence: torch.Tensor
    depth_focus_discrepancy: torch.Tensor
    defocus_residual: torch.Tensor
    refocus_residual: torch.Tensor
    focus_support: torch.Tensor
    generation_support: torch.Tensor
    conflict_score: torch.Tensor
    invalid_score: torch.Tensor
    verdict_logits: torch.Tensor
