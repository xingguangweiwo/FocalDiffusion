"""Local focal-evidence posterior head for physics-gated depth fusion."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalEvidenceEncoder(nn.Module):
    """Predict a per-pixel posterior distribution over focal planes.

    The head intentionally uses local high-pass and adjacent-focus differences,
    preserving the structure of focus-measure methods while learning the local
    scoring function.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 48,
        temperature: float = 0.07,
        use_highpass: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.temperature = temperature
        self.use_highpass = use_highpass
        self.eps = eps

        self.local_encoder = nn.Sequential(
            nn.Conv2d(2 * in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.focal_distance_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.posterior_logit_head = nn.Conv2d(hidden, 1, 1)

    def _normalize_focal_plane_distances(self, focal_plane_distances: torch.Tensor) -> torch.Tensor:
        tau_min = focal_plane_distances.min(dim=1, keepdim=True).values
        tau_max = focal_plane_distances.max(dim=1, keepdim=True).values
        return (focal_plane_distances - tau_min) / (tau_max - tau_min + self.eps)

    def forward(self, focal_stack: torch.Tensor, focal_plane_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        if focal_stack.dim() != 5:
            raise ValueError(f"focal_stack must have shape [B, N, C, H, W], got {tuple(focal_stack.shape)}")
        if focal_plane_distances.dim() == 1:
            focal_plane_distances = focal_plane_distances.unsqueeze(0)

        B, N, C, H, W = focal_stack.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}.")
        if focal_plane_distances.shape != (B, N):
            raise ValueError(
                f"focal_plane_distances must have shape {(B, N)}, got {tuple(focal_plane_distances.shape)}"
            )

        focal_coordinates = self._normalize_focal_plane_distances(focal_plane_distances)

        x = focal_stack.reshape(B * N, C, H, W)
        if self.use_highpass:
            low_frequency = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
            high_frequency = x - low_frequency
        else:
            high_frequency = x
        high_frequency = high_frequency.reshape(B, N, C, H, W)

        adjacent_focus_difference = torch.zeros_like(high_frequency)
        adjacent_focus_difference[:, 1:] = high_frequency[:, 1:] - high_frequency[:, :-1]

        evidence_input = torch.cat([high_frequency, adjacent_focus_difference], dim=2).reshape(B * N, 2 * C, H, W)
        features = self.local_encoder(evidence_input).reshape(B, N, self.hidden, H, W)
        focal_distance_embedding = self.focal_distance_mlp(focal_coordinates.unsqueeze(-1)).to(dtype=features.dtype)
        features = features + focal_distance_embedding[:, :, :, None, None]
        features = features.reshape(B * N, self.hidden, H, W)

        temperature = max(float(self.temperature), self.eps)
        focal_logits = self.posterior_logit_head(features).reshape(B, N, H, W)
        focal_posterior = torch.softmax(focal_logits / temperature, dim=1)
        focal_depth_canonical = (focal_posterior * focal_coordinates[:, :, None, None]).sum(dim=1, keepdim=True)

        focal_entropy = -(focal_posterior * torch.log(focal_posterior + self.eps)).sum(dim=1, keepdim=True)
        focal_entropy = (focal_entropy / math.log(max(N, 2))).clamp(0.0, 1.0)
        focal_peak_confidence = (1.0 - focal_entropy).clamp(0.0, 1.0)

        return {
            "focal_logits": focal_logits,
            "focal_posterior": focal_posterior,
            "focal_depth_canonical": focal_depth_canonical,
            "focal_entropy": focal_entropy,
            "focal_peak_confidence": focal_peak_confidence,
            "focal_coordinates": focal_coordinates,
        }


def build_physical_evidence_features(
    focal_posterior: torch.Tensor,
    focal_entropy: torch.Tensor,
    focal_depth_canonical: torch.Tensor,
    generated_depth_canonical: torch.Tensor,
    generative_uncertainty: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build physical-evidence support maps for focus/prior depth fusion.

    Args:
        focal_posterior: Per-pixel posterior over focal planes with shape [B, N, H, W].
        focal_entropy: Normalized posterior entropy with shape [B, 1, H, W].
        focal_depth_canonical: Focus-derived normalized depth with shape [B, 1, H, W].
        generated_depth_canonical: Decoder-prior normalized depth with shape [B, 1, H, W].
        generative_uncertainty: Decoder uncertainty with shape [B, 1, H, W].

    Returns:
        A 5-channel support tensor and diagnostic support maps.
    """

    if focal_posterior.shape[1] < 2:
        raise ValueError("focal_posterior must contain at least two focal planes to compute posterior_margin.")

    focal_peak_confidence = (1.0 - focal_entropy).clamp(0.0, 1.0)
    top2 = torch.topk(focal_posterior, k=2, dim=1).values
    posterior_margin = top2[:, 0:1] - top2[:, 1:2]
    depth_disagreement = torch.abs(focal_depth_canonical - generated_depth_canonical)

    support_inputs = torch.cat(
        [
            focal_entropy,
            focal_peak_confidence,
            posterior_margin,
            depth_disagreement,
            generative_uncertainty,
        ],
        dim=1,
    )
    support_maps = {
        "focal_peak_confidence": focal_peak_confidence,
        "posterior_margin": posterior_margin,
        "depth_disagreement": depth_disagreement,
    }
    return support_inputs, support_maps


def decode_metric_depth_from_focal_posterior(
    focal_posterior: torch.Tensor,
    focal_plane_distances: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert a focal-plane posterior into metric depth by diopter interpolation.

    Args:
        focal_posterior: Per-pixel focal-plane posterior with shape [B, N, H, W].
        focal_plane_distances: Physical focus distances in meters with shape [B, N] or [N].
        eps: Numerical clamp used to avoid division by zero.

    Returns:
        Metric depth in meters with shape [B, 1, H, W].

    Note:
        This conversion is physically meaningful only when ``focal_plane_distances`` are
        calibrated metric distances. Index-spaced focal coordinates should remain
        normalized/relative outputs.
    """

    if focal_posterior.dim() != 4:
        raise ValueError(
            f"focal_posterior must have shape [B, N, H, W], got {tuple(focal_posterior.shape)}"
        )

    if focal_plane_distances.dim() == 1:
        focal_plane_distances = focal_plane_distances.unsqueeze(0)
    if focal_plane_distances.dim() != 2:
        raise ValueError(
            f"focal_plane_distances must have shape [B, N] or [N], got {tuple(focal_plane_distances.shape)}"
        )

    batch, planes = focal_posterior.shape[:2]
    if focal_plane_distances.shape[1] != planes:
        raise ValueError(
            f"focal_plane_distances must contain {planes} planes, got {focal_plane_distances.shape[1]}"
        )
    if focal_plane_distances.shape[0] == 1 and batch != 1:
        focal_plane_distances = focal_plane_distances.expand(batch, -1)
    elif focal_plane_distances.shape[0] != batch:
        raise ValueError(
            f"focal_plane_distances batch must be 1 or {batch}, got {focal_plane_distances.shape[0]}"
        )

    focal_plane_distances = focal_plane_distances.to(device=focal_posterior.device, dtype=focal_posterior.dtype)
    diopters = 1.0 / focal_plane_distances.clamp(min=eps)
    diopter_pred = (focal_posterior * diopters[:, :, None, None]).sum(dim=1, keepdim=True)
    return 1.0 / diopter_pred.clamp(min=eps)


class PhysicalEvidenceEstimator(nn.Module):
    """Estimate focus/prior/abstention gates and fusion uncertainty from support maps."""

    def __init__(self, in_channels: int = 5, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 4, kernel_size=1),
        )

    def forward(self, support_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.net(support_inputs)
        gate_logits = raw[:, :3]
        uncertainty_logit = raw[:, 3:4]

        gate = torch.softmax(gate_logits, dim=1)
        focal_evidence_weight = gate[:, 0:1]
        generative_prior_weight = gate[:, 1:2]
        abstention_weight = gate[:, 2:3]

        uncertainty_final = torch.sigmoid(uncertainty_logit)
        physical_evidence_support = 1.0 - uncertainty_final

        return {
            "gate_logits": gate_logits,
            "focal_evidence_weight": focal_evidence_weight,
            "generative_prior_weight": generative_prior_weight,
            "abstention_weight": abstention_weight,
            "uncertainty_final": uncertainty_final,
            "physical_evidence_support": physical_evidence_support,
        }
