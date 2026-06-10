"""Local focal-evidence posterior head for physics-gated depth fusion."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalEvidenceEncoder(nn.Module):
    """Predict a per-pixel posterior distribution over focal planes."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 48,
        temperature: float = 0.07,
        use_highpass: bool = True,
        eps: float = 1e-6,
    ) -> None:
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
        if not torch.isfinite(focal_stack).all():
            raise ValueError("focal_stack must contain only finite values.")
        if focal_plane_distances.dim() == 1:
            focal_plane_distances = focal_plane_distances.unsqueeze(0)
        if focal_plane_distances.dim() != 2:
            raise ValueError(
                "focal_plane_distances must have shape [B, N] or [N], "
                f"got {tuple(focal_plane_distances.shape)}"
            )

        batch_size, num_planes, channels, height, width = focal_stack.shape
        if num_planes < 1:
            raise ValueError("focal_stack must contain at least one focal plane.")
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}.")
        if focal_plane_distances.shape[0] == 1 and batch_size != 1:
            focal_plane_distances = focal_plane_distances.expand(batch_size, -1)
        if focal_plane_distances.shape != (batch_size, num_planes):
            raise ValueError(
                "focal_plane_distances must match the focal stack batch and plane dimensions: "
                f"expected {(batch_size, num_planes)}, got {tuple(focal_plane_distances.shape)}"
            )
        if not torch.isfinite(focal_plane_distances).all():
            raise ValueError("focal_plane_distances must contain only finite values.")

        focal_coordinates = self._normalize_focal_plane_distances(focal_plane_distances)
        stacked = focal_stack.reshape(batch_size * num_planes, channels, height, width)
        if self.use_highpass:
            high_frequency = stacked - F.avg_pool2d(stacked, kernel_size=5, stride=1, padding=2)
        else:
            high_frequency = stacked
        high_frequency = high_frequency.reshape(batch_size, num_planes, channels, height, width)

        adjacent_focus_difference = torch.zeros_like(high_frequency)
        adjacent_focus_difference[:, 1:] = high_frequency[:, 1:] - high_frequency[:, :-1]

        evidence_input = torch.cat([high_frequency, adjacent_focus_difference], dim=2)
        evidence_input = evidence_input.reshape(batch_size * num_planes, 2 * channels, height, width)
        features = self.local_encoder(evidence_input).reshape(batch_size, num_planes, self.hidden, height, width)
        focal_distance_embedding = self.focal_distance_mlp(focal_coordinates.unsqueeze(-1)).to(dtype=features.dtype)
        features = features + focal_distance_embedding[:, :, :, None, None]
        features = features.reshape(batch_size * num_planes, self.hidden, height, width)

        temperature = max(float(self.temperature), self.eps)
        focal_logits = self.posterior_logit_head(features).reshape(batch_size, num_planes, height, width)
        focal_posterior = torch.softmax(focal_logits / temperature, dim=1)
        focal_depth_canonical = (focal_posterior * focal_coordinates[:, :, None, None]).sum(dim=1, keepdim=True)
        focal_entropy = -(focal_posterior * torch.log(focal_posterior + self.eps)).sum(dim=1, keepdim=True)
        focal_entropy = (focal_entropy / math.log(max(num_planes, 2))).clamp(0.0, 1.0)

        return {
            "focal_logits": focal_logits,
            "focal_posterior": focal_posterior,
            "focal_depth_canonical": focal_depth_canonical,
            "focal_entropy": focal_entropy,
            "focal_peak_confidence": (1.0 - focal_entropy).clamp(0.0, 1.0),
            "focal_coordinates": focal_coordinates,
        }


def build_physical_evidence_features(
    focal_posterior: torch.Tensor,
    focal_entropy: torch.Tensor,
    focal_depth_canonical: torch.Tensor,
    generated_depth_canonical: torch.Tensor,
    generative_uncertainty: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build compact support maps for focus/prior depth fusion."""

    if focal_posterior.dim() != 4:
        raise ValueError(
            f"focal_posterior must have shape [B, N, H, W], got {tuple(focal_posterior.shape)}"
        )
    if focal_posterior.shape[1] < 2:
        raise ValueError("focal_posterior must contain at least two focal planes to compute posterior_margin.")

    batch, _, height, width = focal_posterior.shape
    expected_map_shape = (batch, 1, height, width)
    for name, value in {
        "focal_entropy": focal_entropy,
        "focal_depth_canonical": focal_depth_canonical,
        "generated_depth_canonical": generated_depth_canonical,
        "generative_uncertainty": generative_uncertainty,
    }.items():
        if value.shape != expected_map_shape:
            raise ValueError(
                f"{name} must have shape {expected_map_shape}, got {tuple(value.shape)}"
            )
    for name, value in {
        "focal_posterior": focal_posterior,
        "focal_entropy": focal_entropy,
        "focal_depth_canonical": focal_depth_canonical,
        "generated_depth_canonical": generated_depth_canonical,
        "generative_uncertainty": generative_uncertainty,
    }.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")

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
    """Convert a calibrated focal-plane posterior into metric depth by diopter interpolation."""

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

    def __init__(self, in_channels: int = 5, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 4, kernel_size=1),
        )

    def forward(self, support_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        if support_inputs.dim() != 4:
            raise ValueError(
                f"support_inputs must have shape [B, C, H, W], got {tuple(support_inputs.shape)}"
            )
        expected_channels = self.net[0].in_channels
        if support_inputs.shape[1] != expected_channels:
            raise ValueError(
                f"support_inputs must have {expected_channels} channels, got {support_inputs.shape[1]}"
            )
        if not torch.isfinite(support_inputs).all():
            raise ValueError("support_inputs must contain only finite values.")

        raw = self.net(support_inputs)
        gate_logits = raw[:, :3]
        uncertainty_logit = raw[:, 3:4]

        gate = torch.softmax(gate_logits, dim=1)
        uncertainty_final = torch.sigmoid(uncertainty_logit)

        return {
            "gate_logits": gate_logits,
            "focal_evidence_weight": gate[:, 0:1],
            "generative_prior_weight": gate[:, 1:2],
            "abstention_weight": gate[:, 2:3],
            "uncertainty_final": uncertainty_final,
            "physical_evidence_support": 1.0 - uncertainty_final,
        }
