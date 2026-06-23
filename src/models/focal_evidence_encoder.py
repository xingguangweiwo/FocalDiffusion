"""Local focal-evidence posterior head for physics-gated depth fusion."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.image_utils import canonical_focal_coordinates


class FocusLikelihoodEstimator(nn.Module):
    """Predict focus likelihood over focal planes with reliability diagnostics."""

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
        coords, _ = canonical_focal_coordinates(
            focal_plane_distances,
            batch_size=focal_plane_distances.shape[0],
            coordinate_type="distance",
            eps=self.eps,
        )
        return coords

    def forward(
        self,
        focal_stack: torch.Tensor,
        focal_plane_distances: torch.Tensor,
        focal_plane_valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
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

        focal_coordinates, focal_plane_valid_mask = canonical_focal_coordinates(
            focal_plane_distances,
            batch_size=batch_size,
            coordinate_type="distance",
            focal_plane_valid_mask=focal_plane_valid_mask,
            eps=self.eps,
        )
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

        focal_logits = self.posterior_logit_head(features).reshape(batch_size, num_planes, height, width)
        texture = high_frequency.detach().abs().mean(dim=(1, 2), keepdim=True).reshape(batch_size, 1, height, width)
        texture_confidence = (texture / (texture.mean(dim=(-2, -1), keepdim=True) + self.eps)).clamp(0.0, 1.0)
        adaptive_temperature = max(float(self.temperature), self.eps) * (1.0 + (1.0 - texture_confidence.detach()))
        masked_logits = focal_logits.masked_fill(~focal_plane_valid_mask[:, :, None, None], -torch.finfo(focal_logits.dtype).max / 4)
        focal_posterior = torch.softmax(masked_logits / adaptive_temperature, dim=1)
        focal_posterior = focal_posterior * focal_plane_valid_mask[:, :, None, None].to(dtype=focal_posterior.dtype)
        focal_posterior = focal_posterior / focal_posterior.sum(dim=1, keepdim=True).clamp(min=self.eps)
        focal_depth_canonical = (focal_posterior * focal_coordinates[:, :, None, None]).sum(dim=1, keepdim=True)
        focal_entropy = -(focal_posterior * torch.log(focal_posterior + self.eps)).sum(dim=1, keepdim=True)
        valid_count = focal_plane_valid_mask.sum(dim=1).clamp(min=2).to(dtype=focal_entropy.dtype)
        focal_entropy = (focal_entropy / torch.log(valid_count.view(batch_size, 1, 1, 1))).clamp(0.0, 1.0)
        top2 = torch.topk(focal_posterior, k=min(2, num_planes), dim=1).values
        posterior_margin = top2[:, 0:1] - (top2[:, 1:2] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0:1]))
        primary_index = focal_posterior.detach().argmax(dim=1, keepdim=True)
        plane_index = torch.arange(num_planes, device=focal_posterior.device).view(1, num_planes, 1, 1)
        secondary_mask = (plane_index - primary_index).abs() > 1
        secondary_mask = secondary_mask & focal_plane_valid_mask[:, :, None, None]
        secondary = focal_posterior.masked_fill(~secondary_mask, 0.0)
        secondary_peak_probability = secondary.amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        valid_indices = torch.arange(num_planes, device=focal_posterior.device).view(1, num_planes)
        first_valid = torch.where(focal_plane_valid_mask, valid_indices, num_planes).amin(dim=1)
        last_valid = torch.where(focal_plane_valid_mask, valid_indices, -1).amax(dim=1)
        boundary_mask = (valid_indices == first_valid[:, None]) | (valid_indices == last_valid[:, None])
        boundary_peak_probability = (focal_posterior * boundary_mask[:, :, None, None].to(dtype=focal_posterior.dtype)).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        plane_availability = focal_plane_valid_mask.float().mean(dim=1).view(batch_size, 1, 1, 1).expand_as(focal_entropy)
        peak_confidence = (1.0 - focal_entropy).clamp(0.0, 1.0)

        return {
            "focal_logits": focal_logits,
            "focal_posterior": focal_posterior,
            "focus_posterior": focal_posterior,
            "focal_depth_canonical": focal_depth_canonical,
            "focus_depth": focal_depth_canonical,
            "focus_depth_canonical": focal_depth_canonical,
            "focal_entropy": focal_entropy,
            "normalized_entropy": focal_entropy,
            "focus_entropy": focal_entropy,
            "focal_peak_confidence": peak_confidence,
            "posterior_margin": posterior_margin,
            "secondary_peak_probability": secondary_peak_probability,
            "texture_confidence": texture_confidence,
            "plane_availability": plane_availability,
            "boundary_peak_probability": boundary_peak_probability,
            # Compatibility keys for existing callers; active code uses the names above.
            "multimodality_score": secondary_peak_probability,
            "focus_coverage_confidence": plane_availability,
            "focal_coordinates": focal_coordinates,
            "focal_plane_valid_mask": focal_plane_valid_mask,
            "posterior_temperature": adaptive_temperature.expand_as(focal_entropy),
        }


def build_reliability_features(
    focal_posterior: torch.Tensor,
    focal_entropy: torch.Tensor | None = None,
    focal_depth_canonical: torch.Tensor | None = None,
    prior_depth_canonical: torch.Tensor | None = None,
    generative_uncertainty: torch.Tensor | None = None,
    *,
    normalized_entropy: torch.Tensor | None = None,
    posterior_margin: torch.Tensor | None = None,
    secondary_peak_probability: torch.Tensor | None = None,
    texture_confidence: torch.Tensor | None = None,
    plane_availability: torch.Tensor | None = None,
    boundary_peak_probability: torch.Tensor | None = None,
    focus_depth: torch.Tensor | None = None,
    prior_depth: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build reliability features for identifiable focus/prior selection."""

    if focal_posterior.dim() != 4:
        raise ValueError(f"focal_posterior must have shape [B, N, H, W], got {tuple(focal_posterior.shape)}")
    if focal_posterior.shape[1] < 2:
        raise ValueError("focal_posterior must contain at least two focal planes to compute reliability features.")
    focus_depth = focus_depth if focus_depth is not None else focal_depth_canonical
    prior_depth = prior_depth if prior_depth is not None else prior_depth_canonical
    entropy = normalized_entropy if normalized_entropy is not None else focal_entropy
    if focus_depth is None or prior_depth is None or generative_uncertainty is None or entropy is None:
        raise ValueError("focus/prior depth, generative_uncertainty, and entropy are required.")

    batch, _, height, width = focal_posterior.shape
    expected = (batch, 1, height, width)
    if posterior_margin is None:
        top2 = torch.topk(focal_posterior, k=2, dim=1).values
        posterior_margin = top2[:, 0:1] - top2[:, 1:2]
    if secondary_peak_probability is None:
        primary = focal_posterior.detach().argmax(dim=1, keepdim=True)
        idx = torch.arange(focal_posterior.shape[1], device=focal_posterior.device).view(1, -1, 1, 1)
        secondary_peak_probability = focal_posterior.masked_fill((idx - primary).abs() <= 1, 0.0).amax(dim=1, keepdim=True)
    default_ones = torch.ones(expected, device=focal_posterior.device, dtype=focal_posterior.dtype)
    default_zeros = torch.zeros(expected, device=focal_posterior.device, dtype=focal_posterior.dtype)
    texture_confidence = default_ones if texture_confidence is None else texture_confidence
    plane_availability = default_ones if plane_availability is None else plane_availability
    boundary_peak_probability = default_zeros if boundary_peak_probability is None else boundary_peak_probability

    values = {
        "normalized_entropy": entropy,
        "posterior_margin": posterior_margin,
        "secondary_peak_probability": secondary_peak_probability,
        "texture_confidence": texture_confidence,
        "plane_availability": plane_availability,
        "boundary_peak_probability": boundary_peak_probability,
        "focus_depth": focus_depth,
        "prior_depth": prior_depth,
        "generative_uncertainty": generative_uncertainty,
    }
    for name, value in values.items():
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")

    depth_disagreement = torch.abs(focus_depth - prior_depth).clamp(0.0, 1.0)
    support_inputs = torch.cat([
        entropy.clamp(0.0, 1.0),
        posterior_margin.clamp(0.0, 1.0),
        secondary_peak_probability.clamp(0.0, 1.0),
        texture_confidence.clamp(0.0, 1.0),
        plane_availability.clamp(0.0, 1.0),
        boundary_peak_probability.clamp(0.0, 1.0),
        depth_disagreement,
        generative_uncertainty.clamp(0.0, 1.0),
    ], dim=1)
    support_maps = {
        "normalized_entropy": entropy.clamp(0.0, 1.0),
        "focal_peak_confidence": (1.0 - entropy).clamp(0.0, 1.0),
        "posterior_margin": posterior_margin.clamp(0.0, 1.0),
        "secondary_peak_probability": secondary_peak_probability.clamp(0.0, 1.0),
        "texture_confidence": texture_confidence.clamp(0.0, 1.0),
        "plane_availability": plane_availability.clamp(0.0, 1.0),
        "boundary_peak_probability": boundary_peak_probability.clamp(0.0, 1.0),
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


class ReliabilityFusionHead(nn.Module):
    """Estimate focus/prior mixture, independent abstention, and error scale."""

    def __init__(self, in_channels: int = 8, hidden: int = 16) -> None:
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
        mixture_logits = raw[:, :2]
        abstention_logit = raw[:, 2:3]
        depth_error_log_scale = raw[:, 3:4]

        mixture = torch.softmax(mixture_logits, dim=1)
        abstention = torch.sigmoid(abstention_logit)
        depth_error_scale = F.softplus(depth_error_log_scale) + 1e-6
        uncertainty_final = (depth_error_scale / (1.0 + depth_error_scale)).clamp(0.0, 1.0)

        return {
            "mixture_logits": mixture_logits,
            "gate_logits": mixture_logits,
            "focal_evidence_weight": mixture[:, 0:1],
            "generative_prior_weight": mixture[:, 1:2],
            "abstention_logit": abstention_logit,
            "abstention_weight": abstention,
            "abstention_probability": abstention,
            "depth_error_log_scale": depth_error_log_scale,
            "depth_error_scale": depth_error_scale,
            "uncertainty_final": uncertainty_final,
            "physical_evidence_support": 1.0 - uncertainty_final,
            "reliability_score": 1.0 - uncertainty_final,
        }


# Backward-compatible aliases for existing checkpoints and imports.
FocalEvidenceEncoder = FocusLikelihoodEstimator
class PhysicalEvidenceEstimator(nn.Module):
    """Backward-compatible three-way support head retained at the compatibility boundary."""

    def __init__(self, in_channels: int = 5, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 4, kernel_size=1),
        )

    def forward(self, support_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        expected_channels = self.net[0].in_channels
        if support_inputs.dim() != 4:
            raise ValueError(f"support_inputs must have shape [B, C, H, W], got {tuple(support_inputs.shape)}")
        if support_inputs.shape[1] != expected_channels:
            raise ValueError(f"support_inputs must have {expected_channels} channels, got {support_inputs.shape[1]}")
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
            "abstention_probability": gate[:, 2:3],
            "uncertainty_final": uncertainty_final,
            "physical_evidence_support": 1.0 - uncertainty_final,
            "reliability_score": 1.0 - uncertainty_final,
        }

def build_physical_evidence_features(
    focal_posterior: torch.Tensor,
    focal_entropy: torch.Tensor,
    focal_depth_canonical: torch.Tensor,
    generated_depth_canonical: torch.Tensor | None = None,
    generative_uncertainty: torch.Tensor | None = None,
    *,
    focus_depth_canonical: torch.Tensor | None = None,
    prior_depth_canonical: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Backward-compatible five-channel wrapper for legacy callers."""
    focus_depth = focus_depth_canonical if focus_depth_canonical is not None else focal_depth_canonical
    prior_depth = prior_depth_canonical if prior_depth_canonical is not None else generated_depth_canonical
    if prior_depth is None or generative_uncertainty is None:
        raise ValueError("prior/generated depth and generative_uncertainty are required.")
    batch, _, height, width = focal_posterior.shape
    expected = (batch, 1, height, width)
    for name, value in {
        "focal_entropy": focal_entropy,
        "focal_depth_canonical": focus_depth,
        "generated_depth_canonical": prior_depth,
        "generative_uncertainty": generative_uncertainty,
    }.items():
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    top2 = torch.topk(focal_posterior, k=2, dim=1).values
    posterior_margin = top2[:, 0:1] - top2[:, 1:2]
    depth_disagreement = torch.abs(focus_depth - prior_depth)
    focal_peak_confidence = (1.0 - focal_entropy).clamp(0.0, 1.0)
    support_inputs = torch.cat([focal_entropy, focal_peak_confidence, posterior_margin, depth_disagreement, generative_uncertainty], dim=1)
    return support_inputs, {
        "focal_peak_confidence": focal_peak_confidence,
        "posterior_margin": posterior_margin,
        "depth_disagreement": depth_disagreement,
    }
