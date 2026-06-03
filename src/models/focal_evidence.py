"""Local focal-evidence posterior head for physics-gated depth fusion."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalEvidenceHead(nn.Module):
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
        self.tau_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.score_head = nn.Conv2d(hidden, 1, 1)

    def _normalize_focus_distances(self, focus_distances: torch.Tensor) -> torch.Tensor:
        tau_min = focus_distances.min(dim=1, keepdim=True).values
        tau_max = focus_distances.max(dim=1, keepdim=True).values
        return (focus_distances - tau_min) / (tau_max - tau_min + self.eps)

    def forward(self, focal_stack: torch.Tensor, focus_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        if focal_stack.dim() != 5:
            raise ValueError(f"focal_stack must have shape [B, N, C, H, W], got {tuple(focal_stack.shape)}")
        if focus_distances.dim() == 1:
            focus_distances = focus_distances.unsqueeze(0)

        B, N, C, H, W = focal_stack.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}.")
        if focus_distances.shape != (B, N):
            raise ValueError(
                f"focus_distances must have shape {(B, N)}, got {tuple(focus_distances.shape)}"
            )

        tau = self._normalize_focus_distances(focus_distances)

        x = focal_stack.reshape(B * N, C, H, W)
        if self.use_highpass:
            low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
            hp = x - low
        else:
            hp = x
        hp = hp.reshape(B, N, C, H, W)

        diff = torch.zeros_like(hp)
        diff[:, 1:] = hp[:, 1:] - hp[:, :-1]

        physics_input = torch.cat([hp, diff], dim=2).reshape(B * N, 2 * C, H, W)
        features = self.local_encoder(physics_input).reshape(B, N, self.hidden, H, W)
        tau_embed = self.tau_mlp(tau.unsqueeze(-1)).to(dtype=features.dtype)
        features = features + tau_embed[:, :, :, None, None]
        features = features.reshape(B * N, self.hidden, H, W)

        temperature = max(float(self.temperature), self.eps)
        focus_logits = self.score_head(features).reshape(B, N, H, W)
        focus_posterior = torch.softmax(focus_logits / temperature, dim=1)
        depth_focus_norm = (focus_posterior * tau[:, :, None, None]).sum(dim=1, keepdim=True)

        entropy = -(focus_posterior * torch.log(focus_posterior + self.eps)).sum(dim=1, keepdim=True)
        entropy = (entropy / math.log(max(N, 2))).clamp(0.0, 1.0)
        focus_peakiness = (1.0 - entropy).clamp(0.0, 1.0)

        return {
            "focus_logits": focus_logits,
            "focus_posterior": focus_posterior,
            "depth_focus_norm": depth_focus_norm,
            "focus_entropy": entropy,
            "focus_peakiness": focus_peakiness,
            "focus_reliability": focus_peakiness,  # Compatibility alias only; not calibrated reliability.
            "focus_coordinates": tau,
            # Compatibility aliases for older scripts/tests.
            "focus_prob": focus_posterior,
            "depth_focus": depth_focus_norm,
            "tau": tau,
        }


def build_support_inputs(
    focus_posterior: torch.Tensor,
    focus_entropy: torch.Tensor,
    depth_focus_norm: torch.Tensor,
    depth_prior_norm: torch.Tensor,
    uncertainty_decoder: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build lightweight physical support inputs.

    Args:
        focus_posterior: Per-pixel posterior over focus planes with shape [B, N, H, W].
        focus_entropy: Normalized posterior entropy with shape [B, 1, H, W].
        depth_focus_norm: Focus-derived normalized depth with shape [B, 1, H, W].
        depth_prior_norm: Decoder-prior normalized depth with shape [B, 1, H, W].
        uncertainty_decoder: Decoder uncertainty with shape [B, 1, H, W].

    Returns:
        A 5-channel support tensor and a dictionary of diagnostic support maps.
    """

    if focus_posterior.shape[1] < 2:
        raise ValueError("focus_posterior must contain at least two focal planes to compute posterior_margin.")

    focus_peakiness = (1.0 - focus_entropy).clamp(0.0, 1.0)
    top2 = torch.topk(focus_posterior, k=2, dim=1).values
    posterior_margin = top2[:, 0:1] - top2[:, 1:2]
    depth_disagreement = torch.abs(depth_focus_norm - depth_prior_norm)

    support_inputs = torch.cat(
        [
            focus_entropy,
            focus_peakiness,
            posterior_margin,
            depth_disagreement,
            uncertainty_decoder,
        ],
        dim=1,
    )
    support_maps = {
        "focus_peakiness": focus_peakiness,
        "posterior_margin": posterior_margin,
        "depth_disagreement": depth_disagreement,
    }
    return support_inputs, support_maps


class PhysicalSupportHead(nn.Module):
    """Tiny head that calibrates focus/prior gates and uncertainty from physical support maps."""

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
        gate_focus = gate[:, 0:1]
        gate_prior = gate[:, 1:2]
        gate_abstain = gate[:, 2:3]

        uncertainty_final = torch.sigmoid(uncertainty_logit)
        physical_support = 1.0 - uncertainty_final

        return {
            "gate_logits": gate_logits,
            "gate_focus": gate_focus,
            "gate_prior": gate_prior,
            "gate_abstain": gate_abstain,
            "uncertainty_final": uncertainty_final,
            "physical_support": physical_support,
        }
