"""Core modules for focal-sweep stack conditioning."""

from __future__ import annotations

from typing import Dict
import math

import torch
import torch.nn as nn


class FocalSweepEncoder(nn.Module):
    """Learned focal-axis encoder using patch tokens and attention across focal sweep."""

    def __init__(self, feature_dim: int = 512, patch_size: int = 8, num_heads: int = 8, depth: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.patch_embed = nn.Conv2d(3, feature_dim, kernel_size=patch_size, stride=patch_size)
        self.tau_embed = nn.Sequential(nn.Linear(16, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4, batch_first=True)
            for _ in range(depth)
        ])
        self.surface_query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.query_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

    @staticmethod
    def normalize_focus_distances(focus_distances: torch.Tensor) -> torch.Tensor:
        tau_min = focus_distances.min(dim=1, keepdim=True).values
        tau_max = focus_distances.max(dim=1, keepdim=True).values
        return (focus_distances - tau_min) / (tau_max - tau_min + 1e-6)

    @staticmethod
    def fourier_embed(tau: torch.Tensor, bands: int = 8) -> torch.Tensor:
        freq = (2.0 ** torch.arange(bands, device=tau.device, dtype=tau.dtype)).view(1, 1, -1)
        x = tau.unsqueeze(-1) * freq * math.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def forward(self, focal_stack: torch.Tensor, focus_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _, H, W = focal_stack.shape
        x = focal_stack.reshape(B * N, 3, H, W)
        tokens_2d = self.patch_embed(x)
        h, w = tokens_2d.shape[-2:]
        tokens = tokens_2d.flatten(2).transpose(1, 2).view(B, N, h * w, self.feature_dim)

        tau = self.normalize_focus_distances(focus_distances)
        tau_tokens = self.tau_embed(self.fourier_embed(tau)).unsqueeze(2)
        tokens = tokens + tau_tokens

        tokens = tokens.permute(0, 2, 1, 3).contiguous().view(B * h * w, N, self.feature_dim)
        for layer in self.layers:
            tokens = layer(tokens)

        query = self.surface_query.expand(tokens.shape[0], -1, -1)
        fused, attn = self.query_attn(query, tokens, tokens, need_weights=True)
        fused = fused.squeeze(1).view(B, h, w, self.feature_dim).permute(0, 3, 1, 2).contiguous()
        attn = attn.squeeze(1).view(B, h, w, N).permute(0, 3, 1, 2).contiguous()
        frame_weights = attn.mean(dim=(-2, -1))

        return {"fused_features": fused, "tau": tau, "attention_weights": frame_weights, "temporal_attention_maps": attn}


class FocalStackProcessor(nn.Module):
    """Clean focal-stack processor wrapping the focal-sweep encoder."""

    def __init__(
        self,
        feature_dim: int = 512,
        num_scales: int = 4,
        max_sequence_length: int = 100,
        dropout: float = 0.1,
        focal_encoder_type: str = "focal_sweep",
        patch_size: int = 8,
        focal_attention_heads: int = 8,
        focal_attention_depth: int = 2,
    ):
        super().__init__()
        del num_scales, dropout
        if focal_encoder_type != "focal_sweep":
            raise ValueError("Only focal_sweep is supported in the cleaned FSDiffusion implementation.")
        self.feature_dim = feature_dim
        self.max_sequence_length = max_sequence_length
        self.focal_encoder_type = focal_encoder_type
        self.focal_sweep_encoder = FocalSweepEncoder(
            feature_dim=feature_dim,
            patch_size=patch_size,
            num_heads=focal_attention_heads,
            depth=focal_attention_depth,
        )

    def forward(self, focal_stack: torch.Tensor, focus_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a focal stack into SD3 conditioning features."""
        B, N, C, H, W = focal_stack.shape
        del B, C, H, W
        if N > self.max_sequence_length:
            raise ValueError(f"Sequence length {N} exceeds maximum {self.max_sequence_length}")
        return self.focal_sweep_encoder(focal_stack, focus_distances)
