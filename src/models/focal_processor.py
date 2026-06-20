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
        self.rank_embed = nn.Sequential(nn.Linear(16, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim))
        self.type_embed = nn.Embedding(32, feature_dim)
        self.physical_embed = nn.Sequential(nn.Linear(1, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4, batch_first=True)
            for _ in range(depth)
        ])
        self.surface_query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.query_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

    @staticmethod
    def normalize_focal_plane_distances(focal_plane_distances: torch.Tensor) -> torch.Tensor:
        tau_min = focal_plane_distances.min(dim=1, keepdim=True).values
        tau_max = focal_plane_distances.max(dim=1, keepdim=True).values
        return (focal_plane_distances - tau_min) / (tau_max - tau_min + 1e-6)

    @staticmethod
    def fourier_embed(tau: torch.Tensor, bands: int = 8) -> torch.Tensor:
        freq = (2.0 ** torch.arange(bands, device=tau.device, dtype=tau.dtype)).view(1, 1, -1)
        x = tau.unsqueeze(-1) * freq * math.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def forward(
        self,
        focal_stack: torch.Tensor,
        focal_plane_distances: torch.Tensor | None = None,
        *,
        focal_plane_ranks: torch.Tensor | None = None,
        focal_plane_canonical_coordinates: torch.Tensor | None = None,
        coordinate_type_id: torch.Tensor | None = None,
        physical_coordinates: torch.Tensor | None = None,
        physical_coordinate_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, _, H, W = focal_stack.shape
        x = focal_stack.reshape(B * N, 3, H, W)
        tokens_2d = self.patch_embed(x)
        h, w = tokens_2d.shape[-2:]
        tokens = tokens_2d.flatten(2).transpose(1, 2).view(B, N, h * w, self.feature_dim)

        if focal_plane_canonical_coordinates is None:
            if focal_plane_distances is None:
                focal_plane_canonical_coordinates = torch.linspace(0, 1, N, device=focal_stack.device, dtype=focal_stack.dtype).unsqueeze(0).expand(B, -1)
            else:
                focal_plane_canonical_coordinates = self.normalize_focal_plane_distances(focal_plane_distances)
        tau = focal_plane_canonical_coordinates.to(device=focal_stack.device, dtype=focal_stack.dtype)
        if tau.dim() == 1:
            tau = tau.unsqueeze(0).expand(B, -1)
        rank = focal_plane_ranks.to(device=focal_stack.device, dtype=focal_stack.dtype) if focal_plane_ranks is not None else torch.linspace(0, 1, N, device=focal_stack.device, dtype=focal_stack.dtype).unsqueeze(0).expand(B, -1)
        if rank.dim() == 1:
            rank = rank.unsqueeze(0).expand(B, -1)
        tau_tokens = self.tau_embed(self.fourier_embed(tau)) + self.rank_embed(self.fourier_embed(rank))
        if coordinate_type_id is not None:
            type_ids = coordinate_type_id.to(device=focal_stack.device).long().view(-1)
            if type_ids.numel() == 1:
                type_ids = type_ids.expand(B)
            tau_tokens = tau_tokens + self.type_embed(type_ids).unsqueeze(1)
        if physical_coordinates is not None and physical_coordinate_mask is not None:
            phys = physical_coordinates.to(device=focal_stack.device, dtype=focal_stack.dtype)
            mask = physical_coordinate_mask.to(device=focal_stack.device, dtype=focal_stack.dtype)
            if phys.dim() == 1: phys = phys.unsqueeze(0).expand(B, -1)
            if mask.dim() == 1: mask = mask.unsqueeze(0).expand(B, -1)
            tau_tokens = tau_tokens + self.physical_embed(self.normalize_focal_plane_distances(phys).unsqueeze(-1)) * mask.unsqueeze(-1)
        tau_tokens = tau_tokens.unsqueeze(2)
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
            raise ValueError("Only focal_sweep is supported in the cleaned FocalStackGeneration implementation.")
        self.feature_dim = feature_dim
        self.max_sequence_length = max_sequence_length
        self.focal_encoder_type = focal_encoder_type
        self.focal_sweep_encoder = FocalSweepEncoder(
            feature_dim=feature_dim,
            patch_size=patch_size,
            num_heads=focal_attention_heads,
            depth=focal_attention_depth,
        )

    def forward(self, focal_stack: torch.Tensor, focal_plane_distances: torch.Tensor | None = None, **coordinate_kwargs) -> Dict[str, torch.Tensor]:
        """Process a focal stack into SD3 conditioning features."""
        B, N, C, H, W = focal_stack.shape
        del B, C, H, W
        if N > self.max_sequence_length:
            raise ValueError(f"Sequence length {N} exceeds maximum {self.max_sequence_length}")
        return self.focal_sweep_encoder(focal_stack, focal_plane_distances, **coordinate_kwargs)
