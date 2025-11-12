"""Dual-output decoder used by FocalDiffusion."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    """Residual convolutional block with optional dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=max(1, out_channels // 16), num_channels=out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=max(1, out_channels // 16), num_channels=out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherited
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + residual)


class DualOutputDecoder(nn.Module):
    """Decode SD3.5 latents into depth logits and RGB latents."""

    def __init__(
        self,
        in_channels: int,
        out_channels_depth: int = 1,
        out_channels_rgb: Optional[int] = None,
        hidden_dims: Optional[Sequence[int]] = None,
        use_skip_connections: bool = True,
        apply_latent_scaling: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = (256, 256, 128)

        self.use_skip_connections = use_skip_connections
        self.apply_latent_scaling = apply_latent_scaling
        self.out_channels_rgb = out_channels_rgb or in_channels

        dims: List[int] = [in_channels, *hidden_dims]

        dilations = [1, 2, 3]
        self.shared = nn.ModuleList([
            _ResidualBlock(dims[i], dims[i + 1], dilation=dilations[min(i, len(dilations) - 1)], dropout=0.1)
            for i in range(len(dims) - 1)
        ])

        self.context_projections = nn.ModuleList([
            nn.Conv2d(dims[i + 1], dims[-1], kernel_size=1) for i in range(len(dims) - 1)
        ])

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dims[-1], dims[-1], kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dims[-1], dims[-1], kernel_size=1),
        )

        branch_block = lambda: nn.Sequential(
            _ResidualBlock(dims[-1], dims[-1], dilation=1, dropout=0.1),
            _ResidualBlock(dims[-1], dims[-1], dilation=2, dropout=0.1),
        )

        self.depth_branch = branch_block()
        self.rgb_branch = branch_block()

        self.depth_head = nn.Conv2d(dims[-1], out_channels_depth, kernel_size=3, padding=1)
        self.rgb_head = nn.Conv2d(dims[-1], self.out_channels_rgb, kernel_size=3, padding=1)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return depth logits and RGB latents for the given diffusion latents."""

        features = latents
        skip: Optional[torch.Tensor] = None
        pyramid: List[torch.Tensor] = []

        for idx, layer in enumerate(self.shared):
            features = layer(features)
            pyramid.append(features)
            if self.use_skip_connections:
                skip = features if skip is None else skip + features

        # Fuse multi-dilation context back into the final representation
        fused_context = torch.zeros_like(pyramid[-1])
        for feat, proj in zip(pyramid, self.context_projections):
            context = proj(feat)
            if context.shape[-2:] != fused_context.shape[-2:]:
                context = F.interpolate(context, size=fused_context.shape[-2:], mode="bilinear", align_corners=False)
            fused_context = fused_context + context

        fused_context = fused_context / len(pyramid)
        global_feat = self.global_context(pyramid[-1])
        fused_context = fused_context + global_feat

        features = pyramid[-1] + fused_context

        depth_features = self.depth_branch(features)
        rgb_features = self.rgb_branch(features)

        if self.use_skip_connections and skip is not None:
            depth_features = depth_features + skip
            rgb_features = rgb_features + skip

        depth_logits = self.depth_head(depth_features)
        rgb_latents = self.rgb_head(rgb_features)

        if self.apply_latent_scaling:
            rgb_latents = rgb_latents * 0.18215

        return depth_logits, rgb_latents
