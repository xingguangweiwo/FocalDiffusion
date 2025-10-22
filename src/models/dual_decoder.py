"""Dual-output decoder used by FocalDiffusion."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Simple convolutional block with SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, out_channels // 16), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherited
        return self.block(x)


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

        self.shared = nn.ModuleList([
            _ConvBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])

        depth_layers: List[nn.Module] = []
        rgb_layers: List[nn.Module] = []
        for _ in range(2):
            depth_layers.append(_ConvBlock(dims[-1], dims[-1]))
            rgb_layers.append(_ConvBlock(dims[-1], dims[-1]))

        self.depth_branch = nn.Sequential(*depth_layers)
        self.rgb_branch = nn.Sequential(*rgb_layers)

        self.depth_head = nn.Conv2d(dims[-1], out_channels_depth, kernel_size=3, padding=1)
        self.rgb_head = nn.Conv2d(dims[-1], self.out_channels_rgb, kernel_size=3, padding=1)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return depth logits and RGB latents for the given diffusion latents."""

        features = latents
        skip: Optional[torch.Tensor] = None

        for layer in self.shared:
            features = layer(features)
            if self.use_skip_connections:
                skip = features if skip is None else skip + features

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
