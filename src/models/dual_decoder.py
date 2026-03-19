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
    """Decode SD3.5 latents into AIF, depth and confidence with cross-task coupling."""

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

        # Feature-level cross-task coupling: AIF features guide depth structure recovery.
        self.aif_to_depth = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
        )
        self.aif_coupling_gate = nn.Sequential(
            nn.Conv2d(dims[-1] * 2, dims[-1], kernel_size=1),
            nn.Sigmoid(),
        )

        self.depth_head = nn.Conv2d(dims[-1], out_channels_depth, kernel_size=3, padding=1)
        self.confidence_head = nn.Conv2d(dims[-1], 1, kernel_size=3, padding=1)
        self.rgb_head = nn.Conv2d(dims[-1], self.out_channels_rgb, kernel_size=3, padding=1)

        # Confidence-guided depth refinement with AIF structure.
        self.depth_refine = nn.Sequential(
            _ResidualBlock(dims[-1] + in_channels + 2, dims[-1], dilation=1, dropout=0.0),
            _ResidualBlock(dims[-1], dims[-1], dilation=1, dropout=0.0),
            nn.Conv2d(dims[-1], out_channels_depth, kernel_size=3, padding=1),
        )

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return refined depth logits, RGB latents, and confidence map."""

        features = latents
        skip: Optional[torch.Tensor] = None
        pyramid: List[torch.Tensor] = []

        for layer in self.shared:
            features = layer(features)
            pyramid.append(features)
            if self.use_skip_connections:
                skip = features if skip is None else skip + features

        fused_context = torch.zeros_like(pyramid[-1])
        for feat, proj in zip(pyramid, self.context_projections):
            context = proj(feat)
            if context.shape[-2:] != fused_context.shape[-2:]:
                context = F.interpolate(context, size=fused_context.shape[-2:], mode="bilinear", align_corners=False)
            fused_context = fused_context + context

        fused_context = fused_context / len(pyramid)
        global_feat = self.global_context(pyramid[-1])
        features = pyramid[-1] + fused_context + global_feat

        depth_features = self.depth_branch(features)
        rgb_features = self.rgb_branch(features)

        if self.use_skip_connections and skip is not None:
            depth_features = depth_features + skip
            rgb_features = rgb_features + skip

        # Feature-level coupling from AIF to depth.
        aif_guidance = self.aif_to_depth(rgb_features)
        coupling_gate = self.aif_coupling_gate(torch.cat([depth_features, aif_guidance], dim=1))
        depth_features = depth_features + coupling_gate * aif_guidance

        depth_logits_coarse = self.depth_head(depth_features)
        confidence = torch.sigmoid(self.confidence_head(depth_features))
        rgb_latents = self.rgb_head(rgb_features)

        if self.apply_latent_scaling:
            rgb_latents = rgb_latents * 0.18215

        # Confidence-guided refinement: low confidence => stronger AIF-structure reliance.
        aif_edges = self._compute_edges(rgb_latents)
        refine_input = torch.cat(
            [
                depth_features,
                rgb_latents,
                aif_edges,
                1.0 - confidence,
            ],
            dim=1,
        )
        depth_delta = self.depth_refine(refine_input)
        depth_logits = depth_logits_coarse + (1.0 - confidence) * depth_delta

        return depth_logits, rgb_latents, confidence

    @staticmethod
    def _compute_edges(image_like: torch.Tensor) -> torch.Tensor:
        """Compute normalized edge magnitude maps from multi-channel feature maps."""

        gray = image_like.mean(dim=1, keepdim=True)
        grad_x = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        grad_y = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        normalized = (magnitude - mean) / std
        return torch.cat([grad_x, grad_y], dim=1) * 0.5 + normalized
