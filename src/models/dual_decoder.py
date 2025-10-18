"""
Dual Output Decoder for simultaneous depth and all-in-focus image generation
Ensures physical consistency between outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class DualOutputDecoder(nn.Module):
    """
    Decoder that generates both depth map and all-in-focus image
    with physical constraints to ensure consistency
    """

    def __init__(
            self,
            in_channels: int = 16,  # SD3.5 latent channels
            out_channels_depth: int = 1,
            out_channels_aif: int = 3,
            hidden_dims: List[int] = [512, 256, 128, 64],
            use_skip_connections: bool = True,
            use_physical_constraints: bool = True,
            use_vae_decoder: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_skip_connections = use_skip_connections
        self.use_physical_constraints = use_physical_constraints
        self.use_vae_decoder = use_vae_decoder

        # Shared encoder path
        self.shared_encoder = nn.ModuleList()
        current_dim = in_channels

        for hidden_dim in hidden_dims[:2]:  # Share first two layers
            self.shared_encoder.append(
                nn.Sequential(
                    nn.Conv2d(current_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                )
            )
            current_dim = hidden_dim

            # Depth-specific decoder
            self.depth_decoder = self._build_decoder_branch(
                current_dim,
                hidden_dims[2:],
                out_channels_depth,
                "depth"
            )

            # AIF-specific decoder
            self.aif_decoder = self._build_decoder_branch(
                current_dim,
                hidden_dims[2:],
                out_channels_aif if not use_vae_decoder else in_channels,
                "AIF"
            )

            # Cross-modality attention for consistency
            self.consistency_attention = CrossModalityAttention(
                depth_dim=hidden_dims[-1],
                rgb_dim=hidden_dims[-1],
                num_heads=8,
            )


class CrossModalityAttention(nn.Module):
    """
    Cross-attention between depth and AIF features
    Ensures consistency between modalities
    """

    def __init__(
            self,
            depth_dim: int,
            aif_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = depth_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Depth attending to RGB
        self.depth_to_aif = nn.MultiheadAttention(
            depth_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.depth_norm = nn.LayerNorm(depth_dim)

        # RGB attending to depth
        self.aif_to_depth = nn.MultiheadAttention(
            aif_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.rgb_norm = nn.LayerNorm(aif_dim)

        # Gating mechanism
        self.depth_gate = nn.Sequential(
            nn.Linear(depth_dim * 2, depth_dim),
            nn.Sigmoid()
        )
        self.aif_gate = nn.Sequential(
            nn.Linear(aif_dim * 2, aif_dim),
            nn.Sigmoid()
        )

    def forward(
            self,
            depth_features: torch.Tensor,
            aif_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modality attention

        Args:
            depth_features: [B, C_d, H, W]
            aif_features: [B, C_r, H, W]

        Returns:
            refined_depth: [B, C_d, H, W]
            refined_aif: [B, C_r, H, W]
        """
        B, C_d, H, W = depth_features.shape
        _, C_r, _, _ = aif_features.shape

        # Reshape for attention
        depth_seq = depth_features.flatten(2).transpose(1, 2)  # [B, H*W, C_d]
        aif_seq = aif_features.flatten(2).transpose(1, 2)  # [B, H*W, C_r]

        # Cross attention
        depth_attn, _ = self.depth_to_rgb(
            depth_seq, aif_seq, aif_seq
        )
        rgb_attn, _ = self.rgb_to_depth(
            aif_seq, depth_seq, depth_seq
        )

        # Residual connection with gating
        depth_combined = torch.cat([depth_seq, depth_attn], dim=-1)
        depth_gate = self.depth_gate(depth_combined)
        depth_refined = depth_seq + depth_gate * self.depth_norm(depth_attn)

        aif_combined = torch.cat([aif_seq, rgb_attn], dim=-1)
        aif_gate = self.rgb_gate(aif_combined)
        aif_refined = aif_seq + aif_gate * self.aif_norm(rgb_attn)

        # Reshape back
        depth_refined = depth_refined.transpose(1, 2).reshape(B, C_d, H, W)
        aif_refined = aif_refined.transpose(1, 2).reshape(B, C_r, H, W)

        return depth_refined, aif_refined

