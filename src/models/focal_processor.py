"""
Focal Stack Processor - Core module for processing focal stacks
Fixed version with complete implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class MultiScaleEncoder(nn.Module):
    """Multi-scale feature encoder for images"""

    def __init__(self, in_channels: int = 3, output_dim: int = 512, num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales

        # Progressive downsampling with feature extraction
        channels = [64, 128, 256, 512]
        self.encoders = nn.ModuleList()

        current_channels = in_channels
        for i in range(num_scales):
            encoder = nn.Sequential(
                nn.Conv2d(current_channels, channels[i], 3, stride=2, padding=1),
                nn.GroupNorm(8, channels[i]),
                nn.SiLU(),
                nn.Conv2d(channels[i], channels[i], 3, padding=1),
                nn.GroupNorm(8, channels[i]),
                nn.SiLU(),
            )
            self.encoders.append(encoder)
            current_channels = channels[i]

        # Project to uniform dimension
        self.projectors = nn.ModuleList([
            nn.Conv2d(channels[i], output_dim, 1) for i in range(num_scales)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dictionary of features at different scales"""
        features = {}
        current = x

        for i in range(self.num_scales):
            current = self.encoders[i](current)
            projected = self.projectors[i](current)
            features[f'scale_{i}'] = projected

        return features


class FocusFeatureNet(nn.Module):
    """Network for extracting focus-specific features"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.focus_encoder = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, focus_distances: torch.Tensor) -> torch.Tensor:
        """Encode focus distances to features"""
        # focus_distances: [B, N]
        B, N = focus_distances.shape

        # Normalize focus distances
        focus_norm = torch.log(focus_distances.clamp(min=0.1))
        focus_norm = (focus_norm - focus_norm.mean()) / (focus_norm.std() + 1e-8)

        # Encode each focus distance
        focus_features = self.focus_encoder(focus_norm.unsqueeze(-1))  # [B, N, D]

        return focus_features


class MultiScaleAdaptiveFusion(nn.Module):
    """Adaptive fusion of multi-scale features"""

    def __init__(self, feature_dim: int, num_scales: int):
        super().__init__()
        self.num_scales = num_scales

        # Attention weights for each scale
        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_scales),
            nn.Softmax(dim=-1)
        )

        # Feature fusion network
        self.fusion_net = nn.Sequential(
            nn.Conv2d(feature_dim * num_scales, feature_dim * 2, 3, padding=1),
            nn.GroupNorm(16, feature_dim * 2),
            nn.SiLU(),
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.SiLU(),
        )

        # Temporal attention for focal stack
        self.temporal_attention = FocalAwareTemporalAttention(feature_dim)

    def forward(self, features: Dict[str, torch.Tensor], focus_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fuse multi-scale features adaptively"""
        # Collect features from all scales
        B = focus_distances.shape[0]
        N = features['scale_0'].shape[1] if features['scale_0'].dim() == 5 else 1

        # Process each scale
        scale_features = []
        attention_summary = None
        for i in range(self.num_scales):
            feat = features[f'scale_{i}']
            if feat.dim() == 5:  # [B, N, D, H, W]
                # Apply temporal attention
                feat_attended, weights, attn_maps = self.temporal_attention(feat, focus_distances)
                scale_features.append(feat_attended)
                attention_summary = {
                    'frame_weights': weights,
                    'attn_maps': attn_maps
                }
            else:
                scale_features.append(feat)

        # Resize all features to the same size (use largest scale)
        target_size = scale_features[0].shape[-2:]
        aligned_features = []
        for feat in scale_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        # Concatenate and fuse
        concat_features = torch.cat(aligned_features, dim=1)
        fused = self.fusion_net(concat_features)

        outputs = {
            'fused_features': fused,
            'multiscale_features': features,
        }

        if attention_summary is not None:
            outputs['attention_weights'] = attention_summary['frame_weights']
            outputs['temporal_attention_maps'] = attention_summary['attn_maps']

        return outputs


class FocalAwareTemporalAttention(nn.Module):
    """Temporal attention for focal stack that preserves spatially-aware importance."""

    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // max(1, num_heads)

        # Multi-head projections
        self.q_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.k_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.v_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, 1)

        # Positional encoding for focus distances
        self.focus_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Relative position encoding between focus planes
        self.relative_position = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Tanh()
        )

        # Query weights derived from focus ordering
        self.query_weight_net = nn.Sequential(
            nn.Linear(1, feature_dim // 2),
            nn.SiLU(),
            nn.Linear(feature_dim // 2, 1)
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        features: torch.Tensor,
        focus_distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse features from a focal stack with temporally-aware attention.

        Args:
            features: [B, N, D, H, W]
            focus_distances: [B, N]

        Returns:
            fused_features: [B, D, H, W]
            frame_weights: [B, N]
            attention_maps: [B, N, H, W]
        """
        B, N, D, H, W = features.shape

        # Add focus-based positional encoding
        focus_encoding = self.focus_encoder(focus_distances.unsqueeze(-1))  # [B, N, D]
        focus_encoding = focus_encoding.unsqueeze(-1).unsqueeze(-1)
        features = features + 0.1 * focus_encoding

        # Reshape for attention computation
        features_flat = features.view(B * N, D, H, W)

        # Compute Q, K, V
        q = self.q_proj(features_flat).view(B, N, self.num_heads, self.head_dim, H * W)
        k = self.k_proj(features_flat).view(B, N, self.num_heads, self.head_dim, H * W)
        v = self.v_proj(features_flat).view(B, N, self.num_heads, self.head_dim, H * W)

        # Transpose for attention
        q = q.permute(0, 2, 4, 1, 3)  # [B, heads, H*W, N, head_dim]
        k = k.permute(0, 2, 4, 3, 1)  # [B, heads, H*W, head_dim, N]
        v = v.permute(0, 2, 4, 1, 3)  # [B, heads, H*W, N, head_dim]

        # Compute attention scores with relative focus bias
        scores = torch.matmul(q, k) / (self.temperature * math.sqrt(self.head_dim))
        relative = focus_distances[:, :, None] - focus_distances[:, None, :]
        relative = self.relative_position(relative.unsqueeze(-1))  # [B, N, N, heads]
        relative = relative.permute(0, 3, 1, 2).unsqueeze(2)  # [B, heads, 1, N, N]
        scores = scores + relative

        attn_weights = F.softmax(scores, dim=-1)  # [B, heads, H*W, N, N]

        # Apply attention over the focal sequence
        attended = torch.matmul(attn_weights, v)  # [B, heads, H*W, N, head_dim]

        # Derive adaptive weights for how much each query frame contributes
        query_logits = self.query_weight_net(focus_distances.unsqueeze(-1)).squeeze(-1)
        query_weights = torch.softmax(query_logits, dim=-1)  # [B, N]
        attended = (attended * query_weights.view(B, 1, 1, N, 1)).sum(dim=3)

        # Reshape back to image grid
        attended = attended.permute(0, 2, 1, 3).contiguous()
        attended = attended.view(B, H * W, self.num_heads * self.head_dim)
        attended = attended.transpose(1, 2).view(B, D, H, W)

        # Output projection
        output = self.out_proj(attended)

        # Aggregate spatial attention maps for diagnostics
        spatial_weights = attn_weights.mean(dim=1)  # [B, H*W, N, N]
        spatial_weights = spatial_weights.mean(dim=-2)  # [B, H*W, N]
        spatial_weights = spatial_weights.view(B, H, W, N).permute(0, 3, 1, 2)

        # Global frame weights (average over spatial locations)
        frame_weights = attn_weights.mean(dim=[1, 2, 3])  # [B, N]

        return output, frame_weights, spatial_weights


class FocalStackProcessor(nn.Module):
    """Main Focal Stack Processor with all components integrated"""

    def __init__(
        self,
        feature_dim: int = 512,
        num_scales: int = 4,
        max_sequence_length: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.max_sequence_length = max_sequence_length

        # Multi-scale encoder
        self.multiscale_encoder = MultiScaleEncoder(
            in_channels=3,
            output_dim=feature_dim,
            num_scales=num_scales
        )

        # Focus feature network
        self.focus_feature_net = FocusFeatureNet(feature_dim)

        # Multi-scale adaptive fusion
        self.adaptive_fusion = MultiScaleAdaptiveFusion(feature_dim, num_scales)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        focal_stack: torch.Tensor,
        focus_distances: torch.Tensor,
        camera_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process focal stack to extract features

        Args:
            focal_stack: [B, N, 3, H, W] - N images in focal stack
            focus_distances: [B, N] - focus distance for each image
            camera_params: Optional camera parameters

        Returns:
            Dictionary containing:
                - fused_features: [B, D, H', W']
                - multiscale_features: Dict of features at different scales
                - attention_weights: [B, N] importance of each focal image
        """
        B, N, C, H, W = focal_stack.shape

        # Check sequence length
        if N > self.max_sequence_length:
            raise ValueError(f"Sequence length {N} exceeds maximum {self.max_sequence_length}")

        # Flatten batch and sequence for encoding
        all_frames = focal_stack.view(B * N, C, H, W)

        # Extract multi-scale features
        multiscale_features = self.multiscale_encoder(all_frames)

        # Reorganize features
        enhanced_features = {}
        for scale_name, features in multiscale_features.items():
            _, D, H_feat, W_feat = features.shape
            # Reshape to [B, N, D, H, W]
            scale_features = features.view(B, N, D, H_feat, W_feat)
            enhanced_features[scale_name] = scale_features

        # Extract focus features
        focus_features = self.focus_feature_net(focus_distances)

        # Multi-scale adaptive fusion
        fused_results = self.adaptive_fusion(enhanced_features, focus_distances)

        # Add focus features to output
        fused_results['focus_features'] = focus_features

        return fused_results
