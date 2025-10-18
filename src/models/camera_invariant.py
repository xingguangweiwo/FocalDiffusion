"""
Camera-Invariant Representations for Focal Stack Processing
Handles arbitrary camera parameters through normalization and relative encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
import math


class CameraInvariantEncoder(nn.Module):
    """
    Encodes camera parameters into invariant representations
    Supports three modes: relative, normalized, and learned
    """

    def __init__(
            self,
            output_dim: int = 512,
            hidden_dim: int = 256,
            num_layers: int = 3,
            use_fourier_features: bool = True,
            fourier_scale: float = 10.0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_fourier_features = use_fourier_features
        self.fourier_scale = fourier_scale

        # Fourier feature embedding
        if use_fourier_features:
            self.fourier_dim = 64
            self.B = nn.Parameter(
                torch.randn(self.fourier_dim // 2, 1) * fourier_scale,
                requires_grad=False
            )

        # Parameter encoders
        self.focal_length_encoder = self._build_encoder("focal_length")
        self.aperture_encoder = self._build_encoder("aperture")
        self.sensor_encoder = self._build_encoder("sensor")

        # Relative relationship encoder
        self.relation_encoder = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Normalization parameters (learnable)
        self.register_buffer('focal_length_mean', torch.tensor(50.0))  # mm
        self.register_buffer('focal_length_std', torch.tensor(50.0))
        self.register_buffer('aperture_mean', torch.tensor(2.8))
        self.register_buffer('aperture_std', torch.tensor(2.0))

    def forward(
            self,
            camera_params: Dict[str, torch.Tensor],
            mode: str = "relative",
            focus_distances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode camera parameters

        Args:
            camera_params: Dict with 'focal_length', 'aperture', 'sensor_size'
            mode: "relative", "normalized", or "learned"
            focus_distances: [B, N] for relative encoding

        Returns:
            camera_features: [B, output_dim]
        """
        if mode == "relative":
            return self.encode_relative(camera_params, focus_distances)
        elif mode == "normalized":
            return self.encode_normalized(camera_params)
        elif mode == "learned":
            return self.encode_learned(camera_params, focus_distances)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def encode_relative(
            self,
            camera_params: Dict[str, torch.Tensor],
            focus_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode camera parameters relative to focus distances
        This creates scale-invariant representations
        """
        B, N = focus_distances.shape

        focal_length = camera_params['focal_length']  # [B] or scalar
        aperture = camera_params['aperture']  # [B] or scalar

        # Ensure proper shapes
        if focal_length.dim() == 0:
            focal_length = focal_length.expand(B)
        if aperture.dim() == 0:
            aperture = aperture.expand(B)

        # Compute relative quantities
        features = []

        # 1. Hyperfocal distance (scale-invariant when normalized)
        hyperfocal = (focal_length ** 2) / (aperture * 0.03)  # 0.03mm = typical CoC limit
        hyperfocal_norm = hyperfocal.unsqueeze(1) / (focus_distances + 1e-6)  # [B, N]
        features.append(self._embed_fourier(hyperfocal_norm) if self.use_fourier_features
                        else hyperfocal_norm.unsqueeze(-1))

        # 2. Relative aperture (f-number relative to focal length is already normalized)
        aperture_features = self._embed_fourier(aperture.unsqueeze(1).expand(-1, N))
        features.append(aperture_features)

        # 3. Depth of field indicators
        near_dof = []
        far_dof = []

        for i in range(N):
            d = focus_distances[:, i]  # [B]

            # Near and far DoF limits (normalized by focus distance)
            H = hyperfocal
            near = (H * d) / (H + d - focal_length + 1e-6)
            far = torch.where(
                d < H - focal_length,
                (H * d) / (H - d - focal_length + 1e-6),
                torch.ones_like(d) * 1e6  # Infinity
            )

            near_norm = near / (d + 1e-6)
            far_norm = torch.minimum(far / (d + 1e-6), torch.ones_like(far) * 10.0)

            near_dof.append(near_norm)
            far_dof.append(far_norm)

        near_dof = torch.stack(near_dof, dim=1)  # [B, N]
        far_dof = torch.stack(far_dof, dim=1)  # [B, N]

        features.append(self._embed_fourier(near_dof))
        features.append(self._embed_fourier(far_dof))

        # 4. Focus distance ratios (pairwise)
        if N > 1:
            ratios = []
            for i in range(N):
                for j in range(i + 1, N):
                    ratio = focus_distances[:, i] / (focus_distances[:, j] + 1e-6)
                    ratios.append(ratio)

            ratios = torch.stack(ratios, dim=1)  # [B, N*(N-1)/2]
            features.append(self._embed_fourier(ratios))

        # Concatenate and encode
        features = torch.cat(features, dim=-1)  # [B, N, D]

        # Pool over focus positions
        features = features.mean(dim=1)  # [B, D]

        # Final encoding
        camera_encoding = self.relation_encoder(features)

        return camera_encoding

    def encode_normalized(self, camera_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Simple normalized encoding of camera parameters
        """
        focal_length = camera_params['focal_length']
        aperture = camera_params['aperture']

        # Normalize using learned statistics
        focal_norm = (focal_length - self.focal_length_mean) / self.focal_length_std
        aperture_norm = (aperture - self.aperture_mean) / self.aperture_std

        # Encode
        focal_feat = self.focal_length_encoder(focal_norm.unsqueeze(-1))
        aperture_feat = self.aperture_encoder(aperture_norm.unsqueeze(-1))

        # Combine
        combined = focal_feat + aperture_feat

        return combined

    def encode_learned(
            self,
            camera_params: Dict[str, torch.Tensor],
            focus_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fully learned encoding with minimal inductive bias
        """
        # Stack all parameters
        focal_length = camera_params['focal_length']
        aperture = camera_params['aperture']

        if focal_length.dim() == 0:
            focal_length = focal_length.unsqueeze(0)
        if aperture.dim() == 0:
            aperture = aperture.unsqueeze(0)

        # Create feature vector
        features = torch.cat([
            focal_length.unsqueeze(-1),
            aperture.unsqueeze(-1),
            focus_distances.mean(dim=1, keepdim=True),
            focus_distances.std(dim=1, keepdim=True),
            focus_distances.min(dim=1, keepdim=True)[0],
            focus_distances.max(dim=1, keepdim=True)[0],
        ], dim=-1)

        # Encode with all three encoders
        focal_feat = self.focal_length_encoder(features)
        aperture_feat = self.aperture_encoder(features)
        sensor_feat = self.sensor_encoder(features)

        # Combine
        combined = torch.cat([focal_feat, aperture_feat, sensor_feat], dim=-1)
        output = self.relation_encoder(combined)

        return output

    def _build_encoder(self, name: str) -> nn.Module:
        """Build a parameter-specific encoder"""
        if self.use_fourier_features:
            input_dim = self.fourier_dim
        else:
            input_dim = 1

        return nn.Sequential(
            nn.Linear(input_dim if name != "sensor" else input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def _embed_fourier(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature embedding"""
        if not self.use_fourier_features:
            return x.unsqueeze(-1)

        # Handle different input shapes
        original_shape = x.shape
        x = x.reshape(-1, 1)

        # Apply Fourier features
        x_proj = 2 * np.pi * x @ self.B.T
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # Reshape back
        features = features.reshape(*original_shape, -1)
        return features


class InvariantCostVolume(nn.Module):
    """
    Camera-invariant cost volume representation
    Inspired by "Deep depth from focal stack with defocus model"
    but with improvements for better generalization
    """

    def __init__(
            self,
            feature_dim: int = 256,
            depth_bins: int = 64,
            min_depth: float = 0.1,
            max_depth: float = 10.0,
            use_log_depth: bool = True,
            temperature: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.depth_bins = depth_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_log_depth = use_log_depth
        self.temperature = temperature

        # Generate depth candidates
        if use_log_depth:
            self.depth_candidates = torch.exp(
                torch.linspace(
                    np.log(min_depth),
                    np.log(max_depth),
                    depth_bins
                )
            )
        else:
            self.depth_candidates = torch.linspace(min_depth, max_depth, depth_bins)

        # Matching network
        self.matching_net = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, 1, 1),
        )

        # Refinement network
        self.refine_net = nn.Sequential(
            nn.Conv3d(depth_bins, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, depth_bins, 1),
        )

    def forward(
            self,
            focal_features: torch.Tensor,  # [B, N, C, H, W]
            focus_distances: torch.Tensor,  # [B, N]
            camera_encoding: torch.Tensor,  # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build camera-invariant cost volume

        Returns:
            cost_volume: [B, D, H, W, depth_bins]
            depth_probs: [B, H, W, depth_bins]
        """
        B, N, C, H, W = focal_features.shape
        device = focal_features.device

        # Move depth candidates to device
        depth_candidates = self.depth_candidates.to(device)

        # Initialize cost volume
        cost_volume = torch.zeros(B, self.depth_bins, H, W, device=device)

        # For each depth candidate
        for d_idx, depth in enumerate(depth_candidates):
            costs = []

            # For each focal image
            for n in range(N):
                # Compute blur kernel size based on CoC
                # This is camera-invariant when properly normalized
                blur_sigma = self._compute_invariant_blur(
                    depth,
                    focus_distances[:, n],
                    camera_encoding
                )

                # Extract features for this image
                feat_n = focal_features[:, n]  # [B, C, H, W]

                # Apply theoretical blur
                feat_blurred = self._apply_blur(feat_n, blur_sigma)

                # Compare with sharp reference (first image or computed)
                if n == 0:
                    feat_ref = feat_n

                # Compute matching cost
                cost = self.matching_net(torch.cat([feat_blurred, feat_ref], dim=1))
                costs.append(cost)

            # Aggregate costs across focal stack
            cost_aggregated = torch.stack(costs, dim=1).mean(dim=1)  # [B, 1, H, W]
            cost_volume[:, d_idx] = cost_aggregated.squeeze(1)

        # Refine cost volume
        cost_volume = cost_volume.unsqueeze(1)  # [B, 1, D, H, W]
        cost_volume = self.refine_net(cost_volume.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        cost_volume = cost_volume.squeeze(1)  # [B, D, H, W]

        # Convert to probabilities
        depth_probs = F.softmax(-cost_volume / self.temperature, dim=1)

        # Transpose for output
        cost_volume = cost_volume.permute(0, 2, 3, 1)  # [B, H, W, D]
        depth_probs = depth_probs.permute(0, 2, 3, 1)  # [B, H, W, D]

        return cost_volume, depth_probs

    def _compute_invariant_blur(
            self,
            depth: torch.Tensor,
            focus_distance: torch.Tensor,
            camera_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute camera-invariant blur amount
        Uses relative quantities instead of absolute CoC
        """
        # Relative defocus amount
        rel_defocus = torch.abs(1.0 / depth - 1.0 / focus_distance) * focus_distance

        # Scale by encoded camera parameters (learned)
        # This allows the network to learn the appropriate scaling
        blur_scale = torch.sigmoid(camera_encoding[:, 0]) * 5.0 + 0.1

        blur_sigma = rel_defocus * blur_scale.unsqueeze(-1).unsqueeze(-1)

        return blur_sigma

    def _apply_blur(self, features: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur with given sigma"""
        # Simplified - in practice use proper Gaussian filtering
        # For now, just return features (would implement proper blur)
        return features


class CameraInvariantFocalProcessor(nn.Module):
    """
    Main focal processor with camera invariance
    Combines all techniques for robustness
    """

    def __init__(
            self,
            feature_dim: int = 512,
            num_scales: int = 4,
            max_sequence_length: int = 100,
            use_cost_volume: bool = True,
            depth_bins: int = 64,
    ):
        super().__init__()

        # Base focal processor (reuse your implementation)
        from .focal_processor import FocalStackProcessor
        self.base_processor = FocalStackProcessor(
            feature_dim=feature_dim,
            num_scales=num_scales,
            max_sequence_length=max_sequence_length,
        )

        # Camera encoder
        self.camera_encoder = CameraInvariantEncoder(
            output_dim=feature_dim,
        )

        # Cost volume (optional)
        self.use_cost_volume = use_cost_volume
        if use_cost_volume:
            self.cost_volume = InvariantCostVolume(
                feature_dim=feature_dim,
                depth_bins=depth_bins,
            )

        # Feature modulation based on camera
        self.camera_modulation = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )

    def forward(
            self,
            focal_stack: torch.Tensor,
            focus_distances: torch.Tensor,
            camera_params: Optional[Dict[str, torch.Tensor]] = None,
            camera_features: Optional[torch.Tensor] = None,
            return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process focal stack with camera invariance
        """
        B, N, C, H, W = focal_stack.shape

        # Get base features
        base_results = self.base_processor(
            focal_stack,
            focus_distances,
            camera_params,
        )

        # Get camera encoding if not provided
        if camera_features is None and camera_params is not None:
            camera_features = self.camera_encoder(
                camera_params,
                mode="relative",
                focus_distances=focus_distances,
            )

        # Modulate features based on camera
        if camera_features is not None:
            # Get modulation parameters
            modulation = self.camera_modulation(camera_features)
            scale, shift = modulation.chunk(2, dim=-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)

            # Apply modulation to fused features
            base_results['fused_features'] = (
                    base_results['fused_features'] * (1 + scale) + shift
            )

        # Build cost volume if enabled
        if self.use_cost_volume and camera_features is not None:
            multiscale_features = base_results['multiscale_features']

            # Use finest scale for cost volume
            finest_features = multiscale_features['scale_0']  # [B, N, D, H', W']

            cost_volume, depth_probs = self.cost_volume(
                finest_features,
                focus_distances,
                camera_features,
            )

            base_results['cost_volume'] = cost_volume
            base_results['depth_probs'] = depth_probs

        if return_intermediate:
            base_results['camera_features'] = camera_features

        return base_results