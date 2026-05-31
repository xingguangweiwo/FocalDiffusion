"""
Camera-Invariant Representations for Focal Stack Processing
Handles arbitrary camera parameters through normalization and relative encoding
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


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

        # Relative relationship encoder.  Relative mode concatenates five
        # Fourier-embedded groups: hyperfocal ratio, aperture, near DoF, far DoF,
        # and a per-focus summary of pairwise focus-distance ratios.
        relation_input_dim = self.fourier_dim * 5 if use_fourier_features else 5
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_input_dim, hidden_dim),
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
        aperture_values = aperture.unsqueeze(1).expand(-1, N)
        aperture_features = self._embed_fourier(aperture_values) if self.use_fourier_features \
            else aperture_values.unsqueeze(-1)
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

        features.append(self._embed_fourier(near_dof) if self.use_fourier_features else near_dof.unsqueeze(-1))
        features.append(self._embed_fourier(far_dof) if self.use_fourier_features else far_dof.unsqueeze(-1))

        # 4. Focus distance ratios. Keep this as a per-focus-position feature so
        # every entry in ``features`` has shape [B, N, D] before concatenation.
        # The previous pair-list representation was [B, N*(N-1)/2, D], which
        # fails for common focal stack sizes (for example N=5, 8, or 9).
        if N > 1:
            pair_ratios = focus_distances.unsqueeze(2) / (focus_distances.unsqueeze(1) + 1e-6)
            off_diagonal = ~torch.eye(N, dtype=torch.bool, device=focus_distances.device)
            ratio_summary = pair_ratios[:, off_diagonal].view(B, N, N - 1).mean(dim=-1)
        else:
            ratio_summary = torch.ones_like(focus_distances)

        features.append(self._embed_fourier(ratio_summary) if self.use_fourier_features
                        else ratio_summary.unsqueeze(-1))

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
