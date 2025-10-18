"""
Loss functions for FocalDiffusion training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import lpips


class FocalDiffusionLoss(nn.Module):
    """Combined loss for FocalDiffusion training"""

    def __init__(
            self,
            diffusion_weight: float = 1.0,
            depth_weight: float = 0.1,
            rgb_weight: float = 0.1,
            consistency_weight: float = 0.05,
            perceptual_weight: float = 0.05,
    ):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.depth_weight = depth_weight
        self.rgb_weight = rgb_weight
        self.consistency_weight = consistency_weight
        self.perceptual_weight = perceptual_weight

        # Initialize loss components
        self.depth_loss = DepthLoss()
        self.consistency_loss = ConsistencyLoss()

        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(
            self,
            noise_pred: torch.Tensor,
            noise_target: torch.Tensor,
            depth_pred: Optional[torch.Tensor] = None,
            depth_target: Optional[torch.Tensor] = None,
            rgb_pred: Optional[torch.Tensor] = None,
            rgb_target: Optional[torch.Tensor] = None,
            focal_features: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss"""
        losses = {}

        # Diffusion loss (main loss)
        losses['diffusion'] = F.mse_loss(noise_pred, noise_target)

        # Depth loss
        if depth_pred is not None and depth_target is not None and self.depth_weight > 0:
            losses['depth'] = self.depth_loss(depth_pred, depth_target)

        # RGB reconstruction loss
        if rgb_pred is not None and rgb_target is not None and self.rgb_weight > 0:
            losses['rgb'] = F.l1_loss(rgb_pred, rgb_target)

            # Perceptual loss
            if self.perceptual_loss is not None and self.perceptual_weight > 0:
                losses['perceptual'] = self.perceptual_loss(rgb_pred, rgb_target)

        # Consistency loss
        if focal_features is not None and self.consistency_weight > 0:
            losses['consistency'] = self.consistency_loss(focal_features)

        # Combine losses
        total_loss = sum(
            losses.get(k, 0) * getattr(self, f'{k}_weight', 1.0)
            for k in losses
        )

        losses['total'] = total_loss
        return losses


class DepthLoss(nn.Module):
    """Depth estimation loss with scale-invariant component"""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute scale-invariant depth loss"""
        # L1 loss
        l1_loss = F.l1_loss(pred, target)

        # Scale-invariant loss
        d = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
        scale_inv_loss = torch.mean(d ** 2) - self.alpha * torch.mean(d) ** 2

        return l1_loss + 0.1 * scale_inv_loss


class ConsistencyLoss(nn.Module):
    """Consistency loss for focal features"""

    def forward(self, focal_features: Dict) -> torch.Tensor:
        """Compute consistency across focal planes"""
        if 'attention_weights' not in focal_features:
            return torch.tensor(0.0)

        weights = focal_features['attention_weights']

        # Entropy regularization - encourage focused attention
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        entropy_loss = entropy.mean()

        # Smoothness regularization
        if weights.shape[-1] > 1:
            diff = torch.diff(weights, dim=-1)
            smoothness_loss = torch.mean(diff ** 2)
        else:
            smoothness_loss = 0.0

        return entropy_loss + smoothness_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using LPIPS"""

    def __init__(self, net: str = 'vgg'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss"""
        # Normalize to [0, 1] for LPIPS
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2

        return self.lpips(pred_norm, target_norm).mean()