"""Loss functions for FocalDiffusion training."""

from __future__ import annotations

import importlib
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalDiffusionLoss(nn.Module):
    """Combined loss used during training."""

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        depth_weight: float = 0.1,
        rgb_weight: float = 0.1,
        consistency_weight: float = 0.05,
        perceptual_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.depth_weight = depth_weight
        self.rgb_weight = rgb_weight
        self.consistency_weight = consistency_weight
        self.perceptual_weight = perceptual_weight

        self.depth_loss = DepthLoss()
        self.consistency_loss = ConsistencyLoss()
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        depth_pred: Optional[torch.Tensor] = None,
        depth_target: Optional[torch.Tensor] = None,
        rgb_pred: Optional[torch.Tensor] = None,
        rgb_target: Optional[torch.Tensor] = None,
        focal_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        losses["diffusion"] = F.mse_loss(noise_pred, noise_target)

        if depth_pred is not None and depth_target is not None and self.depth_weight > 0:
            losses["depth"] = self.depth_loss(depth_pred, depth_target)

        if rgb_pred is not None and rgb_target is not None and self.rgb_weight > 0:
            losses["rgb"] = F.l1_loss(rgb_pred, rgb_target)
            if self.perceptual_loss is not None and self.perceptual_weight > 0:
                losses["perceptual"] = self.perceptual_loss(rgb_pred, rgb_target)

        if focal_features is not None and self.consistency_weight > 0:
            losses["consistency"] = self.consistency_loss(focal_features)

        total_loss = torch.zeros_like(losses["diffusion"])
        for key, value in losses.items():
            weight = getattr(self, f"{key}_weight", 1.0)
            total_loss = total_loss + weight * value
        losses["total"] = total_loss
        return losses


class DepthLoss(nn.Module):
    """Depth estimation loss with a scale-invariant component."""

    def __init__(self, alpha: float = 0.5, min_depth: float = 1e-3) -> None:
        super().__init__()
        self.alpha = alpha
        self.min_depth = min_depth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_safe = pred.clamp(min=self.min_depth)
        target_safe = target.clamp(min=self.min_depth)

        l1_loss = F.l1_loss(pred_safe, target_safe)
        diff = torch.log(pred_safe) - torch.log(target_safe)
        scale_inv = torch.mean(diff ** 2) - self.alpha * torch.mean(diff) ** 2
        return l1_loss + 0.1 * scale_inv


class ConsistencyLoss(nn.Module):
    """Regularises focal attention to remain sharp and smooth."""

    def forward(self, focal_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        weights = focal_features.get("attention_weights")
        if weights is None:
            device = None
            for value in focal_features.values():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        if weights.shape[-1] > 1:
            smoothness = torch.mean(torch.diff(weights, dim=-1) ** 2)
        else:
            smoothness = torch.zeros_like(entropy)
        return entropy + smoothness


class PerceptualLoss(nn.Module):
    """LPIPS perceptual loss wrapper."""

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        spec = importlib.util.find_spec("lpips")
        if spec is None:
            self._lpips = None
            self.register_buffer("_zero", torch.tensor(0.0), persistent=False)
            return

        lpips_module = importlib.import_module("lpips")
        self._lpips = lpips_module.LPIPS(net=net)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2

        if self._lpips is None:
            zero = self._zero.to(device=pred_norm.device, dtype=pred_norm.dtype)
            if pred_norm.ndim == 4:
                zero = zero.reshape(1, 1, 1, 1).expand(pred_norm.shape[0], 1, 1, 1)
            return zero

        loss = self._lpips(pred_norm, target_norm)
        if loss.ndim == 0 and pred_norm.ndim == 4:
            loss = loss.reshape(1, 1, 1, 1).expand(pred_norm.shape[0], 1, 1, 1)
        return loss

