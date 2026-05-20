"""Loss functions for FocalDiffusion training."""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_focus_coordinates(focus_distances: torch.Tensor) -> torch.Tensor:
    tau_min = focus_distances.min(dim=1, keepdim=True).values
    tau_max = focus_distances.max(dim=1, keepdim=True).values
    return (focus_distances - tau_min) / (tau_max - tau_min + 1e-6)


class LearnedSharpnessEstimator(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, focal_stack: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = focal_stack.shape
        x = focal_stack.reshape(b * n, c, h, w)
        s = self.net(x).reshape(b, n, h, w)
        return s


def build_aif_physical(focal_stack: torch.Tensor, shape_norm: torch.Tensor, tau: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = -torch.abs(tau.unsqueeze(-1).unsqueeze(-1) - shape_norm.unsqueeze(1)) / max(temperature, 1e-6)
    w = torch.softmax(logits, dim=1).unsqueeze(2)
    return (w * focal_stack).sum(dim=1)


class FocalDiffusionLoss(nn.Module):
    def __init__(self, diffusion_weight: float = 1.0, depth_weight: float = 0.1, rgb_weight: float = 0.1, consistency_weight: float = 0.05, perceptual_weight: float = 0.05, depth_gradient_weight: float = 0.1, edge_consistency_weight: float = 0.05, confidence_regularization_weight: float = 0.01, focus_energy_weight: float = 0.5, tau_contrast_weight: float = 0.2, uncertainty_weight: float = 0.05, aif_highpass_weight: float = 0.2, focus_temperature: float = 0.07, tau_margin: float = 0.05) -> None:
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.depth_weight = depth_weight
        self.rgb_weight = rgb_weight
        self.consistency_weight = consistency_weight
        self.perceptual_weight = perceptual_weight
        self.depth_gradient_weight = depth_gradient_weight
        self.edge_consistency_weight = edge_consistency_weight
        self.confidence_regularization_weight = confidence_regularization_weight
        self.focus_energy_weight = focus_energy_weight
        self.tau_contrast_weight = tau_contrast_weight
        self.uncertainty_weight = uncertainty_weight
        self.aif_highpass_weight = aif_highpass_weight
        self.focus_temperature = focus_temperature
        self.tau_margin = tau_margin

        self.sharpness_estimator = LearnedSharpnessEstimator()

    def _focus_energy(self, sharpness: torch.Tensor, tau: torch.Tensor, shape_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target = torch.softmax(-torch.abs(tau.unsqueeze(-1).unsqueeze(-1) - shape_norm.unsqueeze(1)) / self.focus_temperature, dim=1)
        observed = torch.softmax(sharpness / self.focus_temperature, dim=1)
        kl = (target * (torch.log(target + 1e-8) - torch.log(observed + 1e-8))).sum(dim=1)
        top2 = torch.topk(sharpness, k=min(2, sharpness.shape[1]), dim=1).values
        gap = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else top2[:, 0])
        reliability = torch.sigmoid(gap)
        return (kl * reliability).mean(), reliability

    def forward(self, diffusion_pred: torch.Tensor, diffusion_target: torch.Tensor, depth_pred: Optional[torch.Tensor] = None, depth_target: Optional[torch.Tensor] = None, rgb_pred: Optional[torch.Tensor] = None, rgb_target: Optional[torch.Tensor] = None, depth_mask: Optional[torch.Tensor] = None, focal_features: Optional[Dict[str, torch.Tensor]] = None, confidence_map: Optional[torch.Tensor] = None, shape_norm: Optional[torch.Tensor] = None, uncertainty: Optional[torch.Tensor] = None, focal_stack: Optional[torch.Tensor] = None, focus_distances: Optional[torch.Tensor] = None, use_tau_contrast: bool = True, use_aif_highpass: bool = True) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {"diffusion": F.mse_loss(diffusion_pred, diffusion_target)}
        if depth_pred is not None and depth_target is not None:
            losses["depth"] = F.l1_loss(depth_pred, depth_target)
        if rgb_pred is not None and rgb_target is not None:
            losses["rgb"] = F.l1_loss(rgb_pred, rgb_target)

        if shape_norm is not None and focal_stack is not None and focus_distances is not None:
            tau = normalize_focus_coordinates(focus_distances)
            sharpness = self.sharpness_estimator(focal_stack)
            pos_energy, reliability = self._focus_energy(sharpness, tau, shape_norm.squeeze(1))
            losses["focus_energy"] = pos_energy
            if use_tau_contrast:
                perm = torch.randperm(tau.shape[1], device=tau.device)
                neg_energy, _ = self._focus_energy(sharpness, tau[:, perm], shape_norm.squeeze(1))
                losses["tau_contrast"] = F.relu(self.tau_margin + pos_energy - neg_energy)
            if uncertainty is not None:
                losses["uncertainty"] = F.l1_loss(uncertainty.squeeze(1), 1.0 - reliability)
            if rgb_pred is not None and use_aif_highpass:
                aif_phys = build_aif_physical(focal_stack, shape_norm.squeeze(1), tau)
                hp_pred = rgb_pred - F.avg_pool2d(rgb_pred, 5, stride=1, padding=2)
                hp_phys = aif_phys - F.avg_pool2d(aif_phys, 5, stride=1, padding=2)
                losses["aif_highpass"] = F.l1_loss(hp_pred, hp_phys)

        total = torch.zeros_like(losses["diffusion"])
        for k, v in losses.items():
            total = total + getattr(self, f"{k}_weight", 1.0) * v
        losses["total"] = total
        return losses


class DepthLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            if mask.dim() == pred.dim() - 1:
                mask = mask.unsqueeze(1)
            return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1)
        return F.l1_loss(pred, target)


class ConsistencyLoss(nn.Module):
    def forward(self, focal_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = focal_features.get('attention_weights')
        if w is None:
            return torch.zeros((), device=next(iter(focal_features.values())).device)
        return -torch.sum(w * torch.log(w + 1e-8), dim=-1).mean()


class PerceptualLoss(nn.Module):
    def __init__(self, net: str = 'vgg') -> None:
        super().__init__()
        self.register_buffer('_zero', torch.tensor(0.0), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._zero.to(device=pred.device, dtype=pred.dtype)
