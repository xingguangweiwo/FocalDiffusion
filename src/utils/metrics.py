"""Evaluation metrics for depth and normal estimation utilities."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _zero_depth_metrics() -> Dict[str, float]:
    return {
        'abs_rel': 0.0,
        'sq_rel': 0.0,
        'rmse': 0.0,
        'rmse_log': 0.0,
        'a1': 0.0,
        'a2': 0.0,
        'a3': 0.0,
    }


def _align_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=reference.device, dtype=torch.bool)
    if mask.shape == reference.shape:
        return mask
    if reference.dim() == mask.dim() + 1 and reference.shape[1] == 1:
        mask = mask.unsqueeze(1)
    elif mask.dim() == reference.dim() + 1 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.shape != reference.shape:
        mask = mask.expand_as(reference)
    return mask


def compute_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute depth estimation metrics.

    Args:
        depth_pred: Predicted depth [B, H, W] or [B, 1, H, W].
        depth_gt: Ground truth depth [B, H, W] or [B, 1, H, W].
        mask: Optional valid pixel mask matching either shape.
    """
    depth_pred = depth_pred.to(dtype=torch.float32).clamp_min(1e-6)
    depth_gt = depth_gt.to(device=depth_pred.device, dtype=torch.float32).clamp_min(1e-6)

    if mask is None:
        mask = torch.ones_like(depth_gt, dtype=torch.bool)
    else:
        mask = _align_mask(mask, depth_gt)

    if not mask.any():
        return _zero_depth_metrics()

    pred = depth_pred[mask]
    gt = depth_gt[mask]

    thresh = torch.maximum(gt / pred, pred / gt)

    metrics = {
        'abs_rel': torch.mean(torch.abs(gt - pred) / gt).item(),
        'sq_rel': torch.mean(((gt - pred) ** 2) / gt).item(),
        'rmse': torch.sqrt(torch.mean((gt - pred) ** 2)).item(),
        'rmse_log': torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2)).item(),
        'a1': (thresh < 1.25).float().mean().item(),
        'a2': (thresh < 1.25 ** 2).float().mean().item(),
        'a3': (thresh < 1.25 ** 3).float().mean().item(),
    }

    return metrics


def compute_normal_metrics(
    normal_pred: torch.Tensor,
    normal_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute surface normal metrics."""
    if mask is None:
        mask = torch.ones(normal_gt.shape[:-1], dtype=torch.bool, device=normal_gt.device)
    else:
        mask = mask.to(device=normal_gt.device, dtype=torch.bool)

    normal_pred = F.normalize(normal_pred, p=2, dim=-1)
    normal_gt = F.normalize(normal_gt, p=2, dim=-1)

    dot_product = (normal_pred * normal_gt).sum(dim=-1)
    dot_product = torch.clamp(dot_product, -1, 1)
    dot_product = dot_product[mask]

    if dot_product.numel() == 0:
        return {
            'mean_angle': 0.0,
            'median_angle': 0.0,
            'rmse_angle': 0.0,
            '11.25': 0.0,
            '22.5': 0.0,
            '30': 0.0,
        }

    angles = torch.acos(dot_product) * 180 / np.pi

    metrics = {
        'mean_angle': angles.mean().item(),
        'median_angle': angles.median().item(),
        'rmse_angle': torch.sqrt((angles ** 2).mean()).item(),
        '11.25': (angles < 11.25).float().mean().item(),
        '22.5': (angles < 22.5).float().mean().item(),
        '30': (angles < 30).float().mean().item(),
    }

    return metrics
