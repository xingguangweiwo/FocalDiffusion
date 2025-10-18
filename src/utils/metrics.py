"""
Evaluation metrics for depth estimation
"""

import torch
import numpy as np
from typing import Dict, Tuple

def compute_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute depth estimation metrics

    Args:
        depth_pred: Predicted depth [B, H, W]
        depth_gt: Ground truth depth [B, H, W]
        mask: Valid pixel mask [B, H, W]
    """
    if mask is None:
        mask = torch.ones_like(depth_gt, dtype=torch.bool)

    # Apply mask
    pred = depth_pred[mask]
    gt = depth_gt[mask]

    # Avoid division by zero
    thresh = torch.maximum((gt / pred), (pred / gt))

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
    """Compute surface normal metrics"""
    if mask is None:
        mask = torch.ones(normal_gt.shape[:-1], dtype=torch.bool)

    # Normalize
    normal_pred = F.normalize(normal_pred, p=2, dim=-1)
    normal_gt = F.normalize(normal_gt, p=2, dim=-1)

    # Compute dot product
    dot_product = (normal_pred * normal_gt).sum(dim=-1)
    dot_product = torch.clamp(dot_product, -1, 1)

    # Apply mask
    dot_product = dot_product[mask]

    # Compute angles
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