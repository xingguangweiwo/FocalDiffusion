"""Shared depth and reliability metrics."""

from typing import Dict, Optional

import numpy as np
import torch


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



def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUROC for binary labels without optional sklearn dependency."""
    scores = scores.flatten().float()
    labels = labels.flatten().bool()
    valid = torch.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    positives = labels.sum()
    negatives = (~labels).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    pos_ranks = ranks[labels].sum()
    auc = (pos_ranks - positives.float() * (positives.float() + 1.0) / 2.0) / (positives.float() * negatives.float())
    return float(auc.item())


def binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute area under the precision-recall curve for binary labels."""
    scores = scores.flatten().float()
    labels = labels.flatten().bool()
    valid = torch.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    positives = labels.sum().float()
    if positives == 0 or scores.numel() == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    tp = torch.cumsum(sorted_labels, dim=0)
    precision = tp / torch.arange(1, sorted_labels.numel() + 1, device=scores.device, dtype=torch.float32)
    recall = tp / positives.clamp(min=1.0)
    recall_prev = torch.cat([recall.new_zeros(1), recall[:-1]])
    return float(((recall - recall_prev) * precision).sum().item())


def spearman_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Spearman rank correlation for finite tensor entries."""
    a = a.flatten().float()
    b = b.flatten().float()
    valid = torch.isfinite(a) & torch.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.numel() < 2:
        return float("nan")
    a_order = torch.argsort(a)
    b_order = torch.argsort(b)
    a_rank = torch.empty_like(a, dtype=torch.float32)
    b_rank = torch.empty_like(b, dtype=torch.float32)
    a_rank[a_order] = torch.arange(a.numel(), device=a.device, dtype=torch.float32)
    b_rank[b_order] = torch.arange(b.numel(), device=b.device, dtype=torch.float32)
    a_rank = a_rank - a_rank.mean()
    b_rank = b_rank - b_rank.mean()
    denom = a_rank.norm() * b_rank.norm()
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((a_rank * b_rank).sum().div(denom).item())


def selective_risk_at_coverage(confidence: torch.Tensor, risk: torch.Tensor, coverage: float) -> float:
    """Mean risk retained at the requested confidence coverage."""
    flat_confidence = confidence.flatten()
    flat_risk = risk.flatten()
    if flat_confidence.numel() == 0:
        return float("nan")
    import math
    coverage_count = max(1, min(flat_confidence.numel(), int(math.ceil(flat_confidence.numel() * min(max(float(coverage), 0.0), 1.0)))))
    top_indices = torch.topk(flat_confidence, k=coverage_count).indices
    return float(flat_risk[top_indices].mean().item())


def risk_coverage_auc(confidence: torch.Tensor, risk: torch.Tensor) -> float:
    """Area under the empirical risk-coverage curve sorted by confidence."""
    flat_confidence = confidence.flatten()
    flat_risk = risk.flatten()
    if flat_confidence.numel() == 0:
        return float("nan")
    order = torch.argsort(flat_confidence, descending=True)
    sorted_risk = flat_risk[order]
    cumulative_risk = torch.cumsum(sorted_risk, dim=0) / torch.arange(1, sorted_risk.numel() + 1, device=sorted_risk.device, dtype=sorted_risk.dtype)
    return float(cumulative_risk.mean().item())


def sparsification_curve(errors: np.ndarray, scores: np.ndarray, fractions: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean retained error after rejecting highest-score pixels."""
    errors = np.asarray(errors, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if fractions is None:
        fractions = np.linspace(0.0, 0.95, 20)
    fractions = np.asarray(list(fractions), dtype=np.float64)
    order = np.argsort(-scores)
    sorted_errors = errors[order]
    values = []
    n = len(errors)
    for frac in fractions:
        start = min(int(round(frac * n)), n - 1)
        values.append(float(np.mean(sorted_errors[start:])))
    return fractions, np.asarray(values, dtype=np.float64)


def ause(errors: np.ndarray, scores: np.ndarray) -> float:
    """Area under sparsification error against an oracle ranking."""
    fractions, curve = sparsification_curve(errors, scores)
    _, oracle = sparsification_curve(errors, errors)
    return float(np.trapz(curve - oracle, fractions))


def aurg(errors: np.ndarray, scores: np.ndarray) -> float:
    """Area under rejection gain relative to random ordering."""
    fractions, curve = sparsification_curve(errors, scores)
    random_curve = np.full_like(curve, np.mean(errors), dtype=np.float64)
    return float(np.trapz(random_curve - curve, fractions))


def rejection_auc(errors: np.ndarray, scores: np.ndarray) -> float:
    """Summarize retained-pixel error as uncertainty-based rejection increases."""
    fractions, curve = sparsification_curve(errors, scores)
    return float(np.trapz(curve, fractions))
