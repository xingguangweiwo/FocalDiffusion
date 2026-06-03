"""Validation utilities for :mod:`src.training.trainer`."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..utils.metrics import compute_metrics


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(pred, target)

    mask = mask.to(device=pred.device, dtype=torch.bool)
    if pred.dim() == mask.dim() + 1 and pred.shape[1] == 1:
        mask = mask.unsqueeze(1)
    elif mask.dim() == pred.dim() + 1 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    if not mask.any():
        return pred.new_tensor(0.0)

    return F.l1_loss(pred[mask], target[mask])

def run_validation(trainer: "FocalDiffusionTrainer", epoch: int) -> Dict[str, float]:
    """Run one validation epoch for the provided trainer instance."""

    _ = epoch
    trainer.pipeline.eval()
    val_metrics = {
        'loss': 0.0,
        'abs_rel': 0.0,
        'rmse': 0.0,
        'normalized_loss': 0.0,
        'focus_entropy_mean': 0.0,
        'focus_reliability_mean': 0.0,
        'gate_focus_mean': 0.0,
        'gate_prior_mean': 0.0,
        'gate_abstain_mean': 0.0,
        'physical_support_mean': 0.0,
        'depth_disagreement_mean': 0.0,
        'posterior_margin_mean': 0.0,
        'uncertainty_final_mean': 0.0,
        'uncertainty_error_l1': 0.0,
        'depth_prior_focus_disagreement': 0.0,
        'uncertainty_mean': 0.0
    }
    metric_depth_batches = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(
            trainer.val_dataloader,
            desc="Validation",
            disable=not trainer.accelerator.is_local_main_process,
        ):
            output = trainer.pipeline(
                focal_stack=batch['focal_stack'],
                focus_distances=batch['focus_distances'],
                num_inference_steps=trainer.config['validation']['num_inference_steps'],
                guidance_scale=trainer.config['validation']['guidance_scale'],
                output_type='pt',
                return_dict=True,
            )

            depth_final_norm = output.depth_map
            if depth_final_norm.dim() == 3:
                depth_final_norm_for_loss = depth_final_norm.unsqueeze(1)
            else:
                depth_final_norm_for_loss = depth_final_norm
            uncertainty = getattr(output, "uncertainty_final", None)
            if uncertainty is None:
                uncertainty = getattr(output, "uncertainty", None)
            if uncertainty is not None:
                val_metrics["uncertainty_mean"] += uncertainty.mean().item()
            if output.focus_entropy is not None:
                val_metrics["focus_entropy_mean"] += output.focus_entropy.mean().item()
            if output.focus_reliability is not None:
                val_metrics["focus_reliability_mean"] += output.focus_reliability.mean().item()
            if output.depth_prior is not None and output.depth_focus is not None:
                val_metrics["depth_prior_focus_disagreement"] += torch.abs(output.depth_prior - output.depth_focus).mean().item()
            if output.gate_focus is not None:
                val_metrics["gate_focus_mean"] += output.gate_focus.mean().item()
            if output.gate_prior is not None:
                val_metrics["gate_prior_mean"] += output.gate_prior.mean().item()
            if output.gate_abstain is not None:
                val_metrics["gate_abstain_mean"] += output.gate_abstain.mean().item()
            if output.physical_support is not None:
                val_metrics["physical_support_mean"] += output.physical_support.mean().item()
            if output.depth_disagreement is not None:
                val_metrics["depth_disagreement_mean"] += output.depth_disagreement.mean().item()
            if output.posterior_margin is not None:
                val_metrics["posterior_margin_mean"] += output.posterior_margin.mean().item()
            if output.uncertainty_final is not None:
                val_metrics["uncertainty_final_mean"] += output.uncertainty_final.mean().item()

            depth_gt = batch.get('depth')
            depth_range = batch.get('depth_range')
            mask = batch.get('valid_mask')
            if mask is not None:
                mask = mask.to(depth_final_norm.device)

            has_metric_target = depth_gt is not None and depth_range is not None
            if has_metric_target:
                depth_gt = depth_gt.to(depth_final_norm.device)
                if depth_gt.dim() == 3:
                    depth_gt = depth_gt.unsqueeze(1)
                depth_range = depth_range.to(depth_final_norm.device)
                depth_min = depth_range[:, 0].view(-1, 1, 1, 1)
                depth_max = depth_range[:, 1].view(-1, 1, 1, 1)
                pred_metric = depth_final_norm_for_loss * (depth_max - depth_min).clamp(min=1e-6) + depth_min
                metrics = compute_metrics(pred_metric.squeeze(1), depth_gt.squeeze(1), mask=mask)
                for key, value in metrics.items():
                    if key in ("abs_rel", "rmse"):
                        val_metrics[key] += value
                metric_depth_batches += 1
                val_loss = _masked_l1(pred_metric, depth_gt, mask)
                val_metrics['loss'] += val_loss.item()
                if uncertainty is not None:
                    uncertainty_for_error = uncertainty.unsqueeze(1) if uncertainty.dim() == 3 else uncertainty
                    if uncertainty_for_error.shape[-2:] != depth_final_norm_for_loss.shape[-2:]:
                        uncertainty_for_error = F.interpolate(
                            uncertainty_for_error,
                            size=depth_final_norm_for_loss.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    error_norm = torch.abs(depth_final_norm_for_loss.detach() - ((depth_gt - depth_min) / (depth_max - depth_min).clamp(min=1e-6)).clamp(0.0, 1.0))
                    val_metrics["uncertainty_error_l1"] += _masked_l1(uncertainty_for_error, error_norm, mask).item()
            elif depth_gt is not None and depth_range is None:
                depth_gt = depth_gt.to(depth_final_norm.device)
                if depth_gt.dim() == 3:
                    depth_gt = depth_gt.unsqueeze(1)
                val_metrics['normalized_loss'] += F.l1_loss(depth_final_norm_for_loss, depth_gt).item()

            num_batches += 1

    if num_batches == 0:
        return val_metrics

    for key in val_metrics:
        if key in ("abs_rel", "rmse"):
            val_metrics[key] = val_metrics[key] / max(metric_depth_batches, 1)
        else:
            val_metrics[key] /= num_batches

    return val_metrics
