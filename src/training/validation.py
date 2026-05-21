"""Validation utilities for :mod:`src.training.trainer`."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..utils.metrics import compute_metrics


def run_validation(trainer: "FocalDiffusionTrainer", epoch: int) -> Dict[str, float]:
    """Run one validation epoch for the provided trainer instance."""

    _ = epoch
    trainer.pipeline.eval()
    val_metrics = {'loss': 0.0, 'abs_rel': 0.0, 'rmse': 0.0, 'normalized_loss': 0.0, 'focus_energy': 0.0, 'uncertainty_mean': 0.0}
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
                camera_params=batch.get('camera_params'),
                num_inference_steps=trainer.config['validation']['num_inference_steps'],
                guidance_scale=trainer.config['validation']['guidance_scale'],
                output_type='pt',
                return_dict=True,
            )

            shape_norm = output.depth_map
            uncertainty = getattr(output, "uncertainty", None)
            if uncertainty is not None:
                val_metrics["uncertainty_mean"] += uncertainty.mean().item()

            depth_gt = batch.get('depth')
            depth_range = batch.get('depth_range')
            mask = batch.get('valid_mask')
            if mask is not None:
                mask = mask.to(shape_norm.device)

            has_metric_target = depth_gt is not None and depth_range is not None
            if has_metric_target:
                depth_gt = depth_gt.to(shape_norm.device).squeeze(1)
                depth_range = depth_range.to(shape_norm.device)
                depth_min = depth_range[:, 0].view(-1, 1, 1)
                depth_max = depth_range[:, 1].view(-1, 1, 1)
                pred_metric = shape_norm * (depth_max - depth_min).clamp(min=1e-6) + depth_min
                metrics = compute_metrics(pred_metric, depth_gt, mask=mask)
                for key, value in metrics.items():
                    if key in ("abs_rel", "rmse"):
                        val_metrics[key] += value
                metric_depth_batches += 1
                val_loss = F.l1_loss(pred_metric[mask], depth_gt[mask]) if mask is not None else F.l1_loss(pred_metric, depth_gt)
                val_metrics['loss'] += val_loss.item()
            elif depth_gt is not None and depth_range is None:
                depth_gt = depth_gt.to(shape_norm.device)
                val_metrics['normalized_loss'] += F.l1_loss(shape_norm.unsqueeze(1), depth_gt).item()

            if hasattr(output, "focus_energy") and output.focus_energy is not None:
                val_metrics["focus_energy"] += float(output.focus_energy.mean().item())

            num_batches += 1

    for key in val_metrics:
        if key in ("abs_rel", "rmse"):
            val_metrics[key] = val_metrics[key] / max(metric_depth_batches, 1)
        else:
            val_metrics[key] /= num_batches

    return val_metrics
