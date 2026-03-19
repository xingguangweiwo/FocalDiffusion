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
    val_metrics = {'loss': 0.0, 'abs_rel': 0.0, 'rmse': 0.0}
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

            depth_gt = batch['depth'].to(output.depth_map.device).squeeze(1)
            mask = batch.get('valid_mask')
            if mask is not None:
                mask = mask.to(output.depth_map.device)

            metrics = compute_metrics(
                output.depth_map,
                depth_gt,
                mask=mask,
            )

            for key, value in metrics.items():
                if key in val_metrics:
                    val_metrics[key] += value

            if mask is not None:
                val_loss = F.l1_loss(output.depth_map[mask], depth_gt[mask])
            else:
                val_loss = F.l1_loss(output.depth_map, depth_gt)
            val_metrics['loss'] += val_loss.item()

            num_batches += 1

    for key in val_metrics:
        val_metrics[key] /= num_batches

    return val_metrics
