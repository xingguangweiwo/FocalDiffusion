"""
Evaluation script for FocalStackGeneration with normalized-depth aware metrics.
"""

import argparse
import warnings
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import yaml
import torch.nn.functional as F

from src.pipelines import load_pipeline
from src.data import create_dataloader
from src.utils.metrics import compute_metrics


def evaluate(args):
    warnings.warn("This script is deprecated and not aligned with the normalized-depth FocalStackGeneration pipeline. Use trainer validation until this script is updated.", DeprecationWarning, stacklevel=2)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading model from {args.checkpoint}...")
    pipeline = load_pipeline(
        checkpoint_path=args.checkpoint,
        base_model_id=config['model']['base_model_id'],
        device=args.device,
        dtype=torch.float16 if args.fp16 else torch.float32,
    )
    pipeline.eval()

    dataloader = create_dataloader(
        dataset_type=args.dataset,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=tuple(config['data']['image_size']),
        focal_stack_size=config['data']['focal_stack_size'],
        focal_range=tuple(config['data']['focal_range']),
        augmentation=False,
    )

    all_metrics = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if args.max_samples and batch_idx * args.batch_size >= args.max_samples:
            break

        focal_stack = batch['focal_stack'].to(args.device)
        focal_plane_distances = batch['focal_plane_distances'].to(args.device)
        depth_gt = batch.get('depth')
        depth_range = batch.get('depth_range')
        with torch.no_grad():
            output = pipeline(
                focal_stack=focal_stack,
                focal_plane_distances=focal_plane_distances,
                num_inference_steps=args.num_inference_steps,
                output_type='pt',
            )

        final_depth_canonical = output.depth_map
        sample_metrics = {}

        if depth_gt is not None and depth_range is not None:
            depth_gt = depth_gt.to(args.device).squeeze(1)
            depth_range = depth_range.to(args.device)
            depth_min = depth_range[:, 0].view(-1, 1, 1)
            depth_max = depth_range[:, 1].view(-1, 1, 1)
            pred_metric = final_depth_canonical * (depth_max - depth_min).clamp(min=1e-6) + depth_min
            sample_metrics.update(compute_metrics(pred_metric, depth_gt))
            sample_metrics["loss"] = F.l1_loss(pred_metric, depth_gt).item()
        elif depth_gt is not None:
            depth_gt = depth_gt.to(args.device)
            if final_depth_canonical.dim() == 3:
                final_depth_canonical = final_depth_canonical.unsqueeze(1)
            if depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(1)
            sample_metrics["normalized_loss"] = F.l1_loss(final_depth_canonical, depth_gt).item()

        uncertainty = getattr(output, "uncertainty_final", None)
        if uncertainty is None:
            uncertainty = output.uncertainty
        sample_metrics["uncertainty_mean"] = float(uncertainty.mean().item()) if uncertainty is not None else 0.0
        sample_metrics["focal_entropy_mean"] = float(output.focal_entropy.mean().item()) if output.focal_entropy is not None else 0.0
        sample_metrics["physical_evidence_support_mean"] = float(output.physical_evidence_support.mean().item()) if output.physical_evidence_support is not None else 0.0
        if output.generated_depth_canonical is not None and output.focal_depth_canonical is not None:
            sample_metrics["generated_focal_depth_disagreement"] = float(torch.abs(output.generated_depth_canonical - output.focal_depth_canonical).mean().item())
        all_metrics.append(sample_metrics)

    keys = sorted({k for m in all_metrics for k in m.keys()})
    final_metrics = {}
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        final_metrics[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'median': float(np.median(values))}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print("\nEvaluation Results:")
    print("-" * 40)
    for key, values in final_metrics.items():
        print(f"{key}: {values['mean']:.4f} ± {values['std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FocalStackGeneration model")
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['hypersim', 'virtual_kitti'], required=True)
    parser.add_argument('--data_root', type=str, required=True, help='Dataset root')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--max_samples', type=int, help='Maximum samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
