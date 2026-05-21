"""
Evaluation script for FocalDiffusion with normalized-shape aware metrics.
"""

import argparse
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

    use_camera_encoder = config.get("model", {}).get("use_camera_encoder", False)
    all_metrics = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if args.max_samples and batch_idx * args.batch_size >= args.max_samples:
            break

        focal_stack = batch['focal_stack'].to(args.device)
        focus_distances = batch['focus_distances'].to(args.device)
        depth_gt = batch.get('depth')
        depth_range = batch.get('depth_range')
        camera_params = batch.get('camera_params') if use_camera_encoder else None

        with torch.no_grad():
            output = pipeline(
                focal_stack=focal_stack,
                focus_distances=focus_distances,
                camera_params=camera_params,
                num_inference_steps=args.num_inference_steps,
                output_type='pt',
            )

        shape_norm = output.depth_map
        sample_metrics = {}

        if depth_gt is not None and depth_range is not None:
            depth_gt = depth_gt.to(args.device).squeeze(1)
            depth_range = depth_range.to(args.device)
            depth_min = depth_range[:, 0].view(-1, 1, 1)
            depth_max = depth_range[:, 1].view(-1, 1, 1)
            pred_metric = shape_norm * (depth_max - depth_min).clamp(min=1e-6) + depth_min
            sample_metrics.update(compute_metrics(pred_metric, depth_gt))
            sample_metrics["loss"] = F.l1_loss(pred_metric, depth_gt).item()
        elif depth_gt is not None:
            depth_gt = depth_gt.to(args.device)
            if shape_norm.dim() == 3:
                shape_norm = shape_norm.unsqueeze(1)
            if depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(1)
            sample_metrics["normalized_loss"] = F.l1_loss(shape_norm, depth_gt).item()

        sample_metrics["uncertainty_mean"] = float(output.uncertainty.mean().item()) if output.uncertainty is not None else 0.0
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
    parser = argparse.ArgumentParser(description="Evaluate FocalDiffusion model")
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
