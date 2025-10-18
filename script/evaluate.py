"""
Evaluation script for FocalDiffusion
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import yaml

from src.pipelines import load_pipeline
from src.data import create_dataloader
from src.utils.metrics import compute_metrics
from src.utils.visualization import visualize_results

def evaluate(args):
    """Main evaluation function"""
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load pipeline
    print(f"Loading model from {args.checkpoint}...")
    pipeline = load_pipeline(
        checkpoint_path=args.checkpoint,
        base_model_id=config['model']['base_model_id'],
        device=args.device,
        dtype=torch.float16 if args.fp16 else torch.float32,
    )
    pipeline.eval()

    # Create dataloader
    print(f"Loading {args.dataset} dataset...")
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

    # Evaluation metrics
    all_metrics = []

    # Evaluation loop
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if args.max_samples and batch_idx * args.batch_size >= args.max_samples:
            break

        # Move to device
        focal_stack = batch['focal_stack'].to(args.device)
        focus_distances = batch['focus_distances'].to(args.device)
        depth_gt = batch['depth'].to(args.device)

        # Generate predictions
        with torch.no_grad():
            output = pipeline(
                focal_stack=focal_stack,
                focus_distances=focus_distances,
                camera_params=batch.get('camera_params'),
                num_inference_steps=args.num_inference_steps,
                output_type='pt',
            )

        # Compute metrics
        metrics = compute_metrics(
            output.depth_map,
            depth_gt.squeeze(1)
        )
        all_metrics.append(metrics)

        # Save visualizations for first few batches
        if args.save_viz and batch_idx < args.num_viz:
            viz_path = Path(args.output_dir) / f"viz_batch_{batch_idx}.png"
            visualize_results(
                {
                    'focal_stack': focal_stack[0].cpu(),
                    'depth': output.depth_map[0].cpu(),
                    'depth_gt': depth_gt[0].cpu(),
                    'all_in_focus': output.all_in_focus_image,
                },
                save_path=viz_path,
                show=False,
            )

    # Aggregate metrics
    final_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        final_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
        }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Print results
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
    parser.add_argument('--save_viz', action='store_true', help='Save visualizations')
    parser.add_argument('--num_viz', type=int, default=10, help='Number of visualizations')

    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()