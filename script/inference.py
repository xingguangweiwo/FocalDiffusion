#!/usr/bin/env python3
"""
Inference script for FocalDiffusion
Processes focal stacks to generate depth maps and all-in-focus images
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import json
import time
from typing import List, Dict, Optional, Union
import logging

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines import FocalDiffusionPipeline, load_pipeline
from src.utils import (
    load_image_stack,
    save_depth_map,
    save_all_in_focus,
    create_visualization,
    parse_exif_data,
    estimate_focus_distances,
    ensure_sentencepiece_installed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run inference with FocalDiffusion model"
    )

    # Model arguments
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to model checkpoint (uses pretrained if not specified)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to inference configuration file'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='stabilityai/stable-diffusion-3.5-large-tensorrt',
        help='Base SD3.5 model ID'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda/cpu)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='fp16',
        choices=['fp16', 'fp32', 'bf16'],
        help='Model precision'
    )

    # Input/output arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input focal stack (directory or comma-separated image paths)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--input-list',
        type=str,
        help='Text file containing list of input directories for batch processing'
    )

    # Focal stack parameters
    parser.add_argument(
        '--focus-distances',
        type=str,
        help='Comma-separated focus distances in meters (auto-detect if not specified)'
    )
    parser.add_argument(
        '--focal-length',
        type=float,
        help='Camera focal length in mm'
    )
    parser.add_argument(
        '--aperture',
        type=float,
        help='Camera aperture (f-number)'
    )
    parser.add_argument(
        '--sensor-size',
        type=float,
        default=0.036,
        help='Sensor size in meters (default: full frame)'
    )

    # Inference parameters
    parser.add_argument(
        '--num-inference-steps',
        type=int,
        default=50,
        help='Number of denoising steps'
    )
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.0,
        help='Guidance scale for inference'
    )
    parser.add_argument(
        '--ensemble-size',
        type=int,
        default=1,
        help='Number of predictions to ensemble for uncertainty'
    )
    parser.add_argument(
        '--camera-invariant-mode',
        type=str,
        default='relative',
        choices=['relative', 'normalized', 'learned'],
        help='Camera invariance mode'
    )

    # Processing options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--save-visualization',
        action='store_true',
        help='Save visualization figures'
    )
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate outputs'
    )

    return parser.parse_args()


def load_focal_stack(input_path: str) -> tuple:
    """
    Load focal stack from directory or list of files

    Returns:
        images: List of PIL images
        image_paths: List of image paths
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        # Load all images from directory
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(sorted(input_path.glob(ext)))

        if not image_paths:
            raise ValueError(f"No images found in {input_path}")

    elif ',' in str(input_path):
        # Comma-separated list of paths
        image_paths = [Path(p.strip()) for p in str(input_path).split(',')]

    else:
        # Single image - look for other images in the same directory
        parent_dir = input_path.parent
        suffix = input_path.suffix

        # Find similar named files
        pattern = input_path.stem.rstrip('0123456789') + '*' + suffix
        image_paths = sorted(parent_dir.glob(pattern))

        if len(image_paths) < 2:
            raise ValueError(f"Need at least 2 images for focal stack, found {len(image_paths)}")

    # Load images
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        images.append(img)

    logger.info(f"Loaded {len(images)} images from focal stack")

    return images, [str(p) for p in image_paths]


def extract_camera_params(image_paths: List[str], args) -> Dict:
    """Extract camera parameters from EXIF or arguments"""
    camera_params = {}

    # Try to get from EXIF first
    if image_paths:
        exif_data = parse_exif_data(image_paths[0])
        if exif_data:
            camera_params = {
                'focal_length': exif_data.get('focal_length', args.focal_length or 50.0) / 1000,
                'aperture': exif_data.get('aperture', args.aperture or 2.8),
                'sensor_size': exif_data.get('sensor_size', args.sensor_size),
            }

    # Override with command line arguments
    if args.focal_length:
        camera_params['focal_length'] = args.focal_length / 1000  # Convert to meters
    if args.aperture:
        camera_params['aperture'] = args.aperture
    if args.sensor_size:
        camera_params['sensor_size'] = args.sensor_size

    # Use defaults if still missing
    if not camera_params:
        camera_params = {
            'focal_length': 0.050,  # 50mm
            'aperture': 2.8,
            'sensor_size': 0.036,  # Full frame
        }

    return camera_params


def extract_focus_distances(
        image_paths: List[str],
        num_images: int,
        args
) -> torch.Tensor:
    """Extract or estimate focus distances"""

    if args.focus_distances:
        # Parse from command line
        distances = [float(x) for x in args.focus_distances.split(',')]

        if len(distances) != num_images:
            logger.warning(
                f"Number of focus distances ({len(distances)}) doesn't match "
                f"number of images ({num_images}). Using estimation instead."
            )
            distances = estimate_focus_distances(num_images)
    else:
        # Try to extract from EXIF
        distances = []
        for img_path in image_paths:
            exif_data = parse_exif_data(img_path)
            if exif_data and 'focus_distance' in exif_data:
                distances.append(exif_data['focus_distance'])

        if len(distances) != num_images:
            # Fallback to estimation
            logger.info("Estimating focus distances...")
            distances = estimate_focus_distances(num_images)

    return torch.tensor(distances, dtype=torch.float32)


def process_focal_stack(
        pipeline,
        images: List[Image.Image],
        focus_distances: torch.Tensor,
        camera_params: Dict,
        args,
        stack_name: str = "focal_stack"
) -> Dict:
    """Process a single focal stack"""

    logger.info(f"Processing {stack_name}...")
    start_time = time.time()

    # Run inference
    with torch.no_grad():
        output = pipeline(
            focal_stack=images,
            focus_distances=focus_distances,
            camera_params=camera_params,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            ensemble_size=args.ensemble_size,
            camera_invariant_mode=args.camera_invariant_mode,
            output_type="pil" if args.save_visualization else "pt",
            return_dict=True,
        )

    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f} seconds")

    # Prepare results
    results = {
        'name': stack_name,
        'depth_map': output.depth_map,
        'all_in_focus': output.all_in_focus_image,
        'depth_colored': output.depth_colored,
        'uncertainty': output.uncertainty,
        'inference_time': inference_time,
        'camera_params': camera_params,
        'focus_distances': focus_distances.tolist(),
    }

    if args.save_intermediate:
        results['focal_features'] = output.focal_features
        results['attention_maps'] = output.attention_maps

    return results


def save_results(results: Dict, output_dir: Path, args):
    """Save processing results"""

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    name = results['name']

    # Save depth map
    if isinstance(results['depth_map'], torch.Tensor):
        depth_np = results['depth_map'].cpu().numpy()
    else:
        depth_np = results['depth_map']

    if depth_np.ndim == 3:
        depth_np = depth_np[0]  # Remove batch dimension

    # Save as numpy array
    np.save(output_dir / f"{name}_depth.npy", depth_np)

    # Save as image
    save_depth_map(depth_np, output_dir / f"{name}_depth.png")

    # Save colored depth if available
    if results['depth_colored'] is not None:
        if isinstance(results['depth_colored'], Image.Image):
            results['depth_colored'].save(output_dir / f"{name}_depth_colored.png")

    # Save all-in-focus image
    if isinstance(results['all_in_focus'], Image.Image):
        results['all_in_focus'].save(output_dir / f"{name}_all_in_focus.png")
    else:
        save_all_in_focus(results['all_in_focus'], output_dir / f"{name}_all_in_focus.png")

    # Save uncertainty if available
    if results.get('uncertainty') is not None:
        uncertainty = results['uncertainty']
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.cpu().numpy()
        np.save(output_dir / f"{name}_uncertainty.npy", uncertainty)

    # Save metadata
    metadata = {
        'name': name,
        'camera_params': results['camera_params'],
        'focus_distances': results['focus_distances'],
        'inference_time': results['inference_time'],
    }

    with open(output_dir / f"{name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create visualization if requested
    if args.save_visualization:
        from src.utils.visualization import visualize_results

        viz_outputs = {
            'depth': depth_np,
            'depth_colored': results['depth_colored'],
            'all_in_focus': results['all_in_focus'],
        }

        if results.get('uncertainty') is not None:
            viz_outputs['uncertainty'] = results['uncertainty']

        visualize_results(
            viz_outputs,
            save_path=output_dir / f"{name}_visualization.png",
            show=False,
        )

    logger.info(f"Results saved to {output_dir}")


def main():
    """Main inference function"""
    args = parse_args()

    # Setup device and dtype
    device = torch.device(args.device)

    dtype_map = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Load pipeline
    if args.model_path:
        logger.info(f"Loading model from checkpoint: {args.model_path}")
        pipeline = load_pipeline(
            checkpoint_path=args.model_path,
            base_model_id=args.base_model,
            device=device,
            dtype=dtype,
        )
    else:
        logger.info(f"Loading pretrained model: {args.base_model}")
        ensure_sentencepiece_installed()
        pipeline = FocalDiffusionPipeline.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
        ).to(device)

    pipeline.eval()

    # Prepare input list
    if args.input_list:
        # Batch processing from list
        with open(args.input_list, 'r') as f:
            input_paths = [line.strip() for line in f if line.strip()]
    else:
        # Single input
        input_paths = [args.input]

    # Process each input
    output_dir = Path(args.output)
    all_results = []

    for input_idx, input_path in enumerate(tqdm(input_paths, desc="Processing")):
        try:
            # Load focal stack
            images, image_paths = load_focal_stack(input_path)

            # Extract parameters
            camera_params = extract_camera_params(image_paths, args)
            focus_distances = extract_focus_distances(image_paths, len(images), args)

            # Convert camera params to tensors
            camera_params_tensor = {
                k: torch.tensor(v, dtype=dtype, device=device)
                for k, v in camera_params.items()
            }

            # Process
            stack_name = Path(input_path).stem if Path(input_path).is_dir() else f"stack_{input_idx:04d}"

            results = process_focal_stack(
                pipeline,
                images,
                focus_distances.to(device),
                camera_params_tensor,
                args,
                stack_name,
            )

            # Save results
            stack_output_dir = output_dir / stack_name if len(input_paths) > 1 else output_dir
            save_results(results, stack_output_dir, args)

            all_results.append(results)

        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            continue

    # Save summary if batch processing
    if len(all_results) > 1:
        summary = {
            'num_processed': len(all_results),
            'average_time': np.mean([r['inference_time'] for r in all_results]),
            'total_time': sum(r['inference_time'] for r in all_results),
            'stacks': [r['name'] for r in all_results],
        }

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nProcessed {len(all_results)} focal stacks")
        logger.info(f"Average inference time: {summary['average_time']:.2f} seconds")

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()