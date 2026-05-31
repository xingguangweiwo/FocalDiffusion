#!/usr/bin/env python3
"""
Inference script for FocalDiffusion
Processes focal stacks to generate depth maps and all-in-focus images
"""

import argparse
import os
import sys
from pathlib import Path
import json
import time
from typing import Dict, List
import logging

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines import load_pipeline
from src.utils import (
    save_depth_map,
    save_all_in_focus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resize_or_pad_to_multiple(
    images: List[Image.Image],
    divisor: int = 16,
) -> tuple[List[Image.Image], Dict[str, List[int]]]:
    """Resize with preserved aspect ratio then pad to divisor multiples."""
    if not images:
        raise ValueError("images must not be empty")

    orig_w, orig_h = images[0].size
    for image in images[1:]:
        if image.size != (orig_w, orig_h):
            raise ValueError("All focal-stack images must share the same size")

    target_h = ((orig_h + divisor - 1) // divisor) * divisor
    target_w = ((orig_w + divisor - 1) // divisor) * divisor
    pad_h = target_h - orig_h
    pad_w = target_w - orig_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_h == 0 and pad_w == 0:
        return images, {
            "original_size": [orig_h, orig_w],
            "inference_size": [target_h, target_w],
            "pad": [top, bottom, left, right],
        }

    padded: List[Image.Image] = []
    for image in images:
        arr = np.array(image)
        arr = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="edge")
        padded.append(Image.fromarray(arr))

    return padded, {
        "original_size": [orig_h, orig_w],
        "inference_size": [target_h, target_w],
        "pad": [top, bottom, left, right],
    }


def unpad_tensor_spatial(tensor: torch.Tensor, pad: List[int]) -> torch.Tensor:
    top, bottom, left, right = pad
    h_end = tensor.shape[-2] - bottom if bottom > 0 else tensor.shape[-2]
    w_end = tensor.shape[-1] - right if right > 0 else tensor.shape[-1]
    return tensor[..., top:h_end, left:w_end]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run inference with FocalDiffusion model"
    )

    # Model arguments
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained FocalDiffusion checkpoint (required for meaningful inference)'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='stabilityai/stable-diffusion-3.5-large',
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
        help='Comma-separated focus distances in meters (uses 0..N-1 index spacing if not specified)'
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
        default=1.0,
        help='Guidance scale for inference (1.0 disables text classifier-free guidance)'
    )
    parser.add_argument('--height', type=int, default=None, help='Optional inference height override')
    parser.add_argument('--width', type=int, default=None, help='Optional inference width override')

    # Processing options
    parser.add_argument(
        '--save-visualization',
        action='store_true',
        help='Save visualization figures'
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


def extract_focus_distances(
        num_images: int,
        args
) -> torch.Tensor:
    """Extract focus distances from CLI values or use index-spaced positions."""

    if args.focus_distances:
        distances = [float(x) for x in args.focus_distances.split(',')]
        if len(distances) != num_images:
            raise ValueError(
                f"Number of focus distances ({len(distances)}) must match "
                f"number of images ({num_images})."
            )
    else:
        distances = list(np.linspace(0.0, float(num_images - 1), num_images))

    return torch.tensor(distances, dtype=torch.float32)


def process_focal_stack(
        pipeline,
        images: List[Image.Image],
        focus_distances: torch.Tensor,
        args,
        stack_name: str = "focal_stack"
) -> Dict:
    """Process a single focal stack"""

    logger.info(f"Processing {stack_name}...")
    start_time = time.time()

    padded_images, size_meta = resize_or_pad_to_multiple(images, divisor=16)

    pipeline_kwargs = {}
    if args.height is not None:
        pipeline_kwargs["height"] = args.height
    if args.width is not None:
        pipeline_kwargs["width"] = args.width

    # Run inference
    with torch.no_grad():
        output = pipeline(
            focal_stack=padded_images,
            focus_distances=focus_distances,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            output_type="pil" if args.save_visualization else "pt",
            return_dict=True,
            **pipeline_kwargs,
        )

    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f} seconds")

    if isinstance(output.depth_map, torch.Tensor):
        output.depth_map = unpad_tensor_spatial(output.depth_map, size_meta["pad"])
    if isinstance(output.uncertainty, torch.Tensor):
        output.uncertainty = unpad_tensor_spatial(output.uncertainty, size_meta["pad"])
    if isinstance(output.all_in_focus_image, torch.Tensor):
        output.all_in_focus_image = unpad_tensor_spatial(output.all_in_focus_image, size_meta["pad"])
    elif isinstance(output.all_in_focus_image, Image.Image):
        top, bottom, left, right = size_meta["pad"]
        w, h = output.all_in_focus_image.size
        output.all_in_focus_image = output.all_in_focus_image.crop((left, top, w - right, h - bottom))

    # Prepare results
    results = {
        'name': stack_name,
        'depth_map': output.depth_map,
        'all_in_focus': output.all_in_focus_image,
        'depth_colored': output.depth_colored,
        'uncertainty': output.uncertainty,
        'inference_time': inference_time,
        'focus_distances': focus_distances.tolist(),
        'original_size': size_meta["original_size"],
        'inference_size': size_meta["inference_size"],
        'output_size': list(output.depth_map.shape[-2:]) if isinstance(output.depth_map, torch.Tensor) else size_meta["original_size"],
    }

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
        'focus_distances': results['focus_distances'],
        'inference_time': results['inference_time'],
        'original_size': results.get('original_size'),
        'inference_size': results.get('inference_size'),
        'output_size': results.get('output_size'),
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

    if not args.model_path:
        raise ValueError("A trained FocalDiffusion checkpoint is required.")
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")

    # Load pipeline
    logger.info(f"Loading model from checkpoint: {args.model_path}")
    pipeline = load_pipeline(
        checkpoint_path=args.model_path,
        base_model_id=args.base_model,
        device=device,
        dtype=dtype,
    )

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
            images, _ = load_focal_stack(input_path)

            # Extract parameters
            focus_distances = extract_focus_distances(len(images), args)

            # Process
            stack_name = Path(input_path).stem if Path(input_path).is_dir() else f"stack_{input_idx:04d}"

            results = process_focal_stack(
                pipeline,
                images,
                focus_distances.to(device),
                args,
                stack_name,
            )

            # Save results
            stack_output_dir = output_dir / stack_name if len(input_paths) > 1 else output_dir
            save_results(results, stack_output_dir, args)

            all_results.append(results)

        except ValueError:
            raise
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
