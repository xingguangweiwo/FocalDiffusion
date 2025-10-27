"""
Pipeline utilities for FocalDiffusion
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Union
import json

def load_pipeline(
        checkpoint_path: Union[str, Path],
        base_model_id: str = "stabilityai/stable-diffusion-3.5-large-tensorrt",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
) -> 'FocalDiffusionPipeline':
    """Load FocalDiffusion pipeline from checkpoint"""
    from .focal_diffusion_pipeline import FocalDiffusionPipeline

    # Load base pipeline
    pipeline = FocalDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load component weights
    if 'focal_processor_state_dict' in checkpoint:
        pipeline.focal_processor.load_state_dict(checkpoint['focal_processor_state_dict'])
    if 'camera_encoder_state_dict' in checkpoint:
        pipeline.camera_encoder.load_state_dict(checkpoint['camera_encoder_state_dict'])
    if 'dual_decoder_state_dict' in checkpoint:
        pipeline.dual_decoder.load_state_dict(checkpoint['dual_decoder_state_dict'])

    # Load config if available
    if 'config' in checkpoint:
        pipeline.config = checkpoint['config']

    return pipeline


def save_pipeline(
        pipeline: 'FocalDiffusionPipeline',
        save_path: Union[str, Path],
        save_full_model: bool = False,
) -> None:
    """Save FocalDiffusion pipeline to checkpoint"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'focal_processor_state_dict': pipeline.focal_processor.state_dict(),
        'camera_encoder_state_dict': pipeline.camera_encoder.state_dict(),
        'dual_decoder_state_dict': pipeline.dual_decoder.state_dict(),
    }

    if save_full_model:
        checkpoint['transformer_state_dict'] = pipeline.transformer.state_dict()

    if hasattr(pipeline, 'config'):
        checkpoint['config'] = pipeline.config

    torch.save(checkpoint, save_path)