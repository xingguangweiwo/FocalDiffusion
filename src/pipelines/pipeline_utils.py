"""
Pipeline utilities for FocalDiffusion
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from ..utils import ensure_sentencepiece_installed

logger = logging.getLogger(__name__)


def _get_transformer_training_mode(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the checkpoint transformer training mode, if it is recorded."""
    if not isinstance(config, dict):
        return None

    training_config = config.get("training")
    if not isinstance(training_config, dict):
        return None

    trainable_modules = training_config.get("trainable_modules")
    if not isinstance(trainable_modules, dict):
        return None

    transformer_mode = trainable_modules.get("transformer")
    return transformer_mode if isinstance(transformer_mode, str) else None


def _checkpoint_uses_lora(checkpoint: Dict[str, Any]) -> bool:
    """Detect whether a checkpoint was saved from a LoRA-wrapped transformer."""
    transformer_state_dict = checkpoint.get("transformer_state_dict")
    if not isinstance(transformer_state_dict, dict):
        return False

    if _get_transformer_training_mode(checkpoint.get("config")) == "lora":
        return True

    return any("lora_" in key for key in transformer_state_dict)


def _configure_lora_from_checkpoint(pipeline: "FocalDiffusionPipeline", checkpoint: Dict[str, Any]) -> None:
    """Rebuild the checkpoint's LoRA adapter structure before loading weights."""
    if not _checkpoint_uses_lora(checkpoint):
        return

    config = checkpoint.get("config") or {}
    training_config = config.get("training", {}) if isinstance(config, dict) else {}

    from peft import LoraConfig, PeftModel, get_peft_model

    if isinstance(pipeline.transformer, PeftModel):
        logger.info("Transformer already has a PEFT adapter; skipping LoRA reconfiguration")
        return

    lora_config = LoraConfig(
        r=training_config.get("lora_rank", 8),
        lora_alpha=training_config.get("lora_alpha", 16),
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=training_config.get("lora_dropout", 0.1),
    )
    pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
    pipeline.register_modules(transformer=pipeline.transformer)
    logger.info(
        "Configured transformer LoRA before checkpoint load: rank=%s alpha=%s dropout=%s",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.lora_dropout,
    )


def load_pipeline(
        checkpoint_path: Union[str, Path],
        base_model_id: str = "stabilityai/stable-diffusion-3.5-large",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
) -> 'FocalDiffusionPipeline':
    """Load FocalDiffusion pipeline from checkpoint"""
    from .focal_diffusion_pipeline import FocalDiffusionPipeline

    # Load base pipeline
    ensure_sentencepiece_installed()
    pipeline = FocalDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Rebuild train-time adapter structures such as LoRA before restoring weights.
    _configure_lora_from_checkpoint(pipeline, checkpoint)

    # Load component weights
    if 'focal_processor_state_dict' in checkpoint:
        pipeline.focal_processor.load_state_dict(checkpoint['focal_processor_state_dict'])
    if 'camera_encoder_state_dict' in checkpoint:
        pipeline.camera_encoder.load_state_dict(checkpoint['camera_encoder_state_dict'])
    if 'dual_decoder_state_dict' in checkpoint:
        pipeline.dual_decoder.load_state_dict(checkpoint['dual_decoder_state_dict'])
    if 'transformer_state_dict' in checkpoint:
        missing, unexpected = pipeline.transformer.load_state_dict(
            checkpoint['transformer_state_dict'],
            strict=False,
        )
        logger.info("Loaded transformer: missing=%s unexpected=%s", missing, unexpected)

    # Load config if available after module registration is complete.
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
