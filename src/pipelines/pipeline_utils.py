"""
Pipeline utilities for FocalDiffusion
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


logger = logging.getLogger(__name__)


def _get_transformer_training_mode(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Return the checkpoint transformer training mode, if present."""
    if not isinstance(config, dict):
        return None

    training_config = config.get('training')
    if not isinstance(training_config, dict):
        return None

    trainable_modules = training_config.get('trainable_modules')
    if not isinstance(trainable_modules, dict):
        return None

    return trainable_modules.get('transformer')


def _state_dict_has_lora(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Check whether a transformer state dict contains PEFT LoRA weights."""
    return any('lora_' in key for key in state_dict)


def _ensure_transformer_lora(
        pipeline: 'FocalDiffusionPipeline',
        config: Optional[Dict[str, Any]],
        transformer_state_dict: Dict[str, torch.Tensor],
) -> None:
    """Rebuild the checkpoint LoRA adapter structure before loading weights."""
    transformer_mode = _get_transformer_training_mode(config)
    if transformer_mode != 'lora' and not _state_dict_has_lora(transformer_state_dict):
        return

    if hasattr(pipeline.transformer, 'peft_config'):
        return

    from peft import LoraConfig, get_peft_model

    training_config = {}
    if isinstance(config, dict) and isinstance(config.get('training'), dict):
        training_config = config['training']

    lora_config = LoraConfig(
        r=training_config.get('lora_rank', 8),
        lora_alpha=training_config.get('lora_alpha', 16),
        target_modules=training_config.get(
            'lora_target_modules',
            ['to_q', 'to_k', 'to_v', 'to_out.0'],
        ),
        lora_dropout=training_config.get('lora_dropout', 0.1),
    )
    pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
    logger.info("Rebuilt transformer LoRA adapters before loading checkpoint")


def load_pipeline(
        checkpoint_path: Union[str, Path],
        base_model_id: str = "stabilityai/stable-diffusion-3.5-large",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
) -> 'FocalDiffusionPipeline':
    """Load FocalDiffusion pipeline from checkpoint"""
    from .focal_diffusion_pipeline import FocalDiffusionPipeline
    from ..utils.env_utils import ensure_sentencepiece_installed

    # Load base pipeline
    ensure_sentencepiece_installed()
    pipeline = FocalDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config before model weights so adapter structure can be restored.
    checkpoint_config = checkpoint.get('config')
    if checkpoint_config is not None:
        pipeline.config = checkpoint_config

    # Load component weights
    if 'focal_processor_state_dict' in checkpoint:
        pipeline.focal_processor.load_state_dict(checkpoint['focal_processor_state_dict'])
    if getattr(pipeline, 'camera_encoder', None) is not None and 'camera_encoder_state_dict' in checkpoint:
        pipeline.camera_encoder.load_state_dict(checkpoint['camera_encoder_state_dict'], strict=False)
    if 'dual_decoder_state_dict' in checkpoint:
        pipeline.dual_decoder.load_state_dict(checkpoint['dual_decoder_state_dict'], strict=False)
    if 'focus_consistency_critic_state_dict' in checkpoint and hasattr(pipeline, 'focus_consistency_critic'):
        pipeline.focus_consistency_critic.load_state_dict(checkpoint['focus_consistency_critic_state_dict'], strict=False)
    if 'transformer_state_dict' in checkpoint:
        transformer_state_dict = checkpoint['transformer_state_dict']
        _ensure_transformer_lora(pipeline, checkpoint_config, transformer_state_dict)
        missing, unexpected = pipeline.transformer.load_state_dict(transformer_state_dict, strict=False)
        logger.info("Loaded transformer: missing=%s unexpected=%s", missing, unexpected)

    return pipeline


def save_pipeline(
        pipeline: 'FocalDiffusionPipeline',
        save_path: Union[str, Path],
        save_full_model: bool = True,
) -> None:
    """Save FocalDiffusion pipeline to checkpoint"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'focal_processor_state_dict': pipeline.focal_processor.state_dict(),
        **({'camera_encoder_state_dict': pipeline.camera_encoder.state_dict()} if getattr(pipeline, 'camera_encoder', None) is not None else {}),
        'dual_decoder_state_dict': pipeline.dual_decoder.state_dict(),
        **({'focus_consistency_critic_state_dict': pipeline.focus_consistency_critic.state_dict()} if hasattr(pipeline, 'focus_consistency_critic') else {}),
    }

    if save_full_model:
        checkpoint['transformer_state_dict'] = pipeline.transformer.state_dict()

    if hasattr(pipeline, 'config'):
        checkpoint['config'] = pipeline.config

    torch.save(checkpoint, save_path)
