"""
Pipeline utilities for FSDiffusion
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



def _model_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the model section from a checkpoint config, if available."""
    if not isinstance(config, dict):
        return {}
    model_config = config.get('model', config)
    return model_config if isinstance(model_config, dict) else {}


def _build_focal_modules_from_config(config: Optional[Dict[str, Any]], vae: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """Build focal modules using the architecture stored in a checkpoint config.

    This avoids shape mismatches when inference checkpoints were trained with
    non-default focal-module dimensions such as ``feature_dim=128``.
    """
    from ..models.dual_decoder import DualOutputDecoder
    from ..models.focal_evidence import FocalEvidenceHead, PhysicalSupportHead
    from ..models.focal_processor import FocalStackProcessor

    model_config = _model_config(config)
    latent_channels = int(getattr(getattr(vae, 'config', object()), 'latent_channels', 16))

    return {
        'focal_processor': FocalStackProcessor(
            feature_dim=int(model_config.get('feature_dim', 512)),
            max_sequence_length=int(model_config.get('max_focal_stack_size', 100)),
            focal_encoder_type=str(model_config.get('focal_encoder_type', 'focal_sweep')),
            patch_size=int(model_config.get('patch_size', 8)),
            focal_attention_heads=int(model_config.get('focal_attention_heads', 8)),
            focal_attention_depth=int(model_config.get('focal_attention_depth', 2)),
        ),
        'focal_evidence_head': FocalEvidenceHead(
            hidden=int(model_config.get('focal_evidence_hidden', 48)),
            temperature=float(model_config.get('focal_evidence_temperature', 0.07)),
        ),
        'physical_support_head': PhysicalSupportHead(
            in_channels=5,
            hidden=int(model_config.get('physical_support_hidden', 16)),
        ),
        'dual_decoder': DualOutputDecoder(
            in_channels=latent_channels,
            out_channels_depth=1,
            out_channels_rgb=latent_channels,
        ),
    }

def load_pipeline(
        checkpoint_path: Union[str, Path],
        base_model_id: str = "stabilityai/stable-diffusion-3.5-large",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
) -> 'FocalDiffusionPipeline':
    """Load FSDiffusion pipeline from checkpoint."""
    from .focal_diffusion_pipeline import FocalDiffusionPipeline
    from ..utils.env_utils import ensure_sentencepiece_installed

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get('config')
    if checkpoint_config is None:
        logger.warning(
            "Checkpoint %s does not contain a config; falling back to default focal modules. "
            "This may fail if the checkpoint used non-default focal-module dimensions.",
            checkpoint_path,
        )

    ensure_sentencepiece_installed()
    pipeline = FocalDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    )

    if checkpoint_config is not None:
        modules = _build_focal_modules_from_config(checkpoint_config, pipeline.vae)
        pipeline.focal_processor = modules['focal_processor']
        pipeline.focal_evidence_head = modules['focal_evidence_head']
        pipeline.dual_decoder = modules['dual_decoder']
        pipeline.physical_support_head = modules['physical_support_head']
        pipeline.config = checkpoint_config

        condition_channels = getattr(pipeline.focal_processor, 'feature_dim', None)
        if hasattr(pipeline.transformer, 'condition_adapter') and condition_channels is not None:
            # Rebuild the condition adapter to match the reconstructed focal processor.
            hidden_size = pipeline.transformer.config.attention_head_dim * pipeline.transformer.config.num_attention_heads
            pipeline.transformer.condition_channels = condition_channels
            pipeline.transformer.condition_adapter = torch.nn.Conv2d(condition_channels, hidden_size, kernel_size=1)

    pipeline = pipeline.to(device)

    # Load component weights after checkpoint-compatible modules have been created.
    if 'focal_processor_state_dict' in checkpoint:
        pipeline.focal_processor.load_state_dict(checkpoint['focal_processor_state_dict'])
    if 'focal_evidence_head_state_dict' in checkpoint:
        pipeline.focal_evidence_head.load_state_dict(checkpoint['focal_evidence_head_state_dict'], strict=False)
    if 'dual_decoder_state_dict' in checkpoint:
        pipeline.dual_decoder.load_state_dict(checkpoint['dual_decoder_state_dict'], strict=False)
    if 'physical_support_head_state_dict' in checkpoint:
        pipeline.physical_support_head.load_state_dict(checkpoint['physical_support_head_state_dict'], strict=False)
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
    """Save FSDiffusion pipeline to checkpoint"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'focal_processor_state_dict': pipeline.focal_processor.state_dict(),
        'focal_evidence_head_state_dict': pipeline.focal_evidence_head.state_dict(),
        'dual_decoder_state_dict': pipeline.dual_decoder.state_dict(),
        'physical_support_head_state_dict': pipeline.physical_support_head.state_dict(),
    }

    if save_full_model:
        checkpoint['transformer_state_dict'] = pipeline.transformer.state_dict()

    if hasattr(pipeline, 'config'):
        checkpoint['config'] = pipeline.config

    torch.save(checkpoint, save_path)
