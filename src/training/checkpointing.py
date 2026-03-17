"""Checkpointing helpers for :mod:`src.training.trainer`."""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    trainer: "FocalDiffusionTrainer",
    epoch: int,
    global_step: int,
    is_best: bool = False,
    is_final: bool = False,
) -> None:
    """Save model checkpoint from trainer state."""
    if not trainer.accelerator.is_main_process:
        return

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'focal_processor_state_dict': trainer.accelerator.unwrap_model(trainer.focal_processor).state_dict(),
        'camera_encoder_state_dict': trainer.accelerator.unwrap_model(trainer.camera_encoder).state_dict(),
        'dual_decoder_state_dict': trainer.accelerator.unwrap_model(trainer.dual_decoder).state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
        'config': trainer.config,
    }

    if trainer.config['training']['trainable_modules']['transformer'] != 'frozen':
        checkpoint['transformer_state_dict'] = trainer.accelerator.unwrap_model(
            trainer.pipeline.transformer
        ).state_dict()

    if trainer.ema is not None:
        checkpoint['ema_state_dict'] = trainer.ema.state_dict()

    if is_best:
        filename = 'best.pt'
    elif is_final:
        filename = 'final.pt'
    else:
        filename = f'checkpoint_epoch_{epoch}.pt'

    save_path = trainer.checkpoint_dir / filename
    torch.save(checkpoint, save_path)
    logger.info("Saved checkpoint: %s", filename)

    if trainer.config['output'].get('save_top_k'):
        cleanup_checkpoints(trainer)


def cleanup_checkpoints(trainer: "FocalDiffusionTrainer") -> None:
    """Keep only the latest configured checkpoint files."""
    save_top_k = trainer.config['output'].get('save_top_k', 3)
    checkpoints = list(trainer.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    checkpoints.sort(key=lambda item: int(item.stem.split('_')[-1]))

    if len(checkpoints) > save_top_k:
        for checkpoint in checkpoints[:-save_top_k]:
            checkpoint.unlink()
            logger.info("Removed old checkpoint: %s", checkpoint.name)


def load_checkpoint(trainer: "FocalDiffusionTrainer", checkpoint_path: str) -> Tuple[int, int]:
    """Load checkpoint and restore optimizer/scheduler/model states."""
    logger.info("Loading checkpoint from %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=trainer.accelerator.device)

    trainer.focal_processor.load_state_dict(checkpoint['focal_processor_state_dict'])
    trainer.camera_encoder.load_state_dict(checkpoint['camera_encoder_state_dict'])
    trainer.dual_decoder.load_state_dict(checkpoint['dual_decoder_state_dict'])

    if 'transformer_state_dict' in checkpoint:
        trainer.pipeline.transformer.load_state_dict(checkpoint['transformer_state_dict'])

    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if trainer.ema is not None and 'ema_state_dict' in checkpoint:
        trainer.ema.load_state_dict(checkpoint['ema_state_dict'])

    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    logger.info("Resumed from epoch %s, global step %s", epoch, global_step)
    return epoch, global_step
