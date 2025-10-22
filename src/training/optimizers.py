"""
Optimizer and scheduler utilities
"""

import torch
from torch.optim import Optimizer
from typing import Any, Dict, Tuple


def get_optimizer(
        model_params,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        **kwargs
) -> Optimizer:
    """Get optimizer based on config"""

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model_params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model_params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model_params,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def get_scheduler(
        optimizer: Optimizer,
        scheduler_type: str = "cosine",
        num_training_steps: int = 1000,
        num_warmup_steps: int = 100,
        **kwargs
) -> Any:
    """Get learning rate scheduler"""
    from diffusers.optimization import get_scheduler as get_diffusers_scheduler

    return get_diffusers_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs
    )