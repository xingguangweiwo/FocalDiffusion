"""Stable Diffusion 3 flow-matching training objective helpers.

SD3/SD3.5 transformers are trained with a rectified-flow target instead of a
DDPM epsilon-prediction target.  The noisy sample is constructed as a linear
interpolation between the clean latent and Gaussian noise:

    x_t = (1 - sigma) * x_0 + sigma * noise

The transformer target is the flow direction:

    target = noise - x_0

Given a predicted flow, the clean latent estimate used by auxiliary heads is:

    x_0_hat = x_t - sigma * predicted_flow
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SD3FlowMatchingBatch:
    """Tensors needed to train an SD3 flow-matching step."""

    noisy_latents: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor
    sigmas: torch.Tensor
    target: torch.Tensor


def _as_broadcastable_sigmas(sigmas: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """Reshape per-sample sigmas so they broadcast over latent channels/spatial dims."""

    sigmas = sigmas.to(device=latents.device, dtype=latents.dtype)
    while sigmas.ndim < latents.ndim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas


def sample_sd3_flow_matching_batch(
    scheduler: object,
    latents: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> SD3FlowMatchingBatch:
    """Sample SD3/FlowMatch timesteps and targets for one training batch.

    The helper uses the scheduler's sigma schedule, mirroring SD3 fine-tuning:
    sample a schedule index, gather the corresponding timestep and sigma, mix
    clean latents with Gaussian noise, and regress the flow ``noise - latents``.
    """

    if noise is None:
        noise = torch.randn_like(latents)

    if not hasattr(scheduler, "set_timesteps"):
        raise AttributeError("SD3 flow matching requires a scheduler with set_timesteps().")

    num_train_timesteps = getattr(getattr(scheduler, "config", None), "num_train_timesteps", None)
    if num_train_timesteps is None:
        raise AttributeError("Scheduler is missing config.num_train_timesteps for SD3 training.")

    # Reset to the full training schedule rather than reusing a shortened
    # inference schedule that may have been installed on the shared scheduler.
    scheduler.set_timesteps(num_train_timesteps, device=latents.device)
    timesteps_attr = getattr(scheduler, "timesteps", None)
    sigmas_attr = getattr(scheduler, "sigmas", None)

    if timesteps_attr is None or sigmas_attr is None:
        raise AttributeError("Scheduler must expose timesteps and sigmas for SD3 flow matching.")

    timesteps_schedule = torch.as_tensor(timesteps_attr, device=latents.device)
    sigmas_schedule = torch.as_tensor(sigmas_attr, device=latents.device, dtype=latents.dtype)
    if timesteps_schedule.ndim != 1 or timesteps_schedule.numel() == 0:
        raise RuntimeError("Scheduler timesteps must be a non-empty 1D tensor.")
    if sigmas_schedule.ndim != 1 or sigmas_schedule.numel() == 0:
        raise RuntimeError("Scheduler sigmas must be a non-empty 1D tensor.")

    # Some FlowMatch schedulers include a final terminal sigma for inference.
    max_index = min(timesteps_schedule.shape[0], sigmas_schedule.shape[0]) - 1
    step_indices = torch.randint(
        0,
        max_index + 1,
        (latents.shape[0],),
        device=latents.device,
        dtype=torch.long,
    )
    timesteps = timesteps_schedule.index_select(0, step_indices).to(dtype=torch.long)
    sigmas = sigmas_schedule.index_select(0, step_indices)
    sigma_broadcast = _as_broadcastable_sigmas(sigmas, latents)

    noisy_latents = (1.0 - sigma_broadcast) * latents + sigma_broadcast * noise
    target = noise - latents

    return SD3FlowMatchingBatch(
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        noise=noise,
        sigmas=sigmas,
        target=target,
    )


def predict_clean_latents_from_flow(
    noisy_latents: torch.Tensor,
    model_pred: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Recover a clean latent estimate from an SD3 flow prediction."""

    sigma_broadcast = _as_broadcastable_sigmas(sigmas, noisy_latents)
    return noisy_latents - sigma_broadcast * model_pred
