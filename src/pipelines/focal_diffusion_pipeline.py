"""Inference pipeline that augments Stable Diffusion 3.5 with focal-stack conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline

try:
    from diffusers.models.transformers import SD3Transformer2DModel
except ImportError as err:  # pragma: no cover - import-time environment check
    raise ImportError(
        "FocalDiffusion requires diffusers>=0.28.0 so the Stable Diffusion 3 transformer is available. "
        "Upgrade via `pip install --upgrade diffusers`."
    ) from err

try:  # diffusers>=0.35 moved Transformer2DModelOutput to modeling_outputs
    from diffusers.models.transformers import Transformer2DModelOutput  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - compatibility shim for newer wheels
    try:
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
    except ImportError as err:
        raise ImportError(
            "FocalDiffusion requires diffusers to expose Transformer2DModelOutput. "
            "Upgrade via `pip install --upgrade diffusers`."
        ) from err
from diffusers.utils import BaseOutput
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from ..models.attention_modules import FocalCrossAttention
from ..models.camera_invariant import CameraInvariantEncoder
from ..models.dual_decoder import DualOutputDecoder
from ..models.focal_processor import FocalStackProcessor


@dataclass
class FocalDiffusionOutput(BaseOutput):
    """Output type returned by :class:`FocalDiffusionPipeline`."""

    depth_map: torch.Tensor
    all_in_focus_image: Union[torch.Tensor, Image.Image]
    depth_colored: Optional[Image.Image] = None
    uncertainty: Optional[torch.Tensor] = None
    attention_maps: Optional[Dict[str, torch.Tensor]] = None
    focal_features: Optional[Dict[str, torch.Tensor]] = None


class FocalInjectedSD3Transformer(nn.Module):
    """Wrapper around the SD3.5 transformer that accepts focal features."""

    def __init__(self, base_transformer: SD3Transformer2DModel) -> None:
        super().__init__()
        self.base_transformer = base_transformer
        self.config = base_transformer.config
        hidden_size = self.config.attention_head_dim * self.config.num_attention_heads
        self.focal_attn = FocalCrossAttention(
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.attention_head_dim,
        )

    def __getattr__(self, name: str) -> Any:
        # Delegate attribute access (e.g. to(), device, dtype, etc.) to the wrapped module.
        if name in {"base_transformer", "focal_attn", "config"}:
            return super().__getattribute__(name)
        return getattr(self.base_transformer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        focal_features: Optional[Dict[str, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        result = self.base_transformer.forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        if focal_features is not None and "fused_features" in focal_features:
            fused = focal_features["fused_features"]
            if fused.dim() == 5:
                # Some processors keep the stack dimension – collapse it.
                fused = fused.mean(dim=1)

            if fused.shape[-2:] != result.shape[-2:]:
                fused = F.interpolate(fused, size=result.shape[-2:], mode="bilinear", align_corners=False)

            fused_seq = fused.flatten(2).transpose(1, 2)
            hidden_seq = result.flatten(2).transpose(1, 2)
            hidden_seq = hidden_seq + self.focal_attn(hidden_seq, fused_seq)
            result = hidden_seq.transpose(1, 2).reshape_as(result)

        if return_dict:
            return Transformer2DModelOutput(sample=result)

        return (result,)


class FocalDiffusionPipeline(StableDiffusion3Pipeline):
    """Stable Diffusion 3.5 pipeline that consumes focal stacks instead of text-only prompts."""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        text_encoder_2: CLIPTextModelWithProjection,
        text_encoder_3: Optional[T5EncoderModel],
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        tokenizer_3: Optional[T5TokenizerFast],
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        focal_processor: Optional[nn.Module] = None,
        camera_encoder: Optional[nn.Module] = None,
        dual_decoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.focal_processor = focal_processor or FocalStackProcessor()
        self.camera_encoder = camera_encoder or CameraInvariantEncoder()
        self.dual_decoder = dual_decoder or DualOutputDecoder(
            in_channels=self.vae.config.latent_channels,
            out_channels_depth=1,
            out_channels_rgb=self.vae.config.latent_channels,
        )

        if not isinstance(self.transformer, FocalInjectedSD3Transformer):
            self.transformer = FocalInjectedSD3Transformer(self.transformer)

    @torch.no_grad()
    def __call__(
        self,
        focal_stack: Union[torch.Tensor, List[Image.Image]],
        focus_distances: torch.Tensor,
        camera_params: Optional[Dict[str, Union[torch.Tensor, float]]] = None,
        prompt: str = "",
        negative_prompt: Optional[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        camera_invariant_mode: str = "relative",
        **kwargs: Any,
    ) -> Union[FocalDiffusionOutput, Tuple[torch.Tensor, Union[torch.Tensor, Image.Image]]]:
        del kwargs  # unused hook for future extensions

        focal_stack = self._ensure_tensor_stack(focal_stack)
        device = self._execution_device
        dtype = self.transformer.dtype

        focal_stack = focal_stack.to(device=device, dtype=dtype)
        if focus_distances.dim() == 1:
            focus_distances = focus_distances.unsqueeze(0)
        focus_distances = focus_distances.to(device=device, dtype=dtype)

        if camera_params is not None:
            camera_params = {
                key: value.to(device=device, dtype=dtype)
                if isinstance(value, torch.Tensor)
                else torch.full((focal_stack.shape[0],), float(value), device=device, dtype=dtype)
                for key, value in camera_params.items()
            }
            camera_features = self.camera_encoder(
                camera_params,
                mode=camera_invariant_mode,
                focus_distances=focus_distances,
            )
        else:
            camera_features = None

        focal_features = self.focal_processor(
            focal_stack,
            focus_distances,
            camera_params=camera_params,
        )
        if camera_features is not None:
            focal_features["camera_features"] = camera_features

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
            device=device,
        )

        batch_size = focal_stack.shape[0] * num_images_per_prompt
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.transformer.config.in_channels,
            height=height // self.vae_scale_factor,
            width=width // self.vae_scale_factor,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for timestep in timesteps:
            latent_model_input = (
                torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=(
                    torch.cat([negative_prompt_embeds, prompt_embeds])
                    if guidance_scale > 1.0
                    else prompt_embeds
                ),
                pooled_projections=(
                    torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
                    if guidance_scale > 1.0
                    else pooled_prompt_embeds
                ),
                focal_features=focal_features,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        depth_logits, rgb_latents = self.dual_decoder(latents)
        depth_probs = torch.sigmoid(depth_logits)
        depth_probs = F.interpolate(
            depth_probs, size=(height, width), mode="bilinear", align_corners=False
        )
        depth_map = depth_probs.squeeze(1)

        if camera_params is not None and "depth_min" in camera_params and "depth_max" in camera_params:
            depth_min = camera_params["depth_min"].to(depth_map.dtype).view(-1, 1, 1)
            depth_max = camera_params["depth_max"].to(depth_map.dtype).view(-1, 1, 1)
            depth_map = depth_min + depth_map * (depth_max - depth_min)

        recon = self.vae.decode(rgb_latents / self.vae.config.scaling_factor, return_dict=False)[0]
        recon = (recon / 2 + 0.5).clamp(0, 1)

        if output_type == "pil":
            image = self.numpy_to_pil(recon.cpu().permute(0, 2, 3, 1).numpy())[0]
            depth_color = self._colorize_depth(depth_map[0].detach().cpu().numpy())
            result: Union[torch.Tensor, Image.Image] = image
        else:
            depth_color = None
            result = recon

        if not return_dict:
            return depth_map, result

        return FocalDiffusionOutput(
            depth_map=depth_map,
            all_in_focus_image=result,
            depth_colored=depth_color,
            focal_features=focal_features,
            attention_maps=None,
        )

    def _ensure_tensor_stack(self, stack: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(stack, torch.Tensor):
            if stack.dim() == 4:  # [N, C, H, W]
                stack = stack.unsqueeze(0)
            return stack

        tensors = []
        import numpy as np  # local import to avoid hard dependency at module import time

        for image in stack:
            rgb = image.convert("RGB")
            array = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
            array = array * 2.0 - 1.0
            tensors.append(array)

        return torch.stack(tensors, dim=0).unsqueeze(0)

    def _colorize_depth(self, depth: np.ndarray) -> Image.Image:
        import numpy as np

        normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        try:
            import matplotlib.pyplot as plt  # type: ignore

            colormap = plt.get_cmap("magma")
            colored = colormap(normalized)[..., :3]
            colored = (colored * 255).astype(np.uint8)
            return Image.fromarray(colored)
        except ModuleNotFoundError:
            gray = (normalized * 255).astype(np.uint8)
            return Image.fromarray(gray)
