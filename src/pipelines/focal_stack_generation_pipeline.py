"""Inference pipeline that augments Stable Diffusion 3.5 with focal-stack conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline

try:
    from diffusers.models.transformers import SD3Transformer2DModel
except ImportError as err:  # pragma: no cover - import-time environment check
    raise ImportError(
        "FocalStackGeneration requires diffusers>=0.28.0 so the Stable Diffusion 3 transformer is available. "
        "Upgrade via `pip install --upgrade diffusers`."
    ) from err

try:  # diffusers>=0.35 moved Transformer2DModelOutput to modeling_outputs
    from diffusers.models.transformers import Transformer2DModelOutput  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - compatibility shim for newer wheels
    try:
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
    except ImportError as err:
        raise ImportError(
            "FocalStackGeneration requires diffusers to expose Transformer2DModelOutput. "
            "Upgrade via `pip install --upgrade diffusers`."
        ) from err
from diffusers.utils import BaseOutput
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from ..models.focal_attention import FocalCrossAttention
from ..models.task_output_decoder import TaskOutputDecoder
from ..models.focal_processor import FocalStackProcessor
from ..models.focal_evidence_encoder import (
    FocalEvidenceEncoder,
    PhysicalEvidenceEstimator,
    build_physical_evidence_features,
    decode_metric_depth_from_focal_posterior,
)

logger = logging.getLogger(__name__)


def _module_device_dtype(module: nn.Module, fallback_device: torch.device, fallback_dtype: torch.dtype) -> Tuple[torch.device, torch.dtype]:
    """Return a module's parameter/buffer device and dtype with a safe fallback."""
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    if tensor is None:
        return fallback_device, fallback_dtype
    return tensor.device, tensor.dtype


@dataclass
class FocalStackGenerationOutput(BaseOutput):
    """Output type returned by :class:`FocalStackGenerationPipeline`."""

    depth_map: torch.Tensor
    all_in_focus_image: Union[torch.Tensor, Image.Image]
    depth_colored: Optional[Image.Image] = None
    uncertainty: Optional[torch.Tensor] = None
    generated_depth_canonical: Optional[torch.Tensor] = None
    focal_depth_canonical: Optional[torch.Tensor] = None
    final_depth_canonical: Optional[torch.Tensor] = None
    focal_posterior: Optional[torch.Tensor] = None
    focal_entropy: Optional[torch.Tensor] = None
    focal_peak_confidence: Optional[torch.Tensor] = None
    physical_evidence_support: Optional[torch.Tensor] = None
    focal_evidence_weight: Optional[torch.Tensor] = None
    generative_prior_weight: Optional[torch.Tensor] = None
    abstention_weight: Optional[torch.Tensor] = None
    posterior_margin: Optional[torch.Tensor] = None
    depth_disagreement: Optional[torch.Tensor] = None
    generative_uncertainty: Optional[torch.Tensor] = None
    uncertainty_focus: Optional[torch.Tensor] = None
    uncertainty_disagreement: Optional[torch.Tensor] = None
    uncertainty_final: Optional[torch.Tensor] = None
    depth_focus_metric: Optional[torch.Tensor] = None


class FocalInjectedSD3Transformer(nn.Module):
    """Wrapper around the SD3.5 transformer that accepts focal features."""

    def __init__(self, base_transformer: SD3Transformer2DModel, condition_channels: int = 512) -> None:
        super().__init__()
        self.base_transformer = base_transformer
        self.condition_channels = condition_channels
        base = self.base_transformer
        self.config = base.config
        hidden_size = self.config.attention_head_dim * self.config.num_attention_heads
        self.focal_attn = FocalCrossAttention(
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.attention_head_dim,
        )
        self.pre_focal_attn = FocalCrossAttention(
            hidden_size=hidden_size,
            num_heads=self.config.num_attention_heads,
            head_dim=self.config.attention_head_dim,
        )

        self.pre_norm = nn.LayerNorm(hidden_size)
        self.post_norm = nn.LayerNorm(hidden_size)
        self.condition_scale = nn.Parameter(torch.tensor(0.5))
        self.pre_scale = nn.Parameter(torch.tensor(0.5))
        self.post_scale = nn.Parameter(torch.tensor(1.0))

        self.condition_adapter = nn.Conv2d(condition_channels, hidden_size, kernel_size=1)
        self._condition_adapters = nn.ModuleDict()
        self.dtype = getattr(base, "dtype", torch.float32)
        self.condition_adapter.to(device=self._base_device, dtype=self.dtype)

        base_param = next(base.parameters(), None)
        if base_param is not None:
            self.to(device=base_param.device, dtype=base_param.dtype)

    @property
    def base_transformer(self) -> SD3Transformer2DModel:
        """Return the wrapped SD3 transformer, restoring it if Accelerate detached it."""

        base = self._modules.get("base_transformer")  # type: ignore[attr-defined]
        if base is not None:
            return cast(SD3Transformer2DModel, base)

        base = getattr(self, "_base_transformer_ref", None)
        if base is None:
            raise AttributeError(
                "FocalInjectedSD3Transformer is missing its base transformer. "
                "Re-wrap the SD3 transformer by calling `attach_base_transformer`."
            )

        self.attach_base_transformer(base)
        return cast(SD3Transformer2DModel, self._modules["base_transformer"])  # type: ignore[attr-defined]

    @base_transformer.setter
    def base_transformer(self, module: SD3Transformer2DModel) -> None:
        if not isinstance(module, SD3Transformer2DModel):
            raise TypeError(
                "FocalInjectedSD3Transformer expects an SD3Transformer2DModel, "
                f"but received {type(module)!r}."
            )

        super().__setattr__("base_transformer", module)
        super().__setattr__("_base_transformer_ref", module)
        super().__setattr__("config", module.config)
        super().__setattr__("dtype", getattr(module, "dtype", torch.float32))

    def attach_base_transformer(self, module: SD3Transformer2DModel) -> None:
        """Explicitly (re)attach the wrapped SD3 transformer."""

        self.base_transformer = module

    @property
    def _base_device(self) -> torch.device:
        try:
            return next(self.base_transformer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def __getattr__(self, name: str) -> Any:
        # Look up this wrapper's registered parameters/modules first, then delegate
        # attribute access (e.g. device, config helpers, etc.) to the wrapped module.
        try:
            return super().__getattr__(name)
        except AttributeError as err:
            try:
                base = self.base_transformer
            except AttributeError:
                raise err
            return getattr(base, name)

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
        base = self.base_transformer
        condition_pre = self._extract_condition(focal_features, hidden_states.shape[-2:], hidden_states.shape[0])
        if condition_pre is not None:
            hidden_states = self._apply_condition(hidden_states, condition_pre, self.pre_focal_attn, self.pre_norm, self.pre_scale)

        result = base.forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        condition_post = self._extract_condition(focal_features, result.shape[-2:], result.shape[0])
        if condition_post is not None:
            result = self._apply_condition(result, condition_post, self.focal_attn, self.post_norm, self.post_scale)

        if return_dict:
            return Transformer2DModelOutput(sample=result)

        return (result,)

    def _get_condition_adapter(self, channels: int) -> nn.Conv2d:
        key = str(channels)
        if key not in self._condition_adapters:
            adapter = nn.Conv2d(channels, self.config.attention_head_dim * self.config.num_attention_heads, kernel_size=1)
            self._condition_adapters[key] = adapter
        return self._condition_adapters[key]

    @staticmethod
    def _match_condition_batch(condition: torch.Tensor, target_batch: int) -> torch.Tensor:
        if condition.shape[0] == target_batch:
            return condition
        if target_batch % condition.shape[0] != 0:
            raise ValueError(
                f"Cannot broadcast focal condition batch {condition.shape[0]} to transformer batch {target_batch}."
            )
        repeat_factor = target_batch // condition.shape[0]
        return condition.repeat(repeat_factor, *([1] * (condition.dim() - 1)))

    def _extract_condition(
        self,
        focal_features: Optional[Dict[str, torch.Tensor]],
        spatial_size: Tuple[int, int],
        target_batch: int,
    ) -> Optional[torch.Tensor]:
        if not focal_features or "fused_features" not in focal_features:
            return None

        fused = focal_features["fused_features"]
        if fused.dim() == 5:
            fused = fused.mean(dim=1)

        if fused.shape[-2:] != spatial_size:
            fused = F.interpolate(fused, size=spatial_size, mode="bilinear", align_corners=False)

        if fused.shape[1] == self.condition_channels:
            conditioned = self.condition_adapter.to(device=fused.device, dtype=fused.dtype)(fused)
        else:
            adapter = self._get_condition_adapter(fused.shape[1]).to(device=fused.device, dtype=fused.dtype)
            conditioned = adapter(fused)

        spatial_maps = focal_features.get("temporal_attention_maps")
        if spatial_maps is not None:
            spatial = spatial_maps.mean(dim=1, keepdim=True)
            spatial = F.interpolate(spatial, size=spatial_size, mode="bilinear", align_corners=False)
            spatial = spatial.to(device=conditioned.device, dtype=conditioned.dtype)
            conditioned = conditioned * (1 + self.condition_scale * spatial)

        return self._match_condition_batch(conditioned, target_batch)

    def _apply_condition(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        attn: FocalCrossAttention,
        norm: nn.LayerNorm,
        scale: nn.Parameter
    ) -> torch.Tensor:
        hidden_seq = hidden_states.flatten(2).transpose(1, 2)
        cond_seq = condition.flatten(2).transpose(1, 2)
        hidden_seq = hidden_seq + scale * attn(norm(hidden_seq), cond_seq)
        return hidden_seq.transpose(1, 2).reshape_as(hidden_states)



class FocalStackGenerationPipeline(StableDiffusion3Pipeline):
    """Stable Diffusion 3.5 pipeline that consumes focal stacks instead of text-only prompts."""

    _MODULE_ATTRS = (
        "vae",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "transformer",
        "focal_processor",
        "focal_evidence_head",
        "task_output_decoder",
        "physical_evidence_support_head",
    )

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
        focal_evidence_head: Optional[nn.Module] = None,
        task_output_decoder: Optional[nn.Module] = None,
        physical_evidence_support_head: Optional[nn.Module] = None,
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

        for orphan in ("feature_extractor", "image_encoder"):
            if hasattr(self.config, orphan):
                delattr(self.config, orphan)
            internal = getattr(self.config, "_internal_dict", None)
            if hasattr(internal, "pop"):
                internal.pop(orphan, None)

        base_optional = tuple(
            name
            for name in getattr(self, "_optional_components", ())
            if name not in {"feature_extractor", "image_encoder"}
        )
        self._optional_components = base_optional + (
            "focal_processor",
            "focal_evidence_head",
            "task_output_decoder",
            "physical_evidence_support_head",
        )

        # Ensure the extended pipeline configuration tracks focal-specific components so
        # serialization remains compatible with diffusers' component bookkeeping.
        self.register_to_config(
            focal_processor=None,
            focal_evidence_head=None,
            task_output_decoder=None,
            physical_evidence_support_head=None,
        )

        self.focal_processor = focal_processor or FocalStackProcessor()
        self.focal_evidence_head = focal_evidence_head or FocalEvidenceEncoder()
        self.task_output_decoder = task_output_decoder or TaskOutputDecoder(
            in_channels=self.vae.config.latent_channels,
            out_channels_depth=1,
            out_channels_rgb=self.vae.config.latent_channels,
        )
        self.physical_evidence_support_head = physical_evidence_support_head or PhysicalEvidenceEstimator(
            in_channels=5,
            hidden=16,
        )

        condition_channels = getattr(self.focal_processor, "feature_dim", None)
        if condition_channels is None:
            raise ValueError(
                "focal_processor must expose a feature_dim attribute so the focal "
                "condition adapter can be created before training starts."
            )
        condition_channels = int(condition_channels)

        if not isinstance(self.transformer, FocalInjectedSD3Transformer):
            self.transformer = FocalInjectedSD3Transformer(
                self.transformer,
                condition_channels=condition_channels,
            )
        else:
            _ = self.transformer.base_transformer
            if self.transformer.condition_channels != condition_channels:
                raise ValueError(
                    "Existing FocalInjectedSD3Transformer condition_channels "
                    f"({self.transformer.condition_channels}) does not match focal_processor.feature_dim "
                    f"({condition_channels})."
                )

        # Ensure diffusers tracks the newly attached modules so that calls to
        # `pipeline.to(device)` migrate them alongside the base SD components.
        # Without registering them, the focal stack processor (and friends)
        # would remain on the CPU, which later causes device mismatches once
        # the training batches are moved to the accelerator.
        self.register_modules(
            transformer=self.transformer,
            focal_processor=self.focal_processor,
            focal_evidence_head=self.focal_evidence_head,
            task_output_decoder=self.task_output_decoder,
            physical_evidence_support_head=self.physical_evidence_support_head,
        )

    @property
    def components(self):  # type: ignore[override]
        """Return the active pipeline components without SD3's unused optional slots."""

        components = {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "transformer": self.transformer,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "focal_processor": self.focal_processor,
            "focal_evidence_head": self.focal_evidence_head,
            "task_output_decoder": self.task_output_decoder,
            "physical_evidence_support_head": self.physical_evidence_support_head,
        }

        if getattr(self, "text_encoder_3", None) is not None:
            components["text_encoder_3"] = self.text_encoder_3
        if getattr(self, "tokenizer_3", None) is not None:
            components["tokenizer_3"] = self.tokenizer_3

        return components

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Move every registered module to the requested device/dtype."""

        super().to(*args, **kwargs)

        for _, module in self._iter_registered_modules():
            if isinstance(module, nn.Module):
                module.to(*args, **kwargs)

        return self

    def _iter_registered_modules(self):
        for attr in self._MODULE_ATTRS:
            module = getattr(self, attr, None)
            if module is None:
                continue
            # Some components (e.g. schedulers) are not nn.Module subclasses.
            if isinstance(module, nn.Module):
                yield attr, module

    def parameters(self, recurse: bool = True):
        """Yield parameters from every learnable submodule."""

        for _, module in self._iter_registered_modules():
            yield from module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Yield named parameters, namespaced by component."""

        for attr, module in self._iter_registered_modules():
            component_prefix = f"{prefix}{attr}." if prefix else f"{attr}."
            for name, param in module.named_parameters(prefix="", recurse=recurse):
                yield component_prefix + name, param

    def train(self, mode: bool = True):
        """Set all registered modules to training or evaluation mode."""

        for _, module in self._iter_registered_modules():
            module.train(mode)
        return self

    def eval(self):
        """Switch all registered modules to evaluation mode."""

        return self.train(False)


    @staticmethod
    def _expand_focal_features_for_model(
        focal_features: Dict[str, Any],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> Dict[str, Any]:
        """Repeat focal conditioning to match the model input batch.

        SD3 classifier-free guidance doubles the latent and text-conditioning
        batches by concatenating unconditional and conditional inputs. Focal
        conditioning is batch-first as well, so it must be expanded in lockstep
        before being passed to the injected transformer.
        """

        repeat_count = max(1, num_images_per_prompt)

        def expand(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                expanded = value
                if repeat_count > 1:
                    expanded = expanded.repeat_interleave(repeat_count, dim=0)
                if do_classifier_free_guidance:
                    expanded = torch.cat([expanded, expanded], dim=0)
                return expanded
            if isinstance(value, dict):
                return {key: expand(item) for key, item in value.items()}
            if isinstance(value, list):
                return [expand(item) for item in value]
            if isinstance(value, tuple):
                return tuple(expand(item) for item in value)
            return value

        return {key: expand(value) for key, value in focal_features.items()}

    @torch.no_grad()
    def __call__(
        self,
        focal_stack: Union[torch.Tensor, List[Image.Image]],
        focal_plane_distances: Optional[torch.Tensor] = None,
        prompt: str = "",
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        focal_distance_mode: str = "normalized",
        **kwargs: Any,
    ) -> Union[FocalStackGenerationOutput, Tuple[torch.Tensor, Union[torch.Tensor, Image.Image]]]:
        """Generate canonical depth and all-in-focus outputs from a focal stack.

        Args:
            focal_stack: Input focal stack as a tensor ``[B, N, C, H, W]`` /
                ``[N, C, H, W]`` or a list of RGB PIL images.
            focal_plane_distances: Per-plane focal coordinates with shape
                ``[N]`` or ``[B, N]``. These are always used to form canonical
                focal coordinates for the model.
            focal_distance_mode: Set to ``"normalized"`` when focal-plane
                values are relative/index-like coordinates; ``depth_focus_metric``
                will be ``None``. Set to ``"metric"`` only when
                ``focal_plane_distances`` are calibrated metric distances; then
                metric focus depth is decoded from ``focal_posterior`` and
                returned as ``depth_focus_metric``.

        Raises:
            ValueError: If ``focal_distance_mode`` is not ``"normalized"`` or
                ``"metric"``.
        """
        if focal_distance_mode not in {"normalized", "metric"}:
            raise ValueError(
                "focal_distance_mode must be either 'normalized' or 'metric', "
                f"got {focal_distance_mode!r}."
            )

        kwargs.clear()
        if focal_plane_distances is None:
            raise ValueError("focal_plane_distances is required.")

        focal_stack = self._ensure_tensor_stack(focal_stack)
        device = self._execution_device
        dtype = self.transformer.dtype

        focal_stack = focal_stack.to(device=device)
        input_h, input_w = focal_stack.shape[-2:]
        target_h = input_h if height is None else int(height)
        target_w = input_w if width is None else int(width)
        height, width = self._make_divisible_size(target_h, target_w, divisor=max(int(self.vae_scale_factor), 16))
        if not isinstance(focal_plane_distances, torch.Tensor):
            focal_plane_distances = torch.as_tensor(focal_plane_distances, dtype=torch.float32)
        if focal_plane_distances.dim() == 1:
            focal_plane_distances = focal_plane_distances.unsqueeze(0)
        if focal_plane_distances.dim() != 2:
            raise ValueError(
                "focal_plane_distances must have shape [B, N] or [N], "
                f"got {tuple(focal_plane_distances.shape)}"
            )
        if focal_plane_distances.shape[0] == 1 and focal_stack.shape[0] != 1:
            focal_plane_distances = focal_plane_distances.expand(focal_stack.shape[0], -1)
        if focal_plane_distances.shape != focal_stack.shape[:2]:
            raise ValueError(
                "focal_plane_distances must match focal_stack batch and focal-plane dimensions: "
                f"expected {tuple(focal_stack.shape[:2])}, got {tuple(focal_plane_distances.shape)}"
            )
        if not torch.isfinite(focal_plane_distances).all():
            raise ValueError("focal_plane_distances must contain only finite values")
        focal_plane_distances = focal_plane_distances.to(device=device)

        evidence_device, evidence_dtype = _module_device_dtype(self.focal_evidence_head, torch.device(device), dtype)
        processor_device, processor_dtype = _module_device_dtype(self.focal_processor, torch.device(device), dtype)
        focal_evidence = self.focal_evidence_head(
            focal_stack.to(device=evidence_device, dtype=evidence_dtype),
            focal_plane_distances.to(device=evidence_device, dtype=evidence_dtype),
        )
        focal_features = self.focal_processor(
            focal_stack.to(device=processor_device, dtype=processor_dtype),
            focal_plane_distances.to(device=processor_device, dtype=processor_dtype),
        )
        focal_features = {
            key: value.to(device=device, dtype=dtype) if isinstance(value, torch.Tensor) else value
            for key, value in focal_features.items()
        }

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

        model_focal_features = self._expand_focal_features_for_model(
            focal_features,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=guidance_scale > 1.0,
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
                focal_features=model_focal_features,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        decoder_device, decoder_dtype = _module_device_dtype(self.task_output_decoder, torch.device(device), dtype)
        decoder_outputs = self.task_output_decoder(latents.to(device=decoder_device, dtype=decoder_dtype))
        generated_depth_canonical = decoder_outputs["generated_depth_canonical"]
        generative_uncertainty = decoder_outputs["uncertainty"]
        all_in_focus_latents = decoder_outputs["all_in_focus_latents"]

        generated_depth_canonical = F.interpolate(generated_depth_canonical, size=(height, width), mode="bilinear", align_corners=False)
        generative_uncertainty = F.interpolate(generative_uncertainty, size=(height, width), mode="bilinear", align_corners=False)
        focal_depth_canonical = F.interpolate(focal_evidence["focal_depth_canonical"], size=(height, width), mode="bilinear", align_corners=False)
        focal_entropy = F.interpolate(focal_evidence["focal_entropy"], size=(height, width), mode="bilinear", align_corners=False)
        focal_posterior = F.interpolate(focal_evidence["focal_posterior"], size=(height, width), mode="bilinear", align_corners=False)
        focal_posterior = focal_posterior / focal_posterior.sum(dim=1, keepdim=True).clamp(min=1e-6)
        if focal_depth_canonical.shape[0] != generated_depth_canonical.shape[0]:
            repeat_factor = generated_depth_canonical.shape[0] // focal_depth_canonical.shape[0]
            focal_depth_canonical = focal_depth_canonical.repeat_interleave(repeat_factor, dim=0)
            focal_entropy = focal_entropy.repeat_interleave(repeat_factor, dim=0)
            focal_posterior = focal_posterior.repeat_interleave(repeat_factor, dim=0)

        support_inputs, support_maps = build_physical_evidence_features(
            focal_posterior=focal_posterior,
            focal_entropy=focal_entropy,
            focal_depth_canonical=focal_depth_canonical,
            generated_depth_canonical=generated_depth_canonical,
            generative_uncertainty=generative_uncertainty,
        )
        support_device, support_dtype = _module_device_dtype(self.physical_evidence_support_head, torch.device(device), dtype)
        support_inputs = support_inputs.to(device=support_device, dtype=support_dtype)
        support_outputs = self.physical_evidence_support_head(support_inputs)
        focal_evidence_weight = support_outputs["focal_evidence_weight"]
        generative_prior_weight = support_outputs["generative_prior_weight"]
        abstention_weight = support_outputs["abstention_weight"]
        gate_sum = (focal_evidence_weight + generative_prior_weight).clamp(min=1e-6)
        focal_evidence_weight_norm = focal_evidence_weight / gate_sum
        generative_prior_weight_norm = generative_prior_weight / gate_sum
        final_depth_canonical = focal_evidence_weight_norm * focal_depth_canonical + generative_prior_weight_norm * generated_depth_canonical
        depth_map = final_depth_canonical.squeeze(1)
        uncertainty_final = torch.maximum(
            support_outputs["uncertainty_final"],
            abstention_weight,
        ).clamp(0.0, 1.0)
        physical_evidence_support = support_outputs["physical_evidence_support"]
        uncertainty_focus = focal_entropy
        uncertainty_disagreement = support_maps["depth_disagreement"]
        depth_focus_metric = None
        if focal_distance_mode == "metric":
            metric_focal_plane_distances = focal_plane_distances
            if metric_focal_plane_distances.shape[0] == 1 and focal_posterior.shape[0] != 1:
                metric_focal_plane_distances = metric_focal_plane_distances.expand(
                    focal_posterior.shape[0],
                    -1,
                )
            depth_focus_metric = decode_metric_depth_from_focal_posterior(
                focal_posterior=focal_posterior,
                focal_plane_distances=metric_focal_plane_distances,
            ).squeeze(1)

        recon = self.vae.decode(all_in_focus_latents / self.vae.config.scaling_factor, return_dict=False)[0]
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

        return FocalStackGenerationOutput(
            depth_map=depth_map,
            all_in_focus_image=result,
            depth_colored=depth_color,
            uncertainty=uncertainty_final.squeeze(1),
            generated_depth_canonical=generated_depth_canonical.squeeze(1),
            focal_depth_canonical=focal_depth_canonical.squeeze(1),
            final_depth_canonical=final_depth_canonical.squeeze(1),
            focal_posterior=focal_posterior,
            focal_entropy=focal_entropy.squeeze(1),
            focal_peak_confidence=support_maps["focal_peak_confidence"].squeeze(1),
            posterior_margin=support_maps["posterior_margin"].squeeze(1),
            depth_disagreement=support_maps["depth_disagreement"].squeeze(1),
            physical_evidence_support=physical_evidence_support.squeeze(1),
            focal_evidence_weight=focal_evidence_weight_norm.squeeze(1),
            generative_prior_weight=generative_prior_weight_norm.squeeze(1),
            abstention_weight=abstention_weight.squeeze(1),
            generative_uncertainty=generative_uncertainty.squeeze(1),
            uncertainty_focus=uncertainty_focus.squeeze(1),
            uncertainty_disagreement=uncertainty_disagreement.squeeze(1),
            uncertainty_final=uncertainty_final.squeeze(1),
            depth_focus_metric=depth_focus_metric,
        )

    @staticmethod
    def _make_divisible_size(height: int, width: int, divisor: int = 16) -> Tuple[int, int]:
        """Round spatial size up to a multiple of ``divisor`` without forcing square outputs."""
        if divisor <= 0:
            raise ValueError("divisor must be positive")
        height = max(1, int(height))
        width = max(1, int(width))
        out_h = int(math.ceil(height / divisor) * divisor)
        out_w = int(math.ceil(width / divisor) * divisor)
        return out_h, out_w


    def _ensure_tensor_stack(self, stack: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        if isinstance(stack, torch.Tensor):
            if stack.dim() == 4:
                stack = stack.unsqueeze(0)
            if stack.dim() != 5:
                raise ValueError(
                    "focal_stack tensor must have shape [B, N, C, H, W] or [N, C, H, W], "
                    f"got {tuple(stack.shape)}"
                )
            if stack.shape[1] < 1:
                raise ValueError("focal_stack tensor must contain at least one focal plane")
            if stack.shape[2] != 3:
                raise ValueError(f"focal_stack tensor must contain RGB images with 3 channels, got {stack.shape[2]}")
            if not torch.isfinite(stack).all():
                raise ValueError("focal_stack tensor must contain only finite values")
            if stack.min() >= 0 and stack.max() <= 1:
                stack = stack * 2.0 - 1.0
            return stack

        if not stack:
            raise ValueError("focal_stack image list must not be empty")

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


# Backward-compatible aliases for external scripts using pre-rename APIs.
FocalDiffusionOutput = FocalStackGenerationOutput
FocalDiffusionPipeline = FocalStackGenerationPipeline
