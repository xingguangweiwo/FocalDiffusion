"""Inference pipeline that augments Stable Diffusion 3.5 with focal-stack conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import logging
import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from ..models.physics_modules import FocalPhysicalVerifier
from ..utils.image_utils import resize_probability_volume, to_model_range, to_unit_range
from ..models.verification_trace import PhysicalVerificationTrace
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


@dataclass(init=False)
class FocalStackGenerationOutput(BaseOutput):
    """Tensor-only output returned by :class:`FocalStackGenerationPipeline`.

    Primary fields mirror the active method: fused depth, AIF reconstruction,
    uncertainty, diagnostics, and an optional refinement summary. Legacy keyword
    aliases are accepted temporarily for checkpoint/output migration.
    """

    depth: torch.Tensor
    all_in_focus: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    diagnostics: Dict[str, Any] = None  # type: ignore[assignment]
    refinement_summary: Dict[str, Any] = None  # type: ignore[assignment]

    def __init__(
        self,
        depth: Optional[torch.Tensor] = None,
        all_in_focus: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        refinement_summary: Optional[Dict[str, Any]] = None,
        **legacy: Any,
    ) -> None:
        if depth is None:
            depth = legacy.pop("depth_map", None)
        if all_in_focus is None:
            all_in_focus = legacy.pop("all_in_focus_image", None)
        legacy_uncertainty_final = legacy.get("uncertainty_final")
        if uncertainty is None:
            uncertainty = legacy.pop("uncertainty_final", legacy.pop("uncertainty", None))
        if depth is None or all_in_focus is None:
            raise TypeError("FocalStackGenerationOutput requires depth and all_in_focus tensors.")
        diagnostics = dict(diagnostics or {})
        refinement_summary = dict(refinement_summary or {})
        legacy.pop("depth_colored", None)
        if legacy_uncertainty_final is not None:
            diagnostics.setdefault("uncertainty_final", legacy_uncertainty_final)
        for key, value in legacy.items():
            if key in {"refinement_history", "accepted_refinement_steps", "rejected_refinement_steps", "physical_risk_before", "physical_risk_after"}:
                refinement_summary[key] = value
            else:
                diagnostics[key] = value
        self.depth = depth
        self.all_in_focus = all_in_focus
        self.uncertainty = uncertainty
        self.diagnostics = diagnostics
        self.refinement_summary = refinement_summary

    @property
    def depth_map(self) -> torch.Tensor:
        return self.depth

    @depth_map.setter
    def depth_map(self, value: torch.Tensor) -> None:
        self.depth = value

    @property
    def all_in_focus_image(self) -> torch.Tensor:
        return self.all_in_focus

    @all_in_focus_image.setter
    def all_in_focus_image(self, value: torch.Tensor) -> None:
        self.all_in_focus = value

    @property
    def depth_colored(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        if name in self.diagnostics:
            return self.diagnostics[name]
        if name in self.refinement_summary:
            return self.refinement_summary[name]
        legacy_optional = {
            "generated_depth_canonical",
            "focal_depth_canonical",
            "final_depth_canonical",
            "focal_posterior",
            "focal_entropy",
            "focal_peak_confidence",
            "physical_evidence_support",
            "focal_evidence_weight",
            "generative_prior_weight",
            "abstention_weight",
            "posterior_margin",
            "depth_disagreement",
            "generative_uncertainty",
            "uncertainty_focus",
            "uncertainty_disagreement",
            "uncertainty_final",
            "depth_focus_metric",
            "physical_verification_trace",
            "refinement_history",
        }
        if name in legacy_optional:
            return None
        raise AttributeError(name)


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

        if fused.shape[1] != self.condition_channels:
            raise ValueError(
                "fused focal feature channels must match the initialized condition adapter; "
                f"expected {self.condition_channels}, got {fused.shape[1]}"
            )
        conditioned = self.condition_adapter.to(device=fused.device, dtype=fused.dtype)(fused)

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
        "physical_verifier",
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
        physical_verifier: Optional[nn.Module] = None,
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
            "physical_verifier",
        )

        # Ensure the extended pipeline configuration tracks focal-specific components so
        # serialization remains compatible with diffusers' component bookkeeping.
        self.register_to_config(
            focal_processor=None,
            focal_evidence_head=None,
            task_output_decoder=None,
            physical_evidence_support_head=None,
            physical_verifier=None,
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
        self.physical_verifier = physical_verifier or FocalPhysicalVerifier()

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
            physical_verifier=self.physical_verifier,
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
            "physical_verifier": self.physical_verifier,
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


    def _get_empty_conditioning(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached empty text conditioning used by the focal-stack method."""
        cache = getattr(self, "_empty_conditioning_cache", None)
        if cache is not None and cache["prompt_embeds"].device == device and cache["prompt_embeds"].dtype == dtype:
            return cache["prompt_embeds"], cache["pooled_prompt_embeds"]
        prompt_embeds, _, pooled_prompt_embeds, _ = self.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype).detach()
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype).detach()
        self._empty_conditioning_cache = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
        return prompt_embeds, pooled_prompt_embeds

    def offload_text_encoders(self) -> None:
        """Move text encoders to CPU after empty conditioning has been cached."""
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.to("cpu")

    def _prepare_focal_inputs(
        self,
        focal_stack: torch.Tensor,
        focal_plane_distances: torch.Tensor,
        height: Optional[int],
        width: Optional[int],
    ) -> Dict[str, Any]:
        """Validate focal tensors, normalize image ranges, and choose output size."""
        if not isinstance(focal_stack, torch.Tensor):
            raise TypeError("FocalStackGenerationPipeline expects focal_stack as a tensor; load PIL images in script/inference.py.")
        focal_stack = self._ensure_tensor_stack(focal_stack)
        device = self._execution_device
        dtype = self.transformer.dtype
        focal_stack = focal_stack.to(device=device)
        focal_stack_unit = to_unit_range(focal_stack.float())
        focal_stack_signed = to_model_range(focal_stack.float())
        input_h, input_w = focal_stack.shape[-2:]
        target_h = input_h if height is None else int(height)
        target_w = input_w if width is None else int(width)
        height, width = self._make_divisible_size(target_h, target_w, divisor=max(int(self.vae_scale_factor), 16))
        if not isinstance(focal_plane_distances, torch.Tensor):
            focal_plane_distances = torch.as_tensor(focal_plane_distances, dtype=torch.float32)
        if focal_plane_distances.dim() == 1:
            focal_plane_distances = focal_plane_distances.unsqueeze(0)
        if focal_plane_distances.dim() != 2:
            raise ValueError(f"focal_plane_distances must have shape [B, N] or [N], got {tuple(focal_plane_distances.shape)}")
        if focal_plane_distances.shape[0] == 1 and focal_stack.shape[0] != 1:
            focal_plane_distances = focal_plane_distances.expand(focal_stack.shape[0], -1)
        if focal_plane_distances.shape != focal_stack.shape[:2]:
            raise ValueError(
                "focal_plane_distances must match focal_stack batch and focal-plane dimensions: "
                f"expected {tuple(focal_stack.shape[:2])}, got {tuple(focal_plane_distances.shape)}"
            )
        if not torch.isfinite(focal_plane_distances).all():
            raise ValueError("focal_plane_distances must contain only finite values")
        return {
            "device": device,
            "dtype": dtype,
            "height": height,
            "width": width,
            "focal_stack_unit": focal_stack_unit,
            "focal_stack_signed": focal_stack_signed,
            "focal_plane_distances": focal_plane_distances.to(device=device),
        }

    def _encode_focal_stack(self, prepared: Dict[str, Any]) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute focal-sweep features and focus-likelihood outputs."""
        device = prepared["device"]
        dtype = prepared["dtype"]
        stack = prepared["focal_stack_signed"]
        distances = prepared["focal_plane_distances"]
        evidence_device, evidence_dtype = _module_device_dtype(self.focal_evidence_head, torch.device(device), dtype)
        processor_device, processor_dtype = _module_device_dtype(self.focal_processor, torch.device(device), dtype)
        focal_evidence = self.focal_evidence_head(
            stack.to(device=evidence_device, dtype=evidence_dtype),
            distances.to(device=evidence_device, dtype=evidence_dtype),
        )
        focal_features = self.focal_processor(
            stack.to(device=processor_device, dtype=processor_dtype),
            distances.to(device=processor_device, dtype=processor_dtype),
        )
        focal_features = {
            key: value.to(device=device, dtype=dtype) if isinstance(value, torch.Tensor) else value
            for key, value in focal_features.items()
        }
        return focal_features, focal_evidence

    def _sample_clean_latent(
        self,
        prepared: Dict[str, Any],
        focal_features: Dict[str, torch.Tensor],
        num_inference_steps: int,
        generator: Optional[torch.Generator],
        latents: Optional[torch.FloatTensor],
    ) -> torch.Tensor:
        """Sample the SD3 latent with focal conditioning and cached empty text conditioning."""
        device = prepared["device"]
        dtype = prepared["dtype"]
        prompt_embeds, pooled_prompt_embeds = self._get_empty_conditioning(torch.device(device), dtype)
        batch_size = prepared["focal_stack_signed"].shape[0]
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.transformer.config.in_channels,
            height=prepared["height"] // self.vae_scale_factor,
            width=prepared["width"] // self.vae_scale_factor,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )
        if prompt_embeds.shape[0] == 1 and batch_size != 1:
            prompt_embeds = prompt_embeds.repeat((batch_size,) + (1,) * (prompt_embeds.ndim - 1))
            pooled_prompt_embeds = pooled_prompt_embeds.repeat((batch_size,) + (1,) * (pooled_prompt_embeds.ndim - 1))
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                focal_features=focal_features,
                return_dict=False,
            )[0]
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        return latents

    def _decode_prediction_heads(self, prepared: Dict[str, Any], latents: torch.Tensor, focal_evidence: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode generative heads and align focus-likelihood maps to output size."""
        device = prepared["device"]
        dtype = prepared["dtype"]
        height = prepared["height"]
        width = prepared["width"]
        decoder_device, decoder_dtype = _module_device_dtype(self.task_output_decoder, torch.device(device), dtype)
        decoder_outputs = self.task_output_decoder(latents.to(device=decoder_device, dtype=decoder_dtype))
        all_in_focus_latents = decoder_outputs["all_in_focus_latents"]
        recon = self.vae.decode(all_in_focus_latents / self.vae.config.scaling_factor, return_dict=False)[0]
        recon = (recon / 2 + 0.5).clamp(0, 1)
        return {
            "generated_depth_canonical": F.interpolate(decoder_outputs["generated_depth_canonical"], size=(height, width), mode="bilinear", align_corners=False),
            "generative_uncertainty": F.interpolate(decoder_outputs["uncertainty"], size=(height, width), mode="bilinear", align_corners=False),
            "focal_depth_canonical": F.interpolate(focal_evidence["focal_depth_canonical"], size=(height, width), mode="bilinear", align_corners=False),
            "focal_entropy": F.interpolate(focal_evidence["focal_entropy"], size=(height, width), mode="bilinear", align_corners=False),
            "focal_posterior": resize_probability_volume(focal_evidence["focal_posterior"], (height, width)),
            "all_in_focus": recon,
        }

    def _compute_reliability_fusion(self, prepared: Dict[str, Any], heads: Dict[str, torch.Tensor], focal_distance_mode: str) -> Dict[str, Any]:
        """Fuse focus depth with the generative prior through the reliability head."""
        device = prepared["device"]
        dtype = prepared["dtype"]
        support_inputs, support_maps = build_physical_evidence_features(
            focal_posterior=heads["focal_posterior"],
            focal_entropy=heads["focal_entropy"],
            focal_depth_canonical=heads["focal_depth_canonical"],
            generated_depth_canonical=heads["generated_depth_canonical"],
            generative_uncertainty=heads["generative_uncertainty"],
        )
        support_device, support_dtype = _module_device_dtype(self.physical_evidence_support_head, torch.device(device), dtype)
        support_outputs = self.physical_evidence_support_head(support_inputs.to(device=support_device, dtype=support_dtype))
        focal_weight = support_outputs["focal_evidence_weight"]
        prior_weight = support_outputs["generative_prior_weight"]
        abstention = support_outputs["abstention_weight"]
        gate_sum = (focal_weight + prior_weight).clamp(min=1e-6)
        focal_weight_norm = focal_weight / gate_sum
        prior_weight_norm = prior_weight / gate_sum
        final_depth = focal_weight_norm * heads["focal_depth_canonical"] + prior_weight_norm * heads["generated_depth_canonical"]
        uncertainty = torch.maximum(support_outputs["uncertainty_final"], abstention).clamp(0.0, 1.0)
        depth_focus_metric = None
        if focal_distance_mode == "metric":
            metric_distances = prepared["focal_plane_distances"]
            if metric_distances.shape[0] == 1 and heads["focal_posterior"].shape[0] != 1:
                metric_distances = metric_distances.expand(heads["focal_posterior"].shape[0], -1)
            depth_focus_metric = decode_metric_depth_from_focal_posterior(heads["focal_posterior"], metric_distances).squeeze(1)
        diagnostics = {
            **heads,
            **support_maps,
            "physical_evidence_support": support_outputs["physical_evidence_support"],
            "focal_evidence_weight": focal_weight_norm,
            "generative_prior_weight": prior_weight_norm,
            "abstention_weight": abstention,
            "uncertainty_focus": heads["focal_entropy"],
            "uncertainty_disagreement": support_maps["depth_disagreement"],
            "uncertainty_final": uncertainty,
            "depth_focus_metric": depth_focus_metric,
        }
        return {"depth": final_depth, "uncertainty": uncertainty, "diagnostics": diagnostics}

    def _compute_consistency_diagnostics(self, prepared: Dict[str, Any], fused: Dict[str, Any]) -> PhysicalVerificationTrace:
        """Evaluate focal-consistency diagnostics for the fused prediction."""
        verifier_device, verifier_dtype = _module_device_dtype(self.physical_verifier, torch.device(prepared["device"]), prepared["dtype"])
        diagnostics = fused["diagnostics"]
        return self.physical_verifier(
            focal_stack=prepared["focal_stack_unit"].to(device=verifier_device, dtype=verifier_dtype),
            focal_plane_distances=prepared["focal_plane_distances"].to(device=verifier_device, dtype=verifier_dtype),
            depth_canonical=fused["depth"].to(device=verifier_device, dtype=verifier_dtype),
            all_in_focus=diagnostics["all_in_focus"].to(device=verifier_device, dtype=verifier_dtype),
            generated_depth_canonical=diagnostics["generated_depth_canonical"].to(device=verifier_device, dtype=verifier_dtype),
        )

    def _run_test_time_optimization(
        self,
        prepared: Dict[str, Any],
        fused: Dict[str, Any],
        trace: PhysicalVerificationTrace,
        *,
        num_refinement_steps: int,
        trace_refinement_epsilon: float,
        refinement_mode: str,
        refinement_seed: int,
        return_refinement_history: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PhysicalVerificationTrace, Dict[str, Any]]:
        """Optionally run per-instance refinement with held-out consistency checks."""
        final_depth = fused["depth"]
        uncertainty = fused["uncertainty"]
        recon = fused["diagnostics"]["all_in_focus"]
        generated_depth = fused["diagnostics"]["generated_depth_canonical"]
        focus_depth = fused["diagnostics"]["focal_depth_canonical"]
        refinement_history: Optional[List[Dict[str, Any]]] = [] if (return_refinement_history and num_refinement_steps > 0) else None
        accepted_steps = 0
        rejected_steps = 0
        epsilon = float(trace_refinement_epsilon)
        initial_risk = self._physical_risk(trace)
        risk_before = initial_risk
        risk_after = initial_risk
        verifier_device, verifier_dtype = _module_device_dtype(self.physical_verifier, torch.device(prepared["device"]), prepared["dtype"])
        for refinement_step in range(num_refinement_steps):
            if refinement_mode == "selective_tto":
                candidate_depth, candidate_uncertainty, candidate_recon = self._selective_test_time_refinement(
                    focal_stack_unit=prepared["focal_stack_unit"],
                    focal_plane_distances=prepared["focal_plane_distances"],
                    final_depth_canonical=final_depth,
                    focus_depth_canonical=focus_depth,
                    prior_depth_canonical=generated_depth,
                    all_in_focus_unit=recon,
                    uncertainty_final=uncertainty,
                    trace=trace,
                    seed=refinement_seed + refinement_step,
                )
            else:
                candidate_depth, candidate_uncertainty = self._apply_trace_refinement(final_depth, focus_depth, generated_depth, uncertainty, trace)
                candidate_recon = recon
            candidate_trace = self.physical_verifier(
                focal_stack=prepared["focal_stack_unit"].to(device=verifier_device, dtype=verifier_dtype),
                focal_plane_distances=prepared["focal_plane_distances"].to(device=verifier_device, dtype=verifier_dtype),
                depth_canonical=candidate_depth.to(device=verifier_device, dtype=verifier_dtype),
                all_in_focus=candidate_recon.to(device=verifier_device, dtype=verifier_dtype),
                generated_depth_canonical=generated_depth.to(device=verifier_device, dtype=verifier_dtype),
            )
            accepted, current_risk, candidate_risk = self._accept_refinement_candidate(trace, candidate_trace, epsilon=epsilon)
            _, val_idx = self._split_refinement_planes(prepared["focal_stack_unit"].shape[1], prepared["focal_stack_unit"].device, refinement_seed + refinement_step)
            current_val = self._heldout_measurement_loss(prepared["focal_stack_unit"], recon, final_depth, prepared["focal_plane_distances"], val_idx)
            candidate_val = self._heldout_measurement_loss(prepared["focal_stack_unit"], candidate_recon, candidate_depth, prepared["focal_plane_distances"], val_idx)
            accepted = accepted and self._should_accept_refinement(current_val, candidate_val, epsilon)
            if accepted:
                final_depth = candidate_depth
                uncertainty = candidate_uncertainty
                recon = candidate_recon
                trace = candidate_trace
                accepted_steps += 1
                risk_after = candidate_risk
            else:
                rejected_steps += 1
                risk_after = current_risk
            if refinement_history is not None:
                refinement_history.append({
                    "step": refinement_step,
                    "accepted": accepted,
                    "physical_risk_before": current_risk,
                    "physical_risk_after": risk_after,
                    "final_depth_canonical": final_depth.detach().cpu(),
                    "uncertainty_final": uncertainty.detach().cpu(),
                    "mean_conflict_score": float(trace.conflict_score.detach().float().mean().item()),
                    "mean_invalid_score": float(trace.invalid_score.detach().float().mean().item()),
                    "mean_focus_support": float(trace.focus_support.detach().float().mean().item()),
                    "mean_generation_support": float(trace.generation_support.detach().float().mean().item()),
                })
            if not accepted:
                break
        return final_depth, uncertainty, recon, trace, {
            "refinement_history": refinement_history,
            "accepted_refinement_steps": accepted_steps,
            "rejected_refinement_steps": rejected_steps,
            "physical_risk_before": risk_before,
            "physical_risk_after": risk_after,
        }

    def _format_output(
        self,
        depth: torch.Tensor,
        all_in_focus: torch.Tensor,
        uncertainty: torch.Tensor,
        diagnostics: Dict[str, Any],
        refinement_summary: Dict[str, Any],
        return_dict: bool,
    ) -> Union[FocalStackGenerationOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """Build a tensor-only output object or tuple."""
        depth_map = depth.squeeze(1)
        diagnostics = dict(diagnostics)
        diagnostics["final_depth_canonical"] = depth_map
        diagnostics["physical_verification_trace"] = diagnostics.get("physical_verification_trace")
        for key, value in list(diagnostics.items()):
            if isinstance(value, torch.Tensor) and value.dim() == 4 and value.shape[1] == 1:
                diagnostics[key] = value.squeeze(1)
        uncertainty_map = uncertainty.squeeze(1)
        if not return_dict:
            return depth_map, all_in_focus
        return FocalStackGenerationOutput(
            depth=depth_map,
            all_in_focus=all_in_focus,
            uncertainty=uncertainty_map,
            diagnostics=diagnostics,
            refinement_summary=refinement_summary,
        )

    @torch.no_grad()
    def __call__(
        self,
        focal_stack: torch.Tensor,
        focal_plane_distances: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        focal_distance_mode: str = "normalized",
        num_refinement_steps: int = 0,
        trace_refinement_epsilon: float = 1e-4,
        refinement_mode: str = "trace_update",
        refinement_seed: int = 0,
        return_refinement_history: bool = False,
        **kwargs: Any,
    ) -> Union[FocalStackGenerationOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the active focal-stack method: encode, sample, decode, fuse, diagnose, optionally refine."""
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected FocalStackGenerationPipeline arguments: {unexpected}")
        if num_refinement_steps < 0:
            raise ValueError("num_refinement_steps must be non-negative.")
        if refinement_mode not in {"trace_update", "selective_tto"}:
            raise ValueError("refinement_mode must be 'trace_update' or 'selective_tto'.")
        if focal_distance_mode not in {"normalized", "metric"}:
            raise ValueError(f"focal_distance_mode must be either 'normalized' or 'metric', got {focal_distance_mode!r}.")
        prepared = self._prepare_focal_inputs(focal_stack, focal_plane_distances, height, width)
        focal_features, focal_evidence = self._encode_focal_stack(prepared)
        clean_latent = self._sample_clean_latent(prepared, focal_features, num_inference_steps, generator, latents)
        heads = self._decode_prediction_heads(prepared, clean_latent, focal_evidence)
        fused = self._compute_reliability_fusion(prepared, heads, focal_distance_mode)
        trace = self._compute_consistency_diagnostics(prepared, fused)
        fused["diagnostics"]["physical_verification_trace"] = trace
        depth, uncertainty, aif, trace, refinement_summary = self._run_test_time_optimization(
            prepared,
            fused,
            trace,
            num_refinement_steps=num_refinement_steps,
            trace_refinement_epsilon=trace_refinement_epsilon,
            refinement_mode=refinement_mode,
            refinement_seed=refinement_seed,
            return_refinement_history=return_refinement_history,
        )
        fused["diagnostics"]["physical_verification_trace"] = trace
        return self._format_output(depth, aif, uncertainty, fused["diagnostics"], refinement_summary, return_dict)



    @staticmethod
    def _split_refinement_planes(num_planes: int, device: torch.device, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministically split focal planes into optimization and held-out subsets."""
        if num_planes < 2:
            idx = torch.arange(num_planes, device=device)
            return idx, idx
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        perm = torch.randperm(num_planes, device=device, generator=generator)
        heldout_count = max(1, num_planes // 3)
        return perm[heldout_count:], perm[:heldout_count]

    @staticmethod
    def _render_consistency_stack(all_in_focus_unit: torch.Tensor, depth_canonical: torch.Tensor, focal_plane_distances: torch.Tensor) -> torch.Tensor:
        """Differentiable approximate focal renderer for test-time consistency when metadata are absent."""
        if depth_canonical.dim() == 3:
            depth_canonical = depth_canonical.unsqueeze(1)
        batch, _, height, width = depth_canonical.shape
        if focal_plane_distances.dim() == 1:
            focal_plane_distances = focal_plane_distances.unsqueeze(0).expand(batch, -1)
        coords = (focal_plane_distances - focal_plane_distances.min(dim=1, keepdim=True).values) / (
            focal_plane_distances.max(dim=1, keepdim=True).values - focal_plane_distances.min(dim=1, keepdim=True).values
        ).clamp(min=1e-6)
        blur = F.avg_pool2d(all_in_focus_unit, kernel_size=5, stride=1, padding=2)
        amount = (depth_canonical[:, None] - coords[:, :, None, None, None]).abs().clamp(0.0, 1.0)
        return all_in_focus_unit[:, None] * (1.0 - amount) + blur[:, None] * amount

    @classmethod
    def _heldout_measurement_loss(
        cls,
        focal_stack_unit: torch.Tensor,
        all_in_focus_unit: torch.Tensor,
        depth_canonical: torch.Tensor,
        focal_plane_distances: torch.Tensor,
        plane_indices: torch.Tensor,
    ) -> torch.Tensor:
        rendered = cls._render_consistency_stack(all_in_focus_unit, depth_canonical, focal_plane_distances)
        residual = rendered.index_select(1, plane_indices) - focal_stack_unit.index_select(1, plane_indices)
        return torch.sqrt(residual.square() + 1e-4).mean()

    @classmethod
    def _selective_test_time_refinement(
        cls,
        *,
        focal_stack_unit: torch.Tensor,
        focal_plane_distances: torch.Tensor,
        final_depth_canonical: torch.Tensor,
        focus_depth_canonical: torch.Tensor,
        prior_depth_canonical: torch.Tensor,
        all_in_focus_unit: torch.Tensor,
        uncertainty_final: torch.Tensor,
        trace: PhysicalVerificationTrace,
        seed: int = 0,
        inner_steps: int = 4,
        lr: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimize detached per-sample AIF/depth state and validate on held-out planes."""
        torch.manual_seed(int(seed))
        depth0 = final_depth_canonical.detach().clamp(1e-4, 1.0 - 1e-4)
        aif0 = all_in_focus_unit.detach().clamp(0.0, 1.0)
        depth_logits = torch.logit(depth0).detach().clone().requires_grad_(True)
        aif_delta = torch.zeros_like(aif0, requires_grad=True)
        train_idx, val_idx = cls._split_refinement_planes(focal_stack_unit.shape[1], focal_stack_unit.device, seed)
        optimizer = torch.optim.Adam([depth_logits, aif_delta], lr=lr)
        focus = focus_depth_canonical.detach()
        prior = prior_depth_canonical.detach()
        texture_gate = trace.texture_confidence.detach().clamp(0.0, 1.0)
        for _ in range(inner_steps):
            optimizer.zero_grad()
            depth = depth_logits.sigmoid()
            aif = (aif0 + aif_delta.tanh() * 0.05).clamp(0.0, 1.0)
            measurement = cls._heldout_measurement_loss(focal_stack_unit, aif, depth, focal_plane_distances, train_idx)
            focus_loss = (texture_gate * (depth - focus).abs()).mean()
            trust = (depth - depth0).square().mean() + 0.25 * (depth - prior).square().mean()
            sharp = -0.01 * (aif[:, :, :, 1:] - aif[:, :, :, :-1]).abs().mean()
            edge = (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs().mean() * 0.005
            loss = measurement + 0.2 * focus_loss + 0.1 * trust + sharp + edge
            loss.backward()
            optimizer.step()
        candidate_depth = depth_logits.sigmoid().detach()
        candidate_aif = (aif0 + aif_delta.tanh() * 0.05).clamp(0.0, 1.0).detach()
        heldout_residual = cls._heldout_measurement_loss(focal_stack_unit, candidate_aif, candidate_depth, focal_plane_distances, val_idx).detach()
        branch_disagreement = (candidate_depth - focus).abs().detach()
        focus_entropy = trace.focus_entropy.detach().clamp(0.0, 1.0)
        split_variance = (candidate_depth - depth0).abs().detach()
        uncertainty = torch.maximum(uncertainty_final.detach(), branch_disagreement)
        uncertainty = torch.maximum(uncertainty, focus_entropy)
        uncertainty = torch.maximum(uncertainty, split_variance)
        uncertainty = torch.maximum(uncertainty, heldout_residual.reshape(1, 1, 1, 1).expand_as(uncertainty)).clamp(0.0, 1.0)
        return candidate_depth, uncertainty, candidate_aif

    @staticmethod
    def _physical_risk(trace: PhysicalVerificationTrace) -> float:
        """Return a scalar physical risk without merging conflict and invalid events."""
        conflict = trace.conflict_score.detach().float().clamp(0.0, 1.0)
        invalid = trace.invalid_score.detach().float().clamp(0.0, 1.0)
        support = trace.focus_support.detach().float().clamp(0.0, 1.0)
        return float((0.5 * conflict + 0.5 * invalid + 0.25 * (1.0 - support)).mean().item())


    @staticmethod
    def _should_accept_refinement(current_risk: torch.Tensor | float, candidate_risk: torch.Tensor | float, epsilon: float = 1e-4) -> bool:
        """Accept a candidate only when its scalar risk improves by ``epsilon``."""
        current = float(current_risk.detach().item()) if isinstance(current_risk, torch.Tensor) else float(current_risk)
        candidate = float(candidate_risk.detach().item()) if isinstance(candidate_risk, torch.Tensor) else float(candidate_risk)
        return candidate <= current - epsilon

    @classmethod
    def _accept_refinement_candidate(
        cls,
        current_trace: PhysicalVerificationTrace,
        candidate_trace: PhysicalVerificationTrace,
        epsilon: float = 1e-4,
    ) -> tuple[bool, float, float]:
        """Accept a candidate only when physical risk decreases by ``epsilon``."""
        current_risk = cls._physical_risk(current_trace)
        candidate_risk = cls._physical_risk(candidate_trace)
        return cls._should_accept_refinement(current_risk, candidate_risk, epsilon), current_risk, candidate_risk

    @staticmethod
    def _apply_trace_refinement(
        final_depth_canonical: torch.Tensor,
        focal_depth_canonical: torch.Tensor,
        generated_depth_canonical: torch.Tensor,
        uncertainty_final: torch.Tensor,
        trace: PhysicalVerificationTrace,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one conservative inference-time trace-guided refinement step."""
        if final_depth_canonical.dim() == 3:
            final_depth_canonical = final_depth_canonical.unsqueeze(1)
        target_size = final_depth_canonical.shape[-2:]

        def resize_map(value: torch.Tensor) -> torch.Tensor:
            """Resize a trace or prediction map to the current depth resolution."""
            if value.dim() == 3:
                value = value.unsqueeze(1)
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
            return value.to(device=final_depth_canonical.device, dtype=final_depth_canonical.dtype)

        focal_depth = resize_map(focal_depth_canonical).clamp(0.0, 1.0)
        generated_depth = resize_map(generated_depth_canonical).clamp(0.0, 1.0)
        conflict = resize_map(trace.conflict_score).detach().clamp(0.0, 1.0)
        invalid = resize_map(trace.invalid_score).detach().clamp(0.0, 1.0)
        focus_support = resize_map(trace.focus_support).detach().clamp(0.0, 1.0)
        generation_support = resize_map(trace.generation_support).detach().clamp(0.0, 1.0)

        reliability = (1.0 - torch.maximum(conflict, invalid)).clamp(0.0, 1.0)
        focus_weight = (focus_support * reliability).clamp(0.0, 1.0)
        generation_weight = (generation_support * reliability).clamp(0.0, 1.0)
        total_weight = (focus_weight + generation_weight).clamp(min=1e-6)
        trace_depth = (focus_weight * focal_depth + generation_weight * generated_depth) / total_weight
        update_weight = (0.35 * reliability * (focus_support + generation_support).mul(0.5)).clamp(0.0, 0.35)
        refined_depth = ((1.0 - update_weight) * final_depth_canonical + update_weight * trace_depth).clamp(0.0, 1.0)

        uncertainty = resize_map(uncertainty_final).clamp(0.0, 1.0)
        refined_uncertainty = torch.maximum(uncertainty, torch.maximum(conflict, invalid)).clamp(0.0, 1.0)
        return refined_depth, refined_uncertainty

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


    def _ensure_tensor_stack(self, stack: torch.Tensor) -> torch.Tensor:
        """Validate and convert a focal-stack tensor to model range."""
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
        return to_model_range(stack)



# Backward-compatible aliases for external scripts using pre-rename APIs.
@dataclass
class FocalDiffusionOutput(FocalStackGenerationOutput):
    """Deprecated compatibility alias for :class:`FocalStackGenerationOutput`."""

    def __post_init__(self) -> None:
        warnings.warn(
            "FocalDiffusionOutput is deprecated; use FocalStackGenerationOutput instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class FocalDiffusionPipeline(FocalStackGenerationPipeline):
    """Deprecated compatibility alias for :class:`FocalStackGenerationPipeline`."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "FocalDiffusionPipeline is deprecated; use FocalStackGenerationPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
