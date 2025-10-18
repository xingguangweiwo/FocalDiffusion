"""
FocalDiffusion Pipeline - Main inference pipeline with SD3.5 integration
Complete implementation with proper SD3.5 architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    AutoencoderKL
)
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.utils import BaseOutput
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)


@dataclass
class FocalDiffusionOutput(BaseOutput):
    """Output class for FocalDiffusion Pipeline"""
    depth_map: torch.Tensor
    all_in_focus_image: Union[torch.Tensor, Image.Image]
    depth_colored: Optional[Image.Image] = None
    uncertainty: Optional[torch.Tensor] = None
    attention_maps: Optional[Dict[str, torch.Tensor]] = None
    focal_features: Optional[Dict[str, torch.Tensor]] = None


class FocalInjectedSD3Transformer(SD3Transformer2DModel):
    """SD3.5 Transformer with focal stack feature injection"""

    def __init__(self, config, focal_processor=None, camera_encoder=None):
        super().__init__(config)

        self.focal_processor = focal_processor
        self.camera_encoder = camera_encoder

        # Add focal cross-attention layers
        self.focal_injection_layers = [4, 8, 12, 16, 20, 24]
        hidden_size = config.attention_head_dim * config.num_attention_heads

        # Add focal cross-attention modules
        self.focal_cross_attns = nn.ModuleList()
        for i in range(config.num_layers):
            if i in self.focal_injection_layers:
                self.focal_cross_attns.append(
                    FocalCrossAttention(
                        hidden_size=hidden_size,
                        num_heads=config.num_attention_heads,
                        head_dim=config.attention_head_dim,
                    )
                )
            else:
                self.focal_cross_attns.append(None)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            pooled_projections: torch.Tensor,
            focal_features: Optional[Dict[str, torch.Tensor]] = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
    ):
        """Forward with focal injection"""

        # Standard SD3.5 processing
        hidden_states = super().forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # Apply focal cross-attention if features provided
        if focal_features is not None:
            B, C, H, W = hidden_states.shape
            hidden_states_seq = hidden_states.flatten(2).transpose(1, 2)

            # Get focal features
            focal_feats = focal_features.get('fused_features')
            if focal_feats is not None:
                # Resize if needed
                if focal_feats.shape[-2:] != (H, W):
                    focal_feats = F.interpolate(focal_feats, size=(H, W), mode='bilinear')

                focal_feats_seq = focal_feats.flatten(2).transpose(1, 2)

                # Apply focal cross-attention at specific layers
                for i, focal_attn in enumerate(self.focal_cross_attns):
                    if focal_attn is not None and i in self.focal_injection_layers:
                        hidden_states_seq = hidden_states_seq + focal_attn(
                            hidden_states_seq,
                            focal_feats_seq
                        )

                hidden_states = hidden_states_seq.transpose(1, 2).reshape(B, C, H, W)

        if return_dict:
            from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=hidden_states)

        return (hidden_states,)


class FocalCrossAttention(nn.Module):
    """Cross-attention module for focal feature injection"""

    def __init__(self, hidden_size: int, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_size)

        # QK normalization for SD3.5
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        B, N, D = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.to_out(out)

        return out


class DualOutputDecoder(nn.Module):
    """Decoder for generating both depth and all-in-focus images"""

    def __init__(
            self,
            in_channels: int = 16,
            out_channels_depth: int = 1,
            out_channels_rgb: int = 3,
            hidden_dims: List[int] = [512, 256, 128, 64],
    ):
        super().__init__()

        # Shared encoder
        self.shared_encoder = nn.ModuleList()
        current_dim = in_channels

        for hidden_dim in hidden_dims[:2]:
            self.shared_encoder.append(
                nn.Sequential(
                    nn.Conv2d(current_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                )
            )
            current_dim = hidden_dim

        # Depth decoder branch
        self.depth_decoder = nn.ModuleList()
        for hidden_dim in hidden_dims[2:]:
            self.depth_decoder.append(
                nn.Sequential(
                    nn.Conv2d(current_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                )
            )
            current_dim = hidden_dim

        self.depth_head = nn.Conv2d(current_dim, out_channels_depth, 1)

        # RGB decoder branch
        current_dim = hidden_dims[1]
        self.rgb_decoder = nn.ModuleList()
        for hidden_dim in hidden_dims[2:]:
            self.rgb_decoder.append(
                nn.Sequential(
                    nn.Conv2d(current_dim, hidden_dim, 3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                )
            )
            current_dim = hidden_dim

        self.rgb_head = nn.Conv2d(current_dim, out_channels_rgb, 1)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate depth and RGB from latents"""

        # Shared encoding
        x = latents
        shared_features = []
        for layer in self.shared_encoder:
            x = layer(x)
            shared_features.append(x)

        # Depth branch
        depth = shared_features[-1]
        for layer in self.depth_decoder:
            depth = layer(depth)
        depth = self.depth_head(depth)

        # RGB branch
        rgb = shared_features[-1]
        for layer in self.rgb_decoder:
            rgb = layer(rgb)
        rgb = self.rgb_head(rgb)

        return depth, rgb


class FocalDiffusionPipeline(StableDiffusion3Pipeline):
    """Main FocalDiffusion Pipeline with SD3.5"""

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
    ):
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

        # Import focal modules
        if focal_processor is None:
            from .focal_processor import FocalStackProcessor
            focal_processor = FocalStackProcessor()

        if camera_encoder is None:
            from .camera_invariant import CameraInvariantEncoder
            camera_encoder = CameraInvariantEncoder()

        if dual_decoder is None:
            dual_decoder = DualOutputDecoder()

        self.focal_processor = focal_processor
        self.camera_encoder = camera_encoder
        self.dual_decoder = dual_decoder

        # Replace transformer with focal-injected version if needed
        if not hasattr(transformer, 'focal_cross_attns'):
            config = transformer.config
            self.transformer = FocalInjectedSD3Transformer(
                config,
                focal_processor=focal_processor,
                camera_encoder=camera_encoder
            )
            # Copy weights from original transformer
            self.transformer.load_state_dict(transformer.state_dict(), strict=False)

    @torch.no_grad()
    def __call__(
            self,
            focal_stack: Union[torch.Tensor, List[Image.Image]],
            focus_distances: torch.Tensor,
            camera_params: Optional[Dict[str, torch.Tensor]] = None,
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
            ensemble_size: int = 1,
            **kwargs,
    ):
        """Generate depth and all-in-focus image from focal stack"""

        # Preprocess focal stack
        if isinstance(focal_stack, list):
            focal_stack = self._preprocess_focal_stack(focal_stack)

        device = self._execution_device
        dtype = self.transformer.dtype

        focal_stack = focal_stack.to(device=device, dtype=dtype)
        focus_distances = focus_distances.to(device=device, dtype=dtype)

        # Extract focal features
        if camera_params is not None:
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

        # Encode prompt with triple encoding (SD3.5 feature)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            self.encode_prompt(
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
        )

        # Prepare latents
        batch_size = focal_stack.shape[0] * num_images_per_prompt
        latents = self.prepare_latents(
            batch_size,
            self.transformer.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
            dtype,
            device,
            generator,
            latents,
        )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Denoising loop with focal injection
        for t in tqdm(timesteps):
            # Expand latents if using guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

            # Add noise
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise with focal injection
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=torch.cat(
                    [negative_prompt_embeds, prompt_embeds]) if guidance_scale > 1.0 else prompt_embeds,
                pooled_projections=torch.cat([negative_pooled_prompt_embeds,
                                              pooled_prompt_embeds]) if guidance_scale > 1.0 else pooled_prompt_embeds,
                focal_features=focal_features,
                return_dict=False,
            )[0]

            # Guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode with dual decoder
        depth_map, all_in_focus = self.dual_decoder(latents)

        # VAE decode for RGB
        all_in_focus = self.vae.decode(all_in_focus / self.vae.config.scaling_factor, return_dict=False)[0]

        # Post-process
        depth_map = torch.sigmoid(depth_map).squeeze(1)
        all_in_focus = (all_in_focus / 2 + 0.5).clamp(0, 1)

        # Convert to desired output format
        if output_type == "pil":
            all_in_focus = self.numpy_to_pil(all_in_focus.cpu().permute(0, 2, 3, 1).numpy())[0]
            depth_colored = self._colorize_depth(depth_map[0].cpu().numpy())
        else:
            depth_colored = None

        if not return_dict:
            return depth_map, all_in_focus

        return FocalDiffusionOutput(
            depth_map=depth_map,
            all_in_focus_image=all_in_focus,
            depth_colored=depth_colored,
            focal_features=focal_features,
        )

    def _preprocess_focal_stack(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert list of PIL images to tensor"""
        tensors = []
        for img in images:
            img = img.convert('RGB')
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            tensor = (tensor * 2.0) - 1.0  # Normalize to [-1, 1]
            tensors.append(tensor)

        focal_stack = torch.stack(tensors).unsqueeze(0)
        return focal_stack

    def _colorize_depth(self, depth: np.ndarray) -> Image.Image:
        """Colorize depth map for visualization"""
        import matplotlib.pyplot as plt

        # Normalize depth
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # Apply colormap
        cmap = plt.get_cmap('spectral')
        depth_colored = cmap(depth_norm)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)

        return Image.fromarray(depth_colored)