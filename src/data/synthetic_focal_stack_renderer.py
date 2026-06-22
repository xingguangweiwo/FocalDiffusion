"""Differentiable focal-stack renderer with metric thin-lens units."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


class SyntheticFocalStackRenderer:
    """Render focal stacks from AIF RGB and metric depth using a blur basis.

    Units: depth, focus distance, focal length, aperture, and pixel size are in
    metres. The main rendering path is batched and differentiable with respect
    to both the all-in-focus image and depth map.
    """

    def __init__(
        self,
        default_f_number: float = 8.0,
        default_focal_length: float = 50e-3,
        default_pixel_size: float = 1.2e-5,
        min_sigma: float = 0.05,
        max_sigma: float = 8.0,
        sigma_quantisation: float = 0.25,
        kernel_limit: int = 25,
        eps: float = 1e-6,
        psf_type: str = "gaussian",
        num_blur_levels: int | None = None,
        exposure_normalize: bool = True,
    ) -> None:
        if psf_type not in {"gaussian", "disc"}:
            raise ValueError("psf_type must be 'gaussian' or 'disc'.")
        self.default_f_number = default_f_number
        self.default_focal_length = default_focal_length
        self.default_pixel_size = default_pixel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_quantisation = max(sigma_quantisation, 1e-3)
        self.kernel_limit = max(kernel_limit + (1 - kernel_limit % 2), 5)
        self.eps = eps
        self.psf_type = psf_type
        self.num_blur_levels = num_blur_levels or int(math.ceil(max_sigma / self.sigma_quantisation)) + 1
        self.exposure_normalize = exposure_normalize

    def generate(
        self,
        all_in_focus: torch.Tensor,
        depth: torch.Tensor,
        focal_plane_distances: torch.Tensor,
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Create a focal stack for focus distances in metres.

        Accepts ``all_in_focus`` as ``[C,H,W]`` or ``[B,C,H,W]`` in unit range,
        ``depth`` as ``[H,W]``, ``[1,H,W]``, ``[B,H,W]`` or ``[B,1,H,W]`` in
        metres, and focal distances as ``[K]`` or ``[B,K]`` in metres. Returns
        ``[K,C,H,W]`` for unbatched input and ``[B,K,C,H,W]`` for batched input.
        """
        unbatched = all_in_focus.dim() == 3
        if unbatched:
            all_in_focus = all_in_focus.unsqueeze(0)
        if all_in_focus.dim() != 4:
            raise ValueError("all_in_focus must have shape [C,H,W] or [B,C,H,W]")
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1) if depth.shape[0] == all_in_focus.shape[0] else depth.unsqueeze(0)
        if depth.dim() != 4 or depth.shape[1] != 1:
            raise ValueError("depth must have shape [H,W], [1,H,W], [B,H,W] or [B,1,H,W]")
        image = all_in_focus.float() if not torch.is_floating_point(all_in_focus) else all_in_focus
        depth = depth.to(device=image.device, dtype=image.dtype)
        if depth.shape[0] == 1 and image.shape[0] != 1:
            depth = depth.expand(image.shape[0], -1, -1, -1)
        if depth.shape[0] != image.shape[0]:
            raise ValueError("depth batch must be 1 or match all_in_focus batch")
        focus = torch.as_tensor(focal_plane_distances, device=image.device, dtype=image.dtype)
        if focus.dim() == 1:
            focus = focus.unsqueeze(0).expand(image.shape[0], -1)
        elif focus.dim() == 2 and focus.shape[0] == 1 and image.shape[0] != 1:
            focus = focus.expand(image.shape[0], -1)
        if focus.dim() != 2 or focus.shape[0] != image.shape[0]:
            raise ValueError("focal_plane_distances must have shape [K] or [B,K]")

        camera = self._prepare_camera_params(camera_params, image.device, image.dtype, image.shape[0])
        sigma = self._compute_sigma_map(depth, focus, camera)
        rendered = self.render_from_sigma(image, sigma)
        return rendered.squeeze(0) if unbatched else rendered

    def render_from_sigma(self, image: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Blend a precomputed blur basis with continuous CoC-derived weights."""
        if sigma.dim() != 4:
            raise ValueError("sigma must have shape [B,K,H,W]")
        batch, channels, height, width = image.shape
        _, planes, _, _ = sigma.shape
        if self.exposure_normalize:
            image = image / image.mean(dim=(-2, -1), keepdim=True).clamp(min=self.eps) * image.detach().mean(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
        levels = torch.linspace(0.0, self.max_sigma, self.num_blur_levels, device=image.device, dtype=image.dtype)
        basis = torch.stack([self._blur(image, level) for level in levels], dim=1)  # [B,L,C,H,W]
        scaled = (sigma.clamp(0.0, self.max_sigma) / self.max_sigma) * (self.num_blur_levels - 1)
        lower = torch.floor(scaled).long().clamp(0, self.num_blur_levels - 1)
        upper = (lower + 1).clamp(0, self.num_blur_levels - 1)
        frac = (scaled - lower.to(dtype=scaled.dtype)).unsqueeze(2)
        gather_shape = (batch, planes, channels, height, width)
        lower_basis = torch.gather(basis, 1, lower[:, :, None].expand(gather_shape))
        upper_basis = torch.gather(basis, 1, upper[:, :, None].expand(gather_shape))
        return lower_basis * (1.0 - frac) + upper_basis * frac

    def _prepare_camera_params(self, camera_params: Optional[Dict[str, torch.Tensor]], device: torch.device, dtype: torch.dtype, batch: int) -> Dict[str, torch.Tensor]:
        camera: Dict[str, torch.Tensor] = {}
        for key, default in {
            "f_number": self.default_f_number,
            "focal_length": self.default_focal_length,
            "pixel_size": self.default_pixel_size,
        }.items():
            value = camera_params.get(key, default) if camera_params else default
            tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
            if tensor.numel() == 1:
                tensor = tensor.expand(batch)
            if tensor.numel() != batch:
                raise ValueError(f"camera parameter {key} must be scalar or length B")
            camera[key] = tensor.view(batch, 1, 1, 1)
        if camera_params and "aperture" in camera_params:
            aperture = torch.as_tensor(camera_params["aperture"], device=device, dtype=dtype).reshape(-1)
            if aperture.numel() == 1:
                aperture = aperture.expand(batch)
            camera["aperture"] = aperture.view(batch, 1, 1, 1)
        else:
            camera["aperture"] = camera["focal_length"] / camera["f_number"].clamp(min=self.eps)
        return camera

    def _compute_sigma_map(self, depth: torch.Tensor, focus: torch.Tensor, camera: Dict[str, torch.Tensor]) -> torch.Tensor:
        focal_length = camera["focal_length"][:, None]
        aperture = camera["aperture"][:, None]
        pixel_size = camera["pixel_size"][:, None]
        focus = focus[:, :, None, None, None].clamp(min=self.eps)
        depth_safe = torch.where(torch.isfinite(depth[:, None]) & (depth[:, None] > self.eps), depth[:, None], focus)
        image_dist_focus = (focus * focal_length) / (focus - focal_length).clamp(min=self.eps)
        coc = aperture * image_dist_focus.abs() * (
            1.0 / focal_length.clamp(min=self.eps)
            - 1.0 / image_dist_focus.clamp(min=self.eps)
            - 1.0 / depth_safe.clamp(min=self.eps)
        ).abs()
        return (coc / (2.0 * pixel_size.clamp(min=self.eps))).squeeze(2).clamp(0.0, self.max_sigma)

    def _blur(self, image: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_t = sigma.to(device=image.device, dtype=image.dtype).clamp(min=self.min_sigma)
        kernel = self._kernel(sigma_t, image.shape[1])
        blurred = F.conv2d(image, kernel, padding=kernel.shape[-1] // 2, groups=image.shape[1])
        identity_weight = (sigma <= self.min_sigma).to(device=image.device, dtype=image.dtype)
        return identity_weight * image + (1.0 - identity_weight) * blurred

    def _kernel(self, sigma: torch.Tensor, channels: int) -> torch.Tensor:
        radius = self.kernel_limit // 2
        coords = torch.arange(-radius, radius + 1, device=sigma.device, dtype=sigma.dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rr2 = xx.square() + yy.square()
        if self.psf_type == "gaussian":
            kernel = torch.exp(-rr2 / (2.0 * sigma.square().clamp(min=self.eps)))
        else:
            edge = torch.sigmoid((sigma.square() - rr2) / (sigma.clamp(min=self.eps)))
            kernel = edge
        kernel = kernel / kernel.sum().clamp(min=self.eps)
        return kernel.view(1, 1, self.kernel_limit, self.kernel_limit).repeat(channels, 1, 1, 1)


# Backward-compatible alias for external scripts using the pre-rename API.
FocalStackSimulator = SyntheticFocalStackRenderer
