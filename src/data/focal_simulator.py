"""Utilities to synthesise focal stacks from an all-in-focus image and depth map."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Optional

import torch
import torch.nn.functional as F


class FocalStackSimulator:
    """Generate focal stacks via a thin-lens circle-of-confusion model.

    The implementation mirrors the MATLAB reference shared by the project authors:
    it approximates the spatially varying point-spread function by quantising the
    per-pixel blur radius and blending Gaussian-blurred versions of the source image.
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
    ) -> None:
        self.default_f_number = default_f_number
        self.default_focal_length = default_focal_length
        self.default_pixel_size = default_pixel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_quantisation = max(sigma_quantisation, 1e-3)
        self.kernel_limit = max(kernel_limit, 5)
        if self.kernel_limit % 2 == 0:
            self.kernel_limit += 1
        self.eps = eps

    def generate(
        self,
        all_in_focus: torch.Tensor,
        depth: torch.Tensor,
        focus_distances: torch.Tensor,
        camera_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Create a focal stack for the requested focus distances.

        Args:
            all_in_focus: Tensor of shape [C, H, W] in range [0, 1].
            depth: Tensor of shape [1, H, W] or [H, W] with metric depth in metres.
            focus_distances: 1D tensor/list with focus distances in metres.
            camera_params: Optional mapping containing camera metadata. Supported keys
                include ``f_number``, ``focal_length``, ``pixel_size`` and ``aperture``.
        """

        if all_in_focus.dim() != 3:
            raise ValueError("all_in_focus must have shape [C, H, W]")

        if depth.dim() == 3 and depth.size(0) == 1:
            depth = depth.squeeze(0)
        if depth.dim() != 2:
            raise ValueError("depth must have shape [H, W] or [1, H, W]")

        if not torch.is_floating_point(all_in_focus):
            all_in_focus = all_in_focus.float()
        if not torch.is_floating_point(depth):
            depth = depth.float()

        if isinstance(focus_distances, torch.Tensor):
            focus_distances_tensor = focus_distances.flatten().to(
                dtype=all_in_focus.dtype, device=all_in_focus.device
            )
        else:
            focus_distances_tensor = torch.as_tensor(
                focus_distances, dtype=all_in_focus.dtype, device=all_in_focus.device
            )

        if focus_distances_tensor.numel() == 0:
            raise ValueError("focus_distances must contain at least one value")

        camera = self._prepare_camera_params(camera_params, all_in_focus.device, all_in_focus.dtype)
        stack = []
        for focus_depth in focus_distances_tensor:
            sigma_map = self._compute_sigma_map(depth, focus_depth, camera)
            frame = self._apply_spatially_variant_blur(all_in_focus, sigma_map)
            stack.append(frame)

        return torch.stack(stack, dim=0)

    def _prepare_camera_params(
        self,
        camera_params: Optional[Dict[str, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        camera: Dict[str, torch.Tensor] = {}
        if camera_params:
            for key, value in camera_params.items():
                if isinstance(value, torch.Tensor):
                    camera[key] = value.to(device=device, dtype=dtype)
                else:
                    camera[key] = torch.tensor(float(value), device=device, dtype=dtype)

        def _get(name: str, default: float) -> torch.Tensor:
            if name in camera:
                return camera[name]
            camera[name] = torch.tensor(default, device=device, dtype=dtype)
            return camera[name]

        # Populate required defaults.
        f_number = _get("f_number", self.default_f_number)
        focal_length = _get("focal_length", self.default_focal_length)
        if "aperture" not in camera:
            camera["aperture"] = focal_length / torch.clamp(f_number, min=self.eps)
        _get("pixel_size", self.default_pixel_size)
        return camera

    def _compute_sigma_map(
        self,
        depth: torch.Tensor,
        focus_distance: torch.Tensor,
        camera: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        focal_length = camera["focal_length"]
        aperture = camera["aperture"]
        pixel_size = camera["pixel_size"]

        depth_safe = torch.where(
            torch.isfinite(depth) & (depth > self.eps), depth, focus_distance.clamp(min=self.eps)
        )

        focus_distance = focus_distance.clamp(min=self.eps)
        image_dist_focus = (focus_distance * focal_length) / (
            focus_distance - focal_length + self.eps
        )

        coc = aperture * torch.abs(image_dist_focus) * torch.abs(
            (1.0 / torch.clamp(focal_length, min=self.eps))
            - (1.0 / torch.clamp(image_dist_focus, min=self.eps))
            - (1.0 / depth_safe)
        )

        sigma = coc / (2.0 * torch.clamp(pixel_size, min=self.eps))
        sigma = torch.clamp(sigma, min=0.0, max=self.max_sigma)
        return sigma

    def _apply_spatially_variant_blur(
        self, image: torch.Tensor, sigma_map: torch.Tensor
    ) -> torch.Tensor:
        if sigma_map.dim() != 2:
            raise ValueError("sigma_map must have shape [H, W]")

        if image.device != sigma_map.device:
            sigma_map = sigma_map.to(device=image.device)

        quantised = torch.round(sigma_map / self.sigma_quantisation) * self.sigma_quantisation
        quantised = torch.clamp(quantised, min=0.0, max=self.max_sigma)

        result = image.clone()
        unique_sigmas = torch.unique(quantised)

        for sigma in unique_sigmas:
            sigma_value = float(sigma.item())
            if sigma_value <= self.min_sigma:
                continue

            kernel = self._gaussian_kernel(sigma_value, image.device, image.dtype)
            padding = kernel.shape[-1] // 2
            blurred = F.conv2d(
                image.unsqueeze(0), kernel, padding=padding, groups=image.shape[0]
            ).squeeze(0)

            mask = (quantised == sigma).to(image.dtype)
            if mask.sum() == 0:
                continue
            mask = mask.unsqueeze(0)
            result = torch.where(mask > 0, blurred, result)

        return result

    @lru_cache(maxsize=128)
    def _gaussian_kernel(
        self, sigma: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        radius = min(int(math.ceil(3 * sigma)), self.kernel_limit // 2)
        kernel_size = 2 * radius + 1
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2 + self.eps))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel = kernel / kernel.sum()
        return kernel.repeat(3, 1, 1, 1)
