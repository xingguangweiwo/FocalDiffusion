"""Attention and physics helpers used by the training stack."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossAttention(nn.Module):
    """Cross-attention used to inject focal features into the SD3 transformer."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_size)

        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(self, hidden_states: torch.Tensor, encoder_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_states)
        v = self.to_v(encoder_states)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attn = torch.matmul(attention_probs, v)

        attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.to_out(attn)


class PhysicsConsistencyModule(nn.Module):
    """
    Enforces physical consistency between outputs using optical models
    """

    def __init__(
            self,
            use_differentiable_rendering: bool = True,
            use_depth_regularization: bool = True,
            use_defocus_consistency: bool = True,
            max_coc_pixels: float = 50.0,
    ):
        super().__init__()

        self.use_differentiable_rendering = use_differentiable_rendering
        self.use_depth_regularization = use_depth_regularization
        self.use_defocus_consistency = use_defocus_consistency
        self.max_coc_pixels = max_coc_pixels

        # Learnable parameters for physical model calibration
        self.coc_scale = nn.Parameter(torch.tensor(1.0))
        self.blur_gamma = nn.Parameter(torch.tensor(2.2))

        # PSF (Point Spread Function) generator
        self.psf_generator = PSFGenerator()

        # Depth regularization network
        if use_depth_regularization:
            self.depth_regularizer = DepthRegularizer()

        # Defocus blur renderer
        if use_differentiable_rendering:
            self.defocus_renderer = DifferentiableDefocusRenderer(
                max_coc_pixels=max_coc_pixels
            )

    def forward(
            self,
            depth: torch.Tensor,
            all_in_focus: torch.Tensor,
            focal_stack: torch.Tensor,
            focus_distances: torch.Tensor,
            camera_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-based consistency losses and refined outputs

        Args:
            depth: Predicted depth map [B, 1, H, W]
            all_in_focus: Predicted all-in-focus image [B, 3, H, W]
            focal_stack: Input focal stack [B, N, 3, H, W]
            focus_distances: Focus distances [B, N]
            camera_params: Camera parameters

        Returns:
            Dictionary with losses and refined outputs
        """
        B, N, C, H, W = focal_stack.shape
        device = depth.device

        losses = {}
        outputs = {}

        # 1. Depth regularization
        if self.use_depth_regularization:
            depth_reg_loss = self.depth_regularizer(depth)
            losses['depth_regularization'] = depth_reg_loss

        # 2. Defocus consistency - render focal stack from depth and AIF
        if self.use_defocus_consistency:
            rendered_stack = self.render_focal_stack(
                all_in_focus, depth, focus_distances, camera_params
            )

            # Compute reconstruction loss
            recon_loss = F.l1_loss(rendered_stack, focal_stack)
            losses['reconstruction'] = recon_loss

            outputs['rendered_stack'] = rendered_stack

        # 3. Circle of Confusion consistency
        coc_maps = self.compute_coc_maps(depth, focus_distances, camera_params)

        # Ensure CoC is physically plausible
        coc_loss = self.compute_coc_consistency_loss(coc_maps, focal_stack)
        losses['coc_consistency'] = coc_loss

        outputs['coc_maps'] = coc_maps

        # 4. All-in-focus sharpness constraint
        sharpness_loss = self.compute_sharpness_loss(all_in_focus, focal_stack)
        losses['sharpness'] = sharpness_loss

        # 5. Depth ordering constraint
        ordering_loss = self.compute_depth_ordering_loss(depth, focal_stack, focus_distances)
        losses['depth_ordering'] = ordering_loss

        # Total physics loss
        total_loss = sum(losses.values())
        losses['total_physics'] = total_loss

        return {
            'losses': losses,
            'outputs': outputs,
            'refined_depth': self.refine_depth(depth, coc_maps),
            'refined_aif': self.refine_all_in_focus(all_in_focus, focal_stack),
        }

    def render_focal_stack(
            self,
            all_in_focus: torch.Tensor,
            depth: torch.Tensor,
            focus_distances: torch.Tensor,
            camera_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Render focal stack from all-in-focus image and depth using defocus blur
        """
        if not self.use_differentiable_rendering:
            return torch.zeros_like(all_in_focus).unsqueeze(1).expand(-1, len(focus_distances), -1, -1, -1)

        B = all_in_focus.shape[0]
        N = focus_distances.shape[1]

        rendered_images = []

        for n in range(N):
            focus_d = focus_distances[:, n]

            # Compute CoC for this focus distance
            coc = self.compute_single_coc(depth, focus_d, camera_params)

            # Apply defocus blur
            blurred = self.defocus_renderer(all_in_focus, coc)
            rendered_images.append(blurred)

        rendered_stack = torch.stack(rendered_images, dim=1)
        return rendered_stack

    def compute_coc_maps(
            self,
            depth: torch.Tensor,
            focus_distances: torch.Tensor,
            camera_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute Circle of Confusion maps for all focus distances
        """
        B, _, H, W = depth.shape
        N = focus_distances.shape[1]

        # Extract camera parameters
        focal_length = camera_params.get('focal_length', torch.tensor(0.050))
        aperture = camera_params.get('aperture', torch.tensor(2.8))

        if focal_length.dim() == 0:
            focal_length = focal_length.unsqueeze(0).expand(B)
        if aperture.dim() == 0:
            aperture = aperture.unsqueeze(0).expand(B)

        # Compute CoC for each focus distance
        coc_maps = []

        for n in range(N):
            focus_d = focus_distances[:, n:n + 1].unsqueeze(-1).unsqueeze(-1)

            # Thin lens equation
            coc = self.compute_single_coc(depth, focus_d, camera_params)
            coc_maps.append(coc)

        return torch.stack(coc_maps, dim=1)

    def compute_single_coc(
            self,
            depth: torch.Tensor,
            focus_distance: torch.Tensor,
            camera_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute Circle of Confusion for a single focus distance
        """
        # Extract camera parameters
        focal_length = camera_params.get('focal_length', torch.tensor(0.050))
        aperture = camera_params.get('aperture', torch.tensor(2.8))

        # Ensure proper shapes
        if focal_length.dim() == 0:
            focal_length = focal_length.view(1, 1, 1, 1)
        else:
            focal_length = focal_length.view(-1, 1, 1, 1)

        if aperture.dim() == 0:
            aperture = aperture.view(1, 1, 1, 1)
        else:
            aperture = aperture.view(-1, 1, 1, 1)

        if focus_distance.dim() < 4:
            while focus_distance.dim() < 4:
                focus_distance = focus_distance.unsqueeze(-1)

        # Compute image distances using thin lens equation
        # 1/f = 1/u + 1/v => v = fu/(u-f)
        v_objects = (focal_length * depth) / (depth - focal_length + 1e-8)
        v_focus = (focal_length * focus_distance) / (focus_distance - focal_length + 1e-8)

        # Circle of Confusion: C = |v - v_focus| * D / v
        # where D is the entrance pupil diameter
        pupil_diameter = focal_length / aperture
        coc = torch.abs(v_objects - v_focus) * pupil_diameter / (v_objects + 1e-8)

        # Apply learned scaling and clamp
        coc = coc * self.coc_scale
        coc = torch.clamp(coc, 0, self.max_coc_pixels)

        return coc

    def compute_coc_consistency_loss(
            self,
            coc_maps: torch.Tensor,
            focal_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ensure CoC maps are consistent with observed blur in focal stack
        """
        B, N, _, H, W = focal_stack.shape

        # Estimate blur from focal stack gradients
        estimated_blur = []

        for n in range(N):
            img = focal_stack[:, n]

            # Compute gradient magnitude as sharpness measure
            grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])

            # Pad to original size
            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = F.pad(grad_y, (0, 0, 0, 1))

            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

            # Inverse gradient magnitude correlates with blur
            blur_estimate = 1.0 / (grad_mag.mean(dim=1, keepdim=True) + 0.1)
            estimated_blur.append(blur_estimate)

        estimated_blur = torch.stack(estimated_blur, dim=1)

        # Normalize both CoC and estimated blur
        coc_norm = coc_maps / (coc_maps.max() + 1e-8)
        blur_norm = estimated_blur / (estimated_blur.max() + 1e-8)

        # Consistency loss
        consistency_loss = F.mse_loss(coc_norm, blur_norm)

        return consistency_loss

    def compute_sharpness_loss(
            self,
            all_in_focus: torch.Tensor,
            focal_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ensure all-in-focus image is sharper than any individual focal image
        """

        # Compute sharpness using Laplacian variance
        def compute_sharpness(img):
            # Laplacian kernel
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

            # Apply Laplacian
            if img.dim() == 4:
                kernel = kernel.expand(img.shape[1], 1, 3, 3)
                laplacian = F.conv2d(img, kernel, padding=1, groups=img.shape[1])
            else:
                kernel = kernel.expand(img.shape[2], 1, 3, 3)
                B, N, C, H, W = img.shape
                img_flat = img.view(B * N, C, H, W)
                laplacian = F.conv2d(img_flat, kernel, padding=1, groups=C)
                laplacian = laplacian.view(B, N, C, H, W)

            # Variance of Laplacian
            sharpness = laplacian.var(dim=(-2, -1))
            return sharpness

        # Sharpness of all-in-focus
        aif_sharpness = compute_sharpness(all_in_focus)

        # Maximum sharpness in focal stack
        stack_sharpness = compute_sharpness(focal_stack)
        max_stack_sharpness = stack_sharpness.max(dim=1)[0]

        # AIF should be at least as sharp as the sharpest focal image
        sharpness_loss = F.relu(max_stack_sharpness - aif_sharpness).mean()

        return sharpness_loss

    def compute_depth_ordering_loss(
            self,
            depth: torch.Tensor,
            focal_stack: torch.Tensor,
            focus_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ensure depth ordering is consistent with focus quality
        """
        B, N = focus_distances.shape

        # Find sharpest image for each pixel
        sharpness_maps = []

        for n in range(N):
            img = focal_stack[:, n]

            # Local sharpness using gradient magnitude
            grad_x = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
            grad_y = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])

            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = F.pad(grad_y, (0, 0, 0, 1))

            sharpness = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8).mean(dim=1, keepdim=True)
            sharpness_maps.append(sharpness)

        sharpness_maps = torch.stack(sharpness_maps, dim=1)

        # Find index of sharpest image for each pixel
        sharpest_idx = sharpness_maps.argmax(dim=1)

        # Corresponding focus distance should be close to depth
        focus_at_sharpest = torch.gather(
            focus_distances.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, depth.shape[-2], depth.shape[-1]),
            1,
            sharpest_idx
        )

        # Ordering loss
        ordering_loss = F.smooth_l1_loss(depth, focus_at_sharpest)

        return ordering_loss

    def refine_depth(
            self,
            depth: torch.Tensor,
            coc_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine depth using CoC information
        """
        # Simple refinement: smooth depth in regions with high CoC
        avg_coc = coc_maps.mean(dim=1)

        # Gaussian smoothing weighted by CoC
        kernel_size = 5
        sigma = 1.0

        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=depth.device)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        # Weight kernel by CoC
        weighted_kernel = kernel * (1 + avg_coc / self.max_coc_pixels)

        # Apply smoothing
        refined_depth = F.conv2d(depth, weighted_kernel, padding=kernel_size // 2)

        return refined_depth

    def refine_all_in_focus(
            self,
            all_in_focus: torch.Tensor,
            focal_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine all-in-focus image by taking sharpest regions from focal stack
        """
        B, N, C, H, W = focal_stack.shape

        # Compute local sharpness for each image
        sharpness_maps = []

        for n in range(N):
            img = focal_stack[:, n]

            # Use Laplacian for sharpness
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=img.dtype, device=img.device).view(1, 1, 3, 3).expand(C, 1, 3, 3)

            laplacian = F.conv2d(img, kernel, padding=1, groups=C)
            sharpness = laplacian.abs().mean(dim=1, keepdim=True)

            # Smooth sharpness map
            sharpness = F.avg_pool2d(sharpness, 3, stride=1, padding=1)
            sharpness_maps.append(sharpness)

        sharpness_maps = torch.stack(sharpness_maps, dim=1)

        # Find weights for each image based on relative sharpness
        sharpness_weights = F.softmax(sharpness_maps * 10, dim=1)  # Temperature scaling

        # Weighted combination of focal stack
        sharpness_weights = sharpness_weights.expand(-1, -1, C, -1, -1)
        refined_aif = (focal_stack * sharpness_weights).sum(dim=1)

        # Blend with original AIF
        blend_weight = 0.3
        refined_aif = blend_weight * all_in_focus + (1 - blend_weight) * refined_aif

        return refined_aif


class PSFGenerator(nn.Module):
    """
    Generate Point Spread Functions for defocus blur
    """

    def __init__(self, psf_size: int = 33):
        super().__init__()
        self.psf_size = psf_size

    def forward(self, coc_radius: torch.Tensor) -> torch.Tensor:
        """
        Generate PSF for given Circle of Confusion radius

        Args:
            coc_radius: CoC radius in pixels [B, 1, H, W]

        Returns:
            PSF kernels [B, H, W, psf_size, psf_size]
        """
        B, _, H, W = coc_radius.shape
        device = coc_radius.device

        # Create coordinate grid
        coords = torch.arange(self.psf_size, device=device, dtype=torch.float32)
        coords = coords - self.psf_size // 2
        y_coords, x_coords = torch.meshgrid(coords, coords, indexing='ij')
        radius_grid = torch.sqrt(x_coords ** 2 + y_coords ** 2).unsqueeze(0).unsqueeze(0)

        # Expand dimensions
        coc_expanded = coc_radius.unsqueeze(-1).unsqueeze(-1)
        radius_grid = radius_grid.unsqueeze(0).unsqueeze(0)

        # Create disk PSF
        psf = (radius_grid <= coc_expanded).float()

        # Normalize
        psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)

        return psf


class DepthRegularizer(nn.Module):
    """
    Regularize depth maps for smoothness and edge preservation
    """

    def __init__(self):
        super().__init__()

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute depth regularization loss
        """
        # Gradient loss
        grad_x = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
        grad_y = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

        # L1 smoothness
        smoothness = grad_x.mean() + grad_y.mean()

        return smoothness


class DifferentiableDefocusRenderer(nn.Module):
    """
    Differentiable defocus blur rendering
    """

    def __init__(self, max_coc_pixels: float = 50.0, num_depth_planes: int = 32):
        super().__init__()
        self.max_coc_pixels = max_coc_pixels
        self.num_depth_planes = num_depth_planes

    def forward(
            self,
            sharp_image: torch.Tensor,
            coc_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply spatially-varying defocus blur

        Args:
            sharp_image: All-in-focus image [B, C, H, W]
            coc_map: Circle of Confusion map [B, 1, H, W]

        Returns:
            Blurred image [B, C, H, W]
        """
        B, C, H, W = sharp_image.shape

        # Quantize CoC into depth planes for efficiency
        coc_planes = torch.linspace(0, self.max_coc_pixels, self.num_depth_planes, device=coc_map.device)

        blurred_layers = []
        weights = []

        for i in range(self.num_depth_planes):
            coc_value = coc_planes[i]

            if coc_value < 0.5:
                # No blur for very small CoC
                blurred = sharp_image
            else:
                # Apply Gaussian blur with sigma proportional to CoC
                sigma = coc_value / 2.355  # Convert CoC to Gaussian sigma
                kernel_size = int(2 * coc_value + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel_size = min(kernel_size, 31)  # Cap kernel size

                # Create Gaussian kernel
                coords = torch.arange(kernel_size, device=sharp_image.device, dtype=torch.float32)
                coords = coords - kernel_size // 2
                g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
                g = g / g.sum()
                kernel = g.unsqueeze(0) * g.unsqueeze(1)
                kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, kernel_size, kernel_size)

                # Apply convolution
                blurred = F.conv2d(sharp_image, kernel, padding=kernel_size // 2, groups=C)

            blurred_layers.append(blurred)

            # Compute weight based on distance to CoC value
            if i == 0:
                weight = (coc_map <= (coc_planes[0] + coc_planes[1]) / 2).float()
            elif i == self.num_depth_planes - 1:
                weight = (coc_map > (coc_planes[-2] + coc_planes[-1]) / 2).float()
            else:
                lower = (coc_planes[i - 1] + coc_planes[i]) / 2
                upper = (coc_planes[i] + coc_planes[i + 1]) / 2
                weight = ((coc_map > lower) & (coc_map <= upper)).float()

            weights.append(weight)

        # Stack and combine
        blurred_stack = torch.stack(blurred_layers, dim=0)
        weight_stack = torch.stack(weights, dim=0)

        # Weighted sum
        result = (blurred_stack * weight_stack.unsqueeze(2)).sum(dim=0)

        return result


__all__ = [
    "FocalCrossAttention",
    "PhysicsConsistencyModule",
    "PSFGenerator",
    "DepthRegularizer",
    "DifferentiableDefocusRenderer",
]
