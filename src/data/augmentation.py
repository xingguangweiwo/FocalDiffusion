"""
Data augmentation for focal stacks
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import cv2


class FocalAugmentation:
    """Augmentation pipeline for focal stacks"""

    def __init__(
            self,
            random_flip_prob: float = 0.5,
            random_crop_prob: float = 0.3,
            brightness_range: Tuple[float, float] = (0.8, 1.2),
            contrast_range: Tuple[float, float] = (0.8, 1.2),
            noise_std: float = 0.01,
            blur_prob: float = 0.1,
    ):
        self.random_flip_prob = random_flip_prob
        self.random_crop_prob = random_crop_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob

    def __call__(
            self,
            focal_stack: torch.Tensor,
            depth: torch.Tensor,
            all_in_focus: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentations to focal stack and depth

        Args:
            focal_stack: [N, C, H, W]
            depth: [1, H, W]
            all_in_focus: Optional [C, H, W]
        """

        # Random horizontal flip
        if np.random.rand() < self.random_flip_prob:
            focal_stack = torch.flip(focal_stack, dims=[-1])
            depth = torch.flip(depth, dims=[-1])
            if all_in_focus is not None:
                all_in_focus = torch.flip(all_in_focus, dims=[-1])

        # Random crop and resize
        if np.random.rand() < self.random_crop_prob:
            focal_stack, depth, all_in_focus = self._random_crop_resize(
                focal_stack, depth, all_in_focus
            )

        # Color augmentation (only on RGB)
        if np.random.rand() < 0.5:
            focal_stack = self._color_augmentation(focal_stack)
            if all_in_focus is not None:
                all_in_focus = self._color_augmentation(all_in_focus.unsqueeze(0)).squeeze(0)

        # Add noise
        if self.noise_std > 0 and np.random.rand() < 0.3:
            noise = torch.randn_like(focal_stack) * self.noise_std
            focal_stack = torch.clamp(focal_stack + noise, -1, 1)

        return focal_stack, depth, all_in_focus

    def _random_crop_resize(
            self,
            focal_stack: torch.Tensor,
            depth: torch.Tensor,
            all_in_focus: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Random crop and resize"""
        N, C, H, W = focal_stack.shape

        # Random crop size
        crop_scale = np.random.uniform(0.8, 1.0)
        crop_h = int(H * crop_scale)
        crop_w = int(W * crop_scale)

        # Random position
        y = np.random.randint(0, H - crop_h + 1)
        x = np.random.randint(0, W - crop_w + 1)

        # Crop
        focal_stack = focal_stack[:, :, y:y + crop_h, x:x + crop_w]
        depth = depth[:, y:y + crop_h, x:x + crop_w]
        if all_in_focus is not None:
            all_in_focus = all_in_focus[:, y:y + crop_h, x:x + crop_w]

        # Resize back
        focal_stack = F.interpolate(focal_stack, size=(H, W), mode='bilinear', align_corners=False)
        depth = F.interpolate(depth.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
        if all_in_focus is not None:
            all_in_focus = F.interpolate(
                all_in_focus.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0)

        return focal_stack, depth, all_in_focus

    def _color_augmentation(self, images: torch.Tensor) -> torch.Tensor:
        """Apply color augmentation"""
        # Random brightness
        brightness = np.random.uniform(*self.brightness_range)
        images = images * brightness

        # Random contrast
        contrast = np.random.uniform(*self.contrast_range)
        mean = images.mean(dim=[2, 3], keepdim=True)
        images = (images - mean) * contrast + mean

        return torch.clamp(images, -1, 1)
