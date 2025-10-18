"""
Image processing utilities for Focal-Depth Diffusion
"""

import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_image_stack(
        image_paths: List[Union[str, Path]],
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load a focal stack from image files

    Args:
        image_paths: List of paths to images
        resize: Optional (width, height) to resize images
        normalize: Whether to normalize to [0, 1]

    Returns:
        focal_stack: Array of shape [N, 3, H, W]
        original_size: (width, height) of original images
    """
    images = []
    original_size = None

    for img_path in image_paths:
        # Load image
        img = Image.open(img_path).convert('RGB')

        if original_size is None:
            original_size = img.size

        # Resize if requested
        if resize is not None:
            img = img.resize(resize, Image.Resampling.LANCZOS)

        # Convert to numpy
        img_np = np.array(img)

        # Normalize
        if normalize:
            img_np = img_np.astype(np.float32) / 255.0

        # Convert to CHW format
        img_np = img_np.transpose(2, 0, 1)
        images.append(img_np)

    # Stack images
    focal_stack = np.stack(images, axis=0)

    return focal_stack, original_size


def save_depth_map(
        depth: np.ndarray,
        save_path: Union[str, Path],
        colormap: str = 'viridis',
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
) -> None:
    """
    Save depth map as image

    Args:
        depth: Depth map array [H, W]
        save_path: Path to save image
        colormap: Matplotlib colormap name
        min_depth: Minimum depth for normalization
        max_depth: Maximum depth for normalization
    """
    # Normalize depth
    if min_depth is None:
        min_depth = depth.min()
    if max_depth is None:
        max_depth = depth.max()

    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(depth_norm)

    # Convert to uint8
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)

    # Save
    Image.fromarray(depth_colored).save(save_path)


def save_all_in_focus(
        image: Union[np.ndarray, torch.Tensor],
        save_path: Union[str, Path],
) -> None:
    """
    Save all-in-focus image

    Args:
        image: Image array [H, W, 3] or [3, H, W]
        save_path: Path to save image
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Handle different formats
    if image.shape[0] == 3 and len(image.shape) == 3:
        # CHW -> HWC
        image = image.transpose(1, 2, 0)

    # Ensure proper range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Save
    Image.fromarray(image).save(save_path)


def colorize_depth(
        depth: np.ndarray,
        colormap: str = 'spectral',
        min_percentile: float = 2.0,
        max_percentile: float = 98.0,
) -> Image.Image:
    """
    Colorize depth map for visualization

    Args:
        depth: Depth map [H, W]
        colormap: Matplotlib colormap
        min_percentile: Percentile for minimum depth
        max_percentile: Percentile for maximum depth

    Returns:
        Colored depth as PIL Image
    """
    # Robust normalization using percentiles
    min_depth = np.percentile(depth, min_percentile)
    max_depth = np.percentile(depth, max_percentile)

    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(depth_norm)

    # Convert to uint8
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(depth_colored)


def create_visualization(
        depth: np.ndarray,
        all_in_focus: Union[np.ndarray, Image.Image],
        uncertainty: Optional[np.ndarray] = None,
        focal_samples: Optional[List[np.ndarray]] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (15, 10),
) -> Optional[plt.Figure]:
    """
    Create a comprehensive visualization of results

    Args:
        depth: Depth map [H, W]
        all_in_focus: All-in-focus image
        uncertainty: Optional uncertainty map
        focal_samples: Optional sample images from focal stack
        save_path: Path to save visualization
        figsize: Figure size

    Returns:
        Figure object if save_path is None
    """
    # Convert inputs
    if isinstance(all_in_focus, Image.Image):
        all_in_focus = np.array(all_in_focus)

    # Determine layout
    n_cols = 2 + (1 if uncertainty is not None else 0)
    n_rows = 1 + (1 if focal_samples is not None else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # All-in-focus image
    axes[0, 0].imshow(all_in_focus)
    axes[0, 0].set_title('All-in-Focus Image')
    axes[0, 0].axis('off')

    # Depth map
    im_depth = axes[0, 1].imshow(depth, cmap='spectral')
    axes[0, 1].set_title('Depth Map')
    axes[0, 1].axis('off')
    plt.colorbar(im_depth, ax=axes[0, 1], fraction=0.046)

    # Uncertainty map
    if uncertainty is not None:
        im_unc = axes[0, 2].imshow(uncertainty, cmap='hot')
        axes[0, 2].set_title('Uncertainty')
        axes[0, 2].axis('off')
        plt.colorbar(im_unc, ax=axes[0, 2], fraction=0.046)

    # Focal samples
    if focal_samples is not None and n_rows > 1:
        n_samples = min(len(focal_samples), n_cols)
        for i in range(n_samples):
            axes[1, i].imshow(focal_samples[i])
            axes[1, i].set_title(f'Focus {i + 1}')
            axes[1, i].axis('off')

        # Hide unused axes
        for i in range(n_samples, n_cols):
            axes[1, i].axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def normalize_depth(
        depth: np.ndarray,
        method: str = 'min_max',
        percentile: float = 98.0,
) -> np.ndarray:
    """
    Normalize depth map

    Args:
        depth: Depth map
        method: Normalization method ('min_max', 'percentile', 'standardize')
        percentile: Percentile for percentile normalization

    Returns:
        Normalized depth
    """
    if method == 'min_max':
        min_d = depth.min()
        max_d = depth.max()
        return (depth - min_d) / (max_d - min_d + 1e-8)

    elif method == 'percentile':
        min_d = np.percentile(depth, 100 - percentile)
        max_d = np.percentile(depth, percentile)
        depth_norm = (depth - min_d) / (max_d - min_d + 1e-8)
        return np.clip(depth_norm, 0, 1)

    elif method == 'standardize':
        mean = depth.mean()
        std = depth.std() + 1e-8
        return (depth - mean) / std

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def visualize_focal_stack(
        focal_stack: Union[np.ndarray, torch.Tensor],
        focus_distances: Optional[List[float]] = None,
        max_images: int = 9,
        save_path: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """
    Visualize a focal stack as a grid

    Args:
        focal_stack: Focal stack [N, 3, H, W] or [N, H, W, 3]
        focus_distances: Optional focus distances for labels
        max_images: Maximum number of images to show
        save_path: Path to save visualization

    Returns:
        Figure object if save_path is None
    """
    if isinstance(focal_stack, torch.Tensor):
        focal_stack = focal_stack.cpu().numpy()

    # Handle different formats
    if focal_stack.shape[1] == 3:
        # [N, 3, H, W] -> [N, H, W, 3]
        focal_stack = focal_stack.transpose(0, 2, 3, 1)

    n_images = min(len(focal_stack), max_images)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]

    for i in range(n_images):
        img = focal_stack[i]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)

        axes[i].imshow(img)

        if focus_distances is not None:
            axes[i].set_title(f'Focus: {focus_distances[i]:.2f}m')
        else:
            axes[i].set_title(f'Image {i + 1}')

        axes[i].axis('off')

    # Hide unused axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def apply_depth_blur(
        image: np.ndarray,
        depth: np.ndarray,
        focus_distance: float,
        aperture: float,
        focal_length: float,
        max_blur: float = 20.0,
) -> np.ndarray:
    """
    Apply depth-based blur to simulate defocus

    Args:
        image: Sharp image [H, W, 3]
        depth: Depth map [H, W] in meters
        focus_distance: Focus distance in meters
        aperture: Aperture f-number
        focal_length: Focal length in meters
        max_blur: Maximum blur radius in pixels

    Returns:
        Blurred image
    """
    # Compute Circle of Confusion
    coc = compute_coc_map(depth, focus_distance, aperture, focal_length)

    # Convert to blur radius in pixels
    # This is a simplification - proper implementation would consider sensor size
    blur_radius = np.abs(coc) * 1000 * max_blur  # Scale factor is arbitrary
    blur_radius = np.clip(blur_radius, 0, max_blur)

    # Apply spatially-varying blur
    # This is a simplified implementation - proper one would use integral images
    blurred = image.copy()

    # Discretize blur levels
    n_levels = 10
    for level in range(n_levels):
        blur_min = level * max_blur / n_levels
        blur_max = (level + 1) * max_blur / n_levels

        # Create mask for this blur level
        mask = (blur_radius >= blur_min) & (blur_radius < blur_max)

        if mask.any():
            # Apply Gaussian blur
            kernel_size = int(2 * blur_max + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1

            blurred_level = cv2.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                blur_max / 3  # Sigma = radius / 3
            )

            # Blend
            mask_3d = mask[:, :, np.newaxis]
            blurred = blurred * (1 - mask_3d) + blurred_level * mask_3d

    return blurred.astype(np.uint8)


def compute_coc_map(
        depth: np.ndarray,
        focus_distance: float,
        aperture: float,
        focal_length: float,
) -> np.ndarray:
    """
    Compute Circle of Confusion map

    Args:
        depth: Depth map in meters
        focus_distance: Focus distance in meters
        aperture: Aperture f-number
        focal_length: Focal length in meters

    Returns:
        CoC map
    """
    # Compute image distances
    image_distance = (focal_length * depth) / (depth - focal_length)
    focus_image_distance = (focal_length * focus_distance) / (focus_distance - focal_length)

    # Entrance pupil diameter
    pupil_diameter = focal_length / aperture

    # Circle of Confusion
    coc = np.abs(image_distance - focus_image_distance) * pupil_diameter / focus_image_distance

    return coc