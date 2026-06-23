"""
Image processing utilities for Focal-Depth Diffusion
"""

import torch
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _get_pyplot() -> Any:
    """Import matplotlib lazily for optional visualization utilities."""
    import matplotlib.pyplot as plt

    return plt


def to_unit_range(image: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor in ``[0, 1]`` or ``[-1, 1]`` to unit range.

    The conversion is explicit and finite-checked so neural/model paths and
    photometric consistency paths share a single range convention.
    """
    if not torch.isfinite(image).all():
        raise ValueError("image tensors must contain only finite values.")
    min_value = image.amin()
    max_value = image.amax()
    if min_value >= 0.0 and max_value <= 1.0:
        return image
    if min_value >= -1.0 and max_value <= 1.0:
        return (image + 1.0) * 0.5
    raise ValueError("image tensors must be in [0, 1] or [-1, 1] range.")


def to_model_range(image: torch.Tensor) -> torch.Tensor:
    """Convert an image tensor in ``[0, 1]`` or ``[-1, 1]`` to model ``[-1, 1]`` range."""
    unit = to_unit_range(image)
    return unit * 2.0 - 1.0


def canonical_focal_coordinates(
        focal_plane_distances: torch.Tensor,
        *,
        batch_size: Optional[int] = None,
        coordinate_type: str = "normalized_rank",
        metric_coordinates: Optional[torch.Tensor] = None,
        focal_plane_valid_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return canonical focal coordinates and a validity mask.

    Supported conventions are ``distance``/``z_position`` (min-max metric
    distance), ``inverse_distance``/``diopter`` (min-max inverse metres),
    ``index`` (0..K-1), and ``normalized_rank`` (0..1 plane rank). Supplying
    ``metric_coordinates`` overrides the coordinate values before canonical
    normalization while retaining shape/batch validation.
    """
    focus = torch.as_tensor(focal_plane_distances)
    if focus.dim() == 1:
        focus = focus.unsqueeze(0)
    if focus.dim() != 2:
        raise ValueError(f"focal_plane_distances must have shape [N] or [B,N], got {tuple(focal_plane_distances.shape)}")
    if batch_size is not None and focus.shape[0] == 1 and batch_size != 1:
        focus = focus.expand(batch_size, -1)
    if batch_size is not None and focus.shape[0] != batch_size:
        raise ValueError(f"focal_plane_distances batch {focus.shape[0]} does not match expected batch {batch_size}.")
    if not torch.isfinite(focus).all():
        raise ValueError("focal_plane_distances must contain only finite values.")
    batch, planes = focus.shape
    if planes < 1:
        raise ValueError("at least one focal plane is required.")

    if focal_plane_valid_mask is None:
        valid = torch.ones((batch, planes), device=focus.device, dtype=torch.bool)
        if planes > 1:
            valid[:, 1:] = (focus[:, 1:] - focus[:, :-1]).abs() > eps
    else:
        valid = focal_plane_valid_mask.to(device=focus.device, dtype=torch.bool)
        if valid.dim() == 1:
            valid = valid.unsqueeze(0).expand(batch, -1)
        if valid.shape != (batch, planes):
            raise ValueError(f"focal_plane_valid_mask must have shape [N] or [B,N], got {tuple(focal_plane_valid_mask.shape)}")
    if not valid.any(dim=1).all():
        raise ValueError("Each sample must contain at least one valid focal plane.")

    coord_type = coordinate_type.lower().replace("-", "_")
    if metric_coordinates is not None:
        coords = metric_coordinates.to(device=focus.device, dtype=focus.dtype)
        if coords.dim() == 1:
            coords = coords.unsqueeze(0).expand(batch, -1)
        if coords.shape != (batch, planes):
            raise ValueError("metric_coordinates must have shape [N] or [B,N] matching focal planes.")
    elif coord_type in {"distance", "z_position", "z"}:
        coords = focus
    elif coord_type in {"inverse_distance", "diopter", "diopters"}:
        coords = 1.0 / focus.clamp(min=eps)
    elif coord_type == "index":
        coords = torch.arange(planes, device=focus.device, dtype=focus.dtype).view(1, planes).expand(batch, -1)
    elif coord_type in {"normalized_rank", "rank", "canonical"}:
        if planes == 1:
            coords = torch.zeros((batch, planes), device=focus.device, dtype=focus.dtype)
        else:
            coords = torch.linspace(0.0, 1.0, planes, device=focus.device, dtype=focus.dtype).view(1, planes).expand(batch, -1)
        return coords, valid
    else:
        raise ValueError(f"Unsupported focal coordinate type: {coordinate_type}")

    masked = coords.masked_fill(~valid, float("nan"))
    mn = torch.nan_to_num(masked, nan=float("inf")).amin(dim=1, keepdim=True)
    mx = torch.nan_to_num(masked, nan=float("-inf")).amax(dim=1, keepdim=True)
    if torch.any((mx - mn).abs() <= eps):
        if planes == 1:
            return torch.zeros_like(coords), valid
        raise ValueError("valid focal coordinates must span a non-zero range.")
    return (coords - mn) / (mx - mn).clamp(min=eps), valid



def metric_depth_to_canonical_depth(
        metric_depth: torch.Tensor,
        focus_near: torch.Tensor | float,
        focus_far: torch.Tensor | float,
        *,
        coordinate_type: str = "distance",
        boundary_eps: float = 1e-3,
        eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map metric depth into the canonical focal-sweep coordinate.

    The canonical coordinate is ``normalize(g(depth), g(focus_near),
    g(focus_far))`` where ``g`` is metric distance, inverse distance
    (diopter), or normalized rank. The returned masks identify pixels inside
    the represented sweep and close to each sweep boundary.
    """
    depth = torch.as_tensor(metric_depth)
    near = torch.as_tensor(focus_near, device=depth.device, dtype=depth.dtype)
    far = torch.as_tensor(focus_far, device=depth.device, dtype=depth.dtype)
    while near.dim() < depth.dim():
        near = near.unsqueeze(-1)
    while far.dim() < depth.dim():
        far = far.unsqueeze(-1)

    coord_type = coordinate_type.lower().replace("-", "_")
    if coord_type in {"distance", "z_position", "z"}:
        value, lo, hi = depth, near, far
    elif coord_type in {"inverse_distance", "diopter", "diopters"}:
        value = 1.0 / depth.clamp(min=eps)
        lo = 1.0 / far.clamp(min=eps)
        hi = 1.0 / near.clamp(min=eps)
    elif coord_type in {"normalized_rank", "rank", "canonical"}:
        value, lo, hi = depth, near, far
    else:
        raise ValueError(f"Unsupported canonical depth coordinate type: {coordinate_type}")

    lower = torch.minimum(lo, hi)
    upper = torch.maximum(lo, hi)
    span = (upper - lower).clamp(min=eps)
    canonical = ((value - lower) / span).clamp(0.0, 1.0)
    within = (value >= lower) & (value <= upper) & torch.isfinite(value)
    near_boundary = within & (((value - lower).abs() / span) <= boundary_eps)
    far_boundary = within & (((upper - value).abs() / span) <= boundary_eps)
    return canonical, within, near_boundary, far_boundary

def resize_probability_volume(
        posterior: torch.Tensor,
        size: Tuple[int, int],
        focal_plane_valid_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    """Resize a focal probability volume, apply plane masks, and renormalize."""
    if posterior.dim() != 4:
        raise ValueError(f"posterior must have shape [B,N,H,W], got {tuple(posterior.shape)}")
    if not torch.isfinite(posterior).all():
        raise ValueError("posterior must contain only finite values.")
    if posterior.shape[-2:] != size:
        posterior = torch.nn.functional.interpolate(posterior, size=size, mode="bilinear", align_corners=False)
    if focal_plane_valid_mask is not None:
        valid = focal_plane_valid_mask.to(device=posterior.device, dtype=torch.bool)
        if valid.dim() == 1:
            valid = valid.unsqueeze(0).expand(posterior.shape[0], -1)
        if valid.shape != posterior.shape[:2]:
            raise ValueError(f"focal_plane_valid_mask must have shape [N] or [B,N], got {tuple(focal_plane_valid_mask.shape)}")
        if not valid.any(dim=1).all():
            raise ValueError("Cannot normalize posterior with an all-invalid focal-plane sample.")
        posterior = posterior * valid[:, :, None, None].to(dtype=posterior.dtype)
    denom = posterior.sum(dim=1, keepdim=True)
    if torch.any(denom <= eps):
        raise ValueError("Cannot normalize posterior with zero probability mass.")
    return posterior / denom.clamp(min=eps)


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
    plt = _get_pyplot()
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
    plt = _get_pyplot()
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
) -> Optional[Any]:
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
    plt = _get_pyplot()

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
        focal_plane_distances: Optional[List[float]] = None,
        max_images: int = 9,
        save_path: Optional[Union[str, Path]] = None,
) -> Optional[Any]:
    """
    Visualize a focal stack as a grid

    Args:
        focal_stack: Focal stack [N, 3, H, W] or [N, H, W, 3]
        focal_plane_distances: Optional focus distances for labels
        max_images: Maximum number of images to show
        save_path: Path to save visualization

    Returns:
        Figure object if save_path is None
    """
    plt = _get_pyplot()

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

        if focal_plane_distances is not None:
            axes[i].set_title(f'Focus: {focal_plane_distances[i]:.2f}m')
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




def visualize_results(
        outputs: Dict[str, Union[np.ndarray, torch.Tensor, Image.Image]],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (20, 10),
) -> Optional[Any]:
    """Visualize common focal-stack outputs in a compact diagnostic grid."""

    plt = _get_pyplot()

    def to_numpy(value: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, Image.Image):
            return np.array(value)
        return value

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    flat_axes = axes.flatten()

    if "focal_stack" in outputs:
        focal_stack = to_numpy(outputs["focal_stack"])
        if focal_stack.ndim == 4:
            if focal_stack.shape[1] == 3:
                focal_stack = focal_stack.transpose(0, 2, 3, 1)
            for index in range(min(3, focal_stack.shape[0])):
                flat_axes[index].imshow(focal_stack[index])
                flat_axes[index].set_title(f"Focus {index + 1}")
                flat_axes[index].axis("off")

    if "all_in_focus" in outputs:
        all_in_focus = to_numpy(outputs["all_in_focus"])
        if all_in_focus.ndim == 3 and all_in_focus.shape[0] == 3:
            all_in_focus = all_in_focus.transpose(1, 2, 0)
        if all_in_focus.max() <= 1.0:
            all_in_focus = (all_in_focus * 255).astype(np.uint8)
        flat_axes[3].imshow(all_in_focus)
        flat_axes[3].set_title("All-in-Focus")
        flat_axes[3].axis("off")

    if "depth" in outputs:
        depth = np.squeeze(to_numpy(outputs["depth"]))
        im = flat_axes[4].imshow(depth, cmap="spectral")
        flat_axes[4].set_title("Depth Map")
        flat_axes[4].axis("off")
        plt.colorbar(im, ax=flat_axes[4], fraction=0.046, pad=0.04)

    if "depth_colored" in outputs:
        flat_axes[5].imshow(to_numpy(outputs["depth_colored"]))
        flat_axes[5].set_title("Depth (Colored)")
        flat_axes[5].axis("off")

    if "uncertainty" in outputs:
        uncertainty = np.squeeze(to_numpy(outputs["uncertainty"]))
        im = flat_axes[6].imshow(uncertainty, cmap="hot")
        flat_axes[6].set_title("Uncertainty")
        flat_axes[6].axis("off")
        plt.colorbar(im, ax=flat_axes[6], fraction=0.046, pad=0.04)

    if "attention_weights" in outputs:
        weights = to_numpy(outputs["attention_weights"])
        if weights.ndim == 1:
            flat_axes[7].bar(range(len(weights)), weights)
            flat_axes[7].set_title("Focus Attention Weights")
            flat_axes[7].set_xlabel("Focus Plane")
            flat_axes[7].set_ylabel("Weight")
        else:
            flat_axes[7].imshow(weights)
            flat_axes[7].set_title("Attention Map")
            flat_axes[7].axis("off")

    for axis in flat_axes:
        if not axis.has_data():
            axis.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
        return None
    plt.close()
    return fig
