"""
Visualization utilities for FocalDiffusion
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Optional, Union
from pathlib import Path


def visualize_results(
        outputs: Dict[str, Union[torch.Tensor, np.ndarray, Image.Image]],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (20, 10),
) -> Optional[plt.Figure]:
    """
    Visualize FocalDiffusion outputs

    Args:
        outputs: Dictionary containing results
        save_path: Path to save figure
        show: Whether to display figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()

    # Helper function to convert to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, Image.Image):
            return np.array(x)
        return x

    # Visualize focal stack samples
    if 'focal_stack' in outputs:
        focal_stack = to_numpy(outputs['focal_stack'])
        if focal_stack.ndim == 4:  # [N, H, W, C]
            n_samples = min(3, focal_stack.shape[0])
            for i in range(n_samples):
                axes[i].imshow(focal_stack[i])
                axes[i].set_title(f'Focus {i + 1}')
                axes[i].axis('off')

    # All-in-focus image
    if 'all_in_focus' in outputs:
        aif = to_numpy(outputs['all_in_focus'])
        if aif.ndim == 3 and aif.shape[0] == 3:  # CHW to HWC
            aif = np.transpose(aif, (1, 2, 0))
        if aif.max() <= 1.0:
            aif = (aif * 255).astype(np.uint8)
        axes[3].imshow(aif)
        axes[3].set_title('All-in-Focus')
        axes[3].axis('off')

    # Depth map
    if 'depth' in outputs:
        depth = to_numpy(outputs['depth'])
        if depth.ndim == 3:
            depth = depth.squeeze()
        im = axes[4].imshow(depth, cmap='spectral')
        axes[4].set_title('Depth Map')
        axes[4].axis('off')
        plt.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    # Colored depth
    if 'depth_colored' in outputs:
        depth_colored = to_numpy(outputs['depth_colored'])
        axes[5].imshow(depth_colored)
        axes[5].set_title('Depth (Colored)')
        axes[5].axis('off')

    # Uncertainty
    if 'uncertainty' in outputs:
        uncertainty = to_numpy(outputs['uncertainty'])
        if uncertainty.ndim == 3:
            uncertainty = uncertainty.squeeze()
        im = axes[6].imshow(uncertainty, cmap='hot')
        axes[6].set_title('Uncertainty')
        axes[6].axis('off')
        plt.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)

    # Attention weights
    if 'attention_weights' in outputs:
        weights = to_numpy(outputs['attention_weights'])
        if weights.ndim == 1:
            axes[7].bar(range(len(weights)), weights)
            axes[7].set_title('Focus Attention Weights')
            axes[7].set_xlabel('Focus Plane')
            axes[7].set_ylabel('Weight')
        else:
            axes[7].imshow(weights)
            axes[7].set_title('Attention Map')
            axes[7].axis('off')

    # Hide unused axes
    for i in range(len(outputs), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()
        return fig

    return None