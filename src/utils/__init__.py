"""FocalDiffusion utilities."""

from .image_utils import (
    load_image_stack,
    save_depth_map,
    save_all_in_focus,
    colorize_depth,
    create_visualization,
)
from .camera_utils import (
    parse_exif_data,
    estimate_focus_distances,
    compute_circle_of_confusion,
    get_depth_of_field,
)
from .metrics import compute_metrics
from .visualization import visualize_results


def ensure_sentencepiece_installed() -> None:
    """Legacy no-op kept for backwards compatibility.

    Older entry points imported :func:`ensure_sentencepiece_installed` to
    perform an optional dependency check. The validation step has since been
    inlined elsewhere, but keeping this stub avoids ``NameError`` crashes for
    users running out-of-date scripts or notebooks that still reference the
    helper.
    """


__all__ = [
    "load_image_stack",
    "save_depth_map",
    "save_all_in_focus",
    "colorize_depth",
    "create_visualization",
    "parse_exif_data",
    "estimate_focus_distances",
    "compute_circle_of_confusion",
    "get_depth_of_field",
    "compute_metrics",
    "visualize_results",
    "ensure_sentencepiece_installed",
]
