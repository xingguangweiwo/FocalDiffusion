"""FocalDiffusion utilities"""

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
    """Ensure the optional sentencepiece dependency is available.

    Stable Diffusion 3.5 relies on a T5 text encoder whose fast tokenizer
    delegates to the :mod:`sentencepiece` Python package.  When the package is
    missing, Hugging Face will download the (multi-gigabyte) checkpoint only to
    raise a ``ValueError`` while instantiating the tokenizer.  Import the module
    up front so we can fail fast with a clear remediation hint.
    """

    try:  # pragma: no cover - import guard depends on runtime environment
        import sentencepiece  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The `sentencepiece` package is required to load Stable Diffusion "
            "3.5 tokenizers. Install it with `pip install sentencepiece` or "
            "`conda install -c conda-forge sentencepiece` and re-run the "
            "command."
        ) from exc


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
