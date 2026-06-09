"""Backward-compatible re-exports for legacy imports.

Use `focal_attention.py` for attention-specific modules and `defocus_physics.py`
for optics/physics helpers.
"""

from .focal_attention import FocalCrossAttention
from .defocus_physics import (
    DepthRegularizer,
    ThinLensDefocusRenderer,
    PSFGenerator,
    DefocusConsistencyModule,
)

__all__ = [
    "FocalCrossAttention",
    "DefocusConsistencyModule",
    "PSFGenerator",
    "DepthRegularizer",
    "ThinLensDefocusRenderer",
]
