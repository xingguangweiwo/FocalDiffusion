"""Backward-compatible re-exports for legacy imports.

Use `focal_attention.py` for attention-specific modules and `physics_modules.py`
for optics/physics helpers.
"""

from .focal_attention import FocalCrossAttention
from .physics_modules import (
    DepthRegularizer,
    DifferentiableDefocusRenderer,
    PSFGenerator,
    PhysicsConsistencyModule,
)

__all__ = [
    "FocalCrossAttention",
    "PhysicsConsistencyModule",
    "PSFGenerator",
    "DepthRegularizer",
    "DifferentiableDefocusRenderer",
]
