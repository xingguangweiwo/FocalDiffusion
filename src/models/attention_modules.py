"""Compatibility shim for historical imports.

Active modules have been split into:
- :mod:`src.models.focal_attention`
- :mod:`src.models.physics_modules`
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
