"""Canonical heterogeneous focal-coordinate runtime protocol."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping, Any, Iterator

import torch


class FocalCoordinateType(IntEnum):
    camera_object_distance = 0
    camera_inverse_distance = 1
    camera_diopter = 2
    lens_motor_position = 3
    lens_encoder_position = 4
    microscope_stage_z = 5
    microscope_objective_z = 6
    microscope_relative_z = 7
    focus_index = 8
    normalized_rank = 9
    coc_radius = 10
    blur_sigma = 11
    unknown_ordered = 12
    unknown_unordered = 13


class FocalCoordinateUnit(IntEnum):
    unknown = 0
    meter = 1
    centimeter = 2
    millimeter = 3
    micrometer = 4
    diopter = 5
    step = 6
    index = 7
    normalized = 8
    pixel = 9


class FocalAxisDirection(IntEnum):
    increasing_near_to_far = 0
    increasing_far_to_near = 1
    unknown = 2


@dataclass
class FocalCoordinateBundle(Mapping[str, torch.Tensor]):
    raw_values: torch.Tensor
    canonical_rank: torch.Tensor
    canonical_coordinate: torch.Tensor
    physical_values: torch.Tensor
    physical_valid_mask: torch.Tensor
    coordinate_type_id: torch.Tensor
    unit_id: torch.Tensor
    axis_direction_id: torch.Tensor
    spacing_confidence: torch.Tensor
    calibration_available: torch.Tensor

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter((
            "raw_values", "canonical_rank", "canonical_coordinate", "physical_values",
            "physical_valid_mask", "coordinate_type_id", "unit_id", "axis_direction_id",
            "spacing_confidence", "calibration_available",
        ))

    def __len__(self) -> int:
        return 10


def _enum_value(enum_cls, value):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        key = value.lower().replace("-", "_")
        aliases = {"m": "meter", "cm": "centimeter", "mm": "millimeter", "um": "micrometer", "µm": "micrometer"}
        key = aliases.get(key, key)
        return enum_cls[key]
    return enum_cls(int(value))


def _as_tensor(values, *, dtype=torch.float32):
    return torch.as_tensor(values, dtype=dtype).flatten()


def _normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    mn, mx = values.min(), values.max()
    return (values - mn) / (mx - mn).clamp(min=1e-6)


def _rank(n: int, direction: FocalAxisDirection, device=None) -> torch.Tensor:
    r = torch.linspace(0.0, 1.0, steps=max(n, 1), device=device)
    if direction == FocalAxisDirection.increasing_far_to_near:
        r = 1.0 - r
    return r


def _length_to(values: torch.Tensor, unit: FocalCoordinateUnit, target: FocalCoordinateUnit) -> torch.Tensor:
    factors = {
        FocalCoordinateUnit.meter: 1.0,
        FocalCoordinateUnit.centimeter: 1e-2,
        FocalCoordinateUnit.millimeter: 1e-3,
        FocalCoordinateUnit.micrometer: 1e-6,
    }
    if unit not in factors:
        raise ValueError(f"unit {unit.name} is not a length unit")
    meters = values * factors[unit]
    if target == FocalCoordinateUnit.meter:
        return meters
    if target == FocalCoordinateUnit.micrometer:
        return meters / 1e-6
    raise ValueError(f"unsupported target unit {target.name}")


def canonicalize_focal_coordinates(
    raw_values: Iterable[float] | torch.Tensor,
    coordinate_type: str | FocalCoordinateType = FocalCoordinateType.unknown_ordered,
    unit: str | FocalCoordinateUnit = FocalCoordinateUnit.unknown,
    axis_direction: str | FocalAxisDirection = FocalAxisDirection.increasing_near_to_far,
    *,
    allow_unknown_ordered: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> FocalCoordinateBundle:
    """Convert heterogeneous focal metadata into a strict zero-shot canonical bundle."""
    del metadata
    ctype = _enum_value(FocalCoordinateType, coordinate_type)
    u = _enum_value(FocalCoordinateUnit, unit)
    axis = _enum_value(FocalAxisDirection, axis_direction)
    raw = _as_tensor(raw_values)
    if raw.numel() == 0:
        raise ValueError("raw_values must contain at least one focal plane")
    if axis == FocalAxisDirection.unknown:
        raise ValueError("axis_direction must be explicit for canonical ranks")
    if not torch.isfinite(raw).all():
        raise ValueError("raw_values must be finite")

    n = raw.numel()
    rank = _rank(n, axis, raw.device)
    physical = torch.zeros_like(raw)
    physical_mask = torch.zeros_like(raw, dtype=torch.bool)
    canonical = rank.clone()
    confidence = torch.tensor(1.0, dtype=torch.float32)
    calibrated = torch.tensor(False)

    camera_distance_types = {FocalCoordinateType.camera_object_distance, FocalCoordinateType.camera_inverse_distance, FocalCoordinateType.camera_diopter}
    microscope_types = {FocalCoordinateType.microscope_stage_z, FocalCoordinateType.microscope_objective_z, FocalCoordinateType.microscope_relative_z}
    motor_types = {FocalCoordinateType.lens_motor_position, FocalCoordinateType.lens_encoder_position, FocalCoordinateType.focus_index, FocalCoordinateType.normalized_rank, FocalCoordinateType.coc_radius, FocalCoordinateType.blur_sigma}

    if ctype == FocalCoordinateType.unknown_unordered:
        canonical = torch.zeros_like(raw)
        confidence = torch.tensor(0.0)
    elif ctype == FocalCoordinateType.unknown_ordered:
        if not allow_unknown_ordered:
            raise ValueError("unknown_ordered coordinates require allow_unknown_ordered=True")
        confidence = torch.tensor(0.25)
    elif ctype in camera_distance_types:
        if ctype == FocalCoordinateType.camera_diopter or ctype == FocalCoordinateType.camera_inverse_distance:
            if u not in {FocalCoordinateUnit.diopter, FocalCoordinateUnit.unknown}:
                raise ValueError("camera_diopter/inverse_distance coordinates require diopter or unknown unit")
            diopters = raw
            meters = 1.0 / raw.clamp(min=1e-6)
        else:
            if u == FocalCoordinateUnit.unknown:
                raise ValueError("camera_object_distance requires an explicit length unit")
            meters = _length_to(raw, u, FocalCoordinateUnit.meter)
            diopters = 1.0 / meters.clamp(min=1e-6)
        physical = meters
        physical_mask[:] = True
        canonical = _normalize(diopters)  # camera distance canonicalizes in diopter space.
        calibrated = torch.tensor(True)
    elif ctype in microscope_types:
        if u == FocalCoordinateUnit.unknown:
            raise ValueError("microscope z coordinates require an explicit length unit")
        physical = _length_to(raw, u, FocalCoordinateUnit.micrometer)
        physical_mask[:] = True
        canonical = _normalize(physical)
        calibrated = torch.tensor(True)
    elif ctype in motor_types:
        valid = {
            FocalCoordinateType.focus_index: {FocalCoordinateUnit.index, FocalCoordinateUnit.unknown},
            FocalCoordinateType.normalized_rank: {FocalCoordinateUnit.normalized, FocalCoordinateUnit.unknown},
            FocalCoordinateType.coc_radius: {FocalCoordinateUnit.pixel, FocalCoordinateUnit.unknown},
            FocalCoordinateType.blur_sigma: {FocalCoordinateUnit.pixel, FocalCoordinateUnit.unknown},
        }.get(ctype, {FocalCoordinateUnit.step, FocalCoordinateUnit.index, FocalCoordinateUnit.unknown})
        if u not in valid:
            raise ValueError(f"unit {u.name} is incompatible with {ctype.name}")
        canonical = _normalize(raw) if ctype != FocalCoordinateType.normalized_rank else raw.clamp(0.0, 1.0)
    else:
        raise ValueError(f"unsupported focal coordinate type {ctype.name}")

    if axis == FocalAxisDirection.increasing_far_to_near and ctype != FocalCoordinateType.unknown_ordered:
        canonical = 1.0 - canonical
    return FocalCoordinateBundle(raw, rank, canonical.float(), physical.float(), physical_mask, torch.tensor(int(ctype)), torch.tensor(int(u)), torch.tensor(int(axis)), confidence, calibrated)
