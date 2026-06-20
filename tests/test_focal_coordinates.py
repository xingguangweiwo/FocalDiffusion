import pytest
import torch

from src.data.focal_coordinates import canonicalize_focal_coordinates, FocalCoordinateType
from src.models.focal_evidence_encoder import FocalEvidenceEncoder
from src.models.physics_modules import FocalPhysicalVerifier


def test_m_cm_mm_conversion_consistent():
    m = canonicalize_focal_coordinates([1, 2], 'camera_object_distance', 'meter', 'increasing_near_to_far')
    cm = canonicalize_focal_coordinates([100, 200], 'camera_object_distance', 'centimeter', 'increasing_near_to_far')
    mm = canonicalize_focal_coordinates([1000, 2000], 'camera_object_distance', 'millimeter', 'increasing_near_to_far')
    assert torch.allclose(m.physical_values, cm.physical_values)
    assert torch.allclose(m.physical_values, mm.physical_values)
    assert torch.allclose(m.canonical_coordinate, cm.canonical_coordinate)


def test_object_distance_and_diopter_mapping_consistent():
    obj = canonicalize_focal_coordinates([1.0, 0.5], 'camera_object_distance', 'meter', 'increasing_near_to_far')
    dio = canonicalize_focal_coordinates([1.0, 2.0], 'camera_diopter', 'diopter', 'increasing_near_to_far')
    assert torch.allclose(obj.canonical_coordinate, dio.canonical_coordinate)
    assert torch.allclose(obj.physical_values, dio.physical_values)


def test_reverse_axis_flips_rank_and_coordinate():
    fwd = canonicalize_focal_coordinates([1, 2, 3], 'focus_index', 'index', 'increasing_near_to_far')
    rev = canonicalize_focal_coordinates([1, 2, 3], 'focus_index', 'index', 'increasing_far_to_near')
    assert torch.allclose(rev.canonical_rank, 1 - fwd.canonical_rank)
    assert torch.allclose(rev.canonical_coordinate, 1 - fwd.canonical_coordinate)


def test_microscope_z_stays_micrometers_not_object_distance():
    z = canonicalize_focal_coordinates([0.0, 10.0], 'microscope_stage_z', 'micrometer', 'increasing_near_to_far')
    assert torch.allclose(z.physical_values, torch.tensor([0.0, 10.0]))
    assert z.coordinate_type_id.item() == FocalCoordinateType.microscope_stage_z


def test_motor_index_only_canonical_no_physical():
    b = canonicalize_focal_coordinates([10, 20, 30], 'lens_motor_position', 'step', 'increasing_near_to_far')
    assert torch.allclose(b.canonical_coordinate, torch.tensor([0.0, 0.5, 1.0]))
    assert not b.physical_valid_mask.any()


def test_unknown_ordered_rank_protocol_metric_disabled():
    b = canonicalize_focal_coordinates([3, 1, 2], 'unknown_ordered', 'unknown', 'increasing_near_to_far')
    assert torch.allclose(b.canonical_rank, torch.tensor([0.0, 0.5, 1.0]))
    assert b.spacing_confidence.item() < 1.0
    verifier = FocalPhysicalVerifier(verification_protocol='rank')
    stack = torch.rand(1, 3, 3, 8, 8)
    trace = verifier(stack, b.canonical_rank.unsqueeze(0), torch.rand(1, 1, 8, 8), torch.rand(1, 3, 8, 8))
    assert torch.allclose(trace.stack_reprojection_residual, torch.zeros_like(trace.stack_reprojection_residual))


def test_mixed_coordinate_types_can_feed_batch():
    a = canonicalize_focal_coordinates([1, 2, 4], 'camera_object_distance', 'meter', 'increasing_near_to_far')
    b = canonicalize_focal_coordinates([0, 5, 10], 'lens_motor_position', 'step', 'increasing_near_to_far')
    stack = torch.rand(2, 3, 3, 8, 8)
    out = FocalEvidenceEncoder(hidden=8)(
        stack,
        focal_plane_canonical_coordinates=torch.stack([a.canonical_coordinate, b.canonical_coordinate]),
        focal_plane_ranks=torch.stack([a.canonical_rank, b.canonical_rank]),
        coordinate_type_id=torch.stack([a.coordinate_type_id, b.coordinate_type_id]),
        physical_coordinates=torch.stack([a.physical_values, b.physical_values]),
        physical_coordinate_mask=torch.stack([a.physical_valid_mask, b.physical_valid_mask]),
    )
    assert out['focal_posterior'].shape[:2] == (2, 3)


def test_invalid_type_unit_combination_errors():
    with pytest.raises(ValueError):
        canonicalize_focal_coordinates([1, 2], 'camera_object_distance', 'diopter', 'increasing_near_to_far')
