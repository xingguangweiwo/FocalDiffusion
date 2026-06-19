from types import SimpleNamespace

import math
import pytest
import torch

from script.evaluate import compute_trace_metrics
from script.train import _validate_coordinate_protocols


def _toy_output(conflict, invalid, uncertainty, reference_verdict=None):
    trace = SimpleNamespace(
        focus_support=torch.ones_like(conflict),
        conflict_score=conflict,
        invalid_score=invalid,
        verdict_scores=torch.zeros((conflict.shape[0], 3, *conflict.shape[-2:])),
    )
    return SimpleNamespace(
        physical_verification_trace=trace,
        uncertainty=uncertainty,
        uncertainty_final=None,
        reference_verdict=reference_verdict,
    )


def test_hcpvr_increases_for_high_confidence_high_violation():
    low_violation = _toy_output(
        conflict=torch.full((1, 1, 2, 2), 0.1),
        invalid=torch.full((1, 1, 2, 2), 0.1),
        uncertainty=torch.full((1, 1, 2, 2), 0.05),
    )
    high_violation = _toy_output(
        conflict=torch.full((1, 1, 2, 2), 0.9),
        invalid=torch.full((1, 1, 2, 2), 0.1),
        uncertainty=torch.full((1, 1, 2, 2), 0.05),
    )

    low_metrics = compute_trace_metrics(low_violation, confidence_threshold=0.5, violation_threshold=0.5)
    high_metrics = compute_trace_metrics(high_violation, confidence_threshold=0.5, violation_threshold=0.5)

    assert low_metrics["high_confidence_physical_violation_rate"] == pytest.approx(0.0)
    assert high_metrics["high_confidence_physical_violation_rate"] == pytest.approx(1.0)
    assert high_metrics["physical_risk_coverage_auc"] > low_metrics["physical_risk_coverage_auc"]
    forbidden_metric_keys = {"".join(("FC", "PV")), "".join(("PV", "PR", "_at_coverage"))}
    assert forbidden_metric_keys.isdisjoint(high_metrics)


def test_high_invalid_rejection_reduces_accepted_coverage():
    accepted = _toy_output(
        conflict=torch.full((1, 1, 2, 2), 0.1),
        invalid=torch.full((1, 1, 2, 2), 0.1),
        uncertainty=torch.full((1, 1, 2, 2), 0.05),
    )
    rejected = _toy_output(
        conflict=torch.full((1, 1, 2, 2), 0.1),
        invalid=torch.full((1, 1, 2, 2), 0.9),
        uncertainty=torch.full((1, 1, 2, 2), 0.05),
    )

    accepted_metrics = compute_trace_metrics(accepted, confidence_threshold=0.5, violation_threshold=0.5)
    rejected_metrics = compute_trace_metrics(rejected, confidence_threshold=0.5, violation_threshold=0.5)

    assert accepted_metrics["accepted_coverage"] == pytest.approx(1.0)
    assert rejected_metrics["accepted_coverage"] == pytest.approx(0.0)
    assert rejected_metrics["accepted_coverage"] < accepted_metrics["accepted_coverage"]


def test_low_violation_high_confidence_has_near_zero_hcpvr():
    output = _toy_output(
        conflict=torch.tensor([[[[0.01, 0.05], [0.1, 0.2]]]]),
        invalid=torch.tensor([[[[0.02, 0.03], [0.04, 0.05]]]]),
        uncertainty=torch.full((1, 1, 2, 2), 0.01),
    )

    metrics = compute_trace_metrics(output, confidence_threshold=0.5, violation_threshold=0.5)

    assert metrics["high_confidence_physical_violation_rate"] == pytest.approx(0.0)
    assert metrics["selective_physical_risk_at_coverage"] == pytest.approx(0.0925)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["reference_type"] == "internal_verifier"


def test_hcpvr_denominator_zero_returns_nan_and_coverage_not_fake_zero():
    output = _toy_output(
        conflict=torch.full((1, 1, 2, 2), 0.9),
        invalid=torch.full((1, 1, 2, 2), 0.9),
        uncertainty=torch.full((1, 1, 2, 2), 1.0),
    )

    metrics = compute_trace_metrics(output, confidence_threshold=0.5, conflict_threshold=0.5, invalid_threshold=0.5, coverage=0.25)

    assert math.isnan(metrics["high_confidence_physical_violation_rate"])
    assert metrics["coverage"] == pytest.approx(0.25)
    assert metrics["accepted_coverage"] == pytest.approx(0.0)
    assert metrics["invalid_overconfidence_rate"] == pytest.approx(0.0)


def test_selective_risk_is_computed_per_sample_before_averaging_for_different_k_and_resolution():
    output = _toy_output(
        conflict=torch.tensor([
            [[[0.0, 1.0, 1.0, 1.0]]],
            [[[0.0, 0.0, 0.0, 1.0]]],
        ]),
        invalid=torch.zeros(2, 1, 1, 4),
        uncertainty=torch.tensor([
            [[[0.0, 0.1, 0.2, 0.3]]],
            [[[0.0, 0.1, 0.2, 0.3]]],
        ]),
    )

    metrics = compute_trace_metrics(
        output,
        confidence_threshold=0.0,
        conflict_threshold=0.5,
        invalid_threshold=0.5,
        coverage=0.5,
    )

    assert metrics["selective_physical_risk_at_coverage"] == pytest.approx(0.25)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert "physical_risk_coverage_auc" in metrics
    assert "error_violation_detection_auroc" in metrics
    assert "error_violation_detection_auprc" in metrics
    assert "uncertainty_error_spearman" in metrics


def test_reference_agreement_metrics_only_with_reference_verdict():
    output = _toy_output(
        conflict=torch.tensor([[[[0.1, 0.9], [0.1, 0.1]]]]),
        invalid=torch.tensor([[[[0.1, 0.1], [0.9, 0.1]]]]),
        uncertainty=torch.zeros(1, 1, 2, 2),
    )
    no_reference = compute_trace_metrics(output)
    assert "verifier_agreement_accuracy" not in no_reference

    output.reference_verdict = torch.tensor([[[0, 1], [2, 0]]])
    with_reference = compute_trace_metrics(output, reference_type="heldout_verifier")
    assert with_reference["reference_type"] == "heldout_verifier"
    assert with_reference["verifier_agreement_accuracy"] == pytest.approx(1.0)
    assert with_reference["verifier_agreement_macro_f1"] == pytest.approx(1.0)


def _source(name, unit="m", coord_type="distance"):
    return {
        "name": name,
        "data_root": "/tmp/data",
        "filelist": "/tmp/list.txt",
        "focal_coordinate_type": coord_type,
        "focal_coordinate_unit": unit,
        "depth_coordinate_type": "metric_depth",
        "camera_calibration": True,
        "evaluation_mode": "calibrated",
    }


def test_m_and_mm_linear_units_are_protocol_compatible_and_canonical_metrics_match():
    data_cfg = {
        "train_sources": [_source("meters", "m"), _source("millimeters", "mm")],
        "self_improvement_sources": [_source("adapt", "m")],
        "val_sources": [_source("val", "m")],
        "test_sources": [_source("test", "m")],
    }
    _validate_coordinate_protocols(data_cfg)
    conflict = torch.tensor([[[[0.1, 0.8], [0.2, 0.4]]]])
    invalid = torch.tensor([[[[0.1, 0.2], [0.1, 0.7]]]])
    uncertainty = torch.tensor([[[[0.1, 0.1], [0.8, 0.2]]]])
    meters = _toy_output(conflict, invalid, uncertainty)
    millimeters = _toy_output(conflict.clone(), invalid.clone(), uncertainty.clone())
    meters.focal_plane_distances = torch.tensor([[0.5, 1.0, 2.0]])
    millimeters.focal_plane_distances = meters.focal_plane_distances * 1000.0
    assert compute_trace_metrics(meters) == compute_trace_metrics(millimeters)


def test_distance_and_inverse_distance_are_not_linear_equivalent():
    data_cfg = {
        "train_sources": [_source("distance", "m", "distance"), _source("inverse", "1_per_m", "inverse_distance")],
        "self_improvement_sources": [_source("adapt", "m")],
        "val_sources": [_source("val", "m")],
        "test_sources": [_source("test", "m")],
    }
    with pytest.raises(ValueError, match="mixed focal coordinate types"):
        _validate_coordinate_protocols(data_cfg)
