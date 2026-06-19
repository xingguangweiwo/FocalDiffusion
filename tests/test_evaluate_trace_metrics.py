from types import SimpleNamespace

import pytest
import torch

from script.evaluate import compute_trace_metrics


def _toy_output(conflict, invalid, uncertainty):
    trace = SimpleNamespace(
        focus_support=torch.ones_like(conflict),
        conflict_score=conflict,
        invalid_score=invalid,
        verdict_logits=torch.zeros((conflict.shape[0], 2, *conflict.shape[-2:])),
    )
    return SimpleNamespace(
        physical_verification_trace=trace,
        uncertainty=uncertainty,
        uncertainty_final=None,
    )


def test_fcpv_increases_for_high_confidence_high_violation():
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

    assert low_metrics["FCPV"] == pytest.approx(0.0)
    assert high_metrics["FCPV"] == pytest.approx(1.0)
    assert high_metrics["FCPV_soft"] > low_metrics["FCPV_soft"]


def test_high_invalid_rejection_reduces_coverage():
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


def test_low_violation_high_confidence_has_near_zero_fcpv():
    output = _toy_output(
        conflict=torch.tensor([[[[0.01, 0.05], [0.1, 0.2]]]]),
        invalid=torch.tensor([[[[0.02, 0.03], [0.04, 0.05]]]]),
        uncertainty=torch.full((1, 1, 2, 2), 0.01),
    )

    metrics = compute_trace_metrics(output, confidence_threshold=0.5, violation_threshold=0.5)

    assert metrics["FCPV"] == pytest.approx(0.0)
    assert metrics["PVPR_at_coverage"] == pytest.approx(1.0)
    assert metrics["coverage"] == pytest.approx(1.0)


def test_phr_denominator_zero_returns_nan():
    output = _toy_output(
        conflict=torch.full((1, 1, 2, 2), 0.9),
        invalid=torch.full((1, 1, 2, 2), 0.9),
        uncertainty=torch.full((1, 1, 2, 2), 0.05),
    )

    metrics = compute_trace_metrics(output, confidence_threshold=0.5, conflict_threshold=0.5, invalid_threshold=0.5)

    assert torch.isnan(torch.tensor(metrics["PHR"]))
    assert metrics["accepted_coverage"] == pytest.approx(0.0)
    assert metrics["IOR"] == pytest.approx(1.0)


def test_vpr_at_coverage_is_computed_per_sample_before_averaging():
    output = _toy_output(
        conflict=torch.tensor([
            [[[0.0, 0.9, 0.9, 0.9]]],
            [[[0.0, 0.0, 0.0, 0.9]]],
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
        vpr_coverage=0.5,
    )

    assert metrics["VPR_at_coverage_mean"] == pytest.approx(0.75)
    assert metrics["VPR_at_coverage_std"] == pytest.approx(0.25)
    assert metrics["coverage"] == pytest.approx(0.5)
