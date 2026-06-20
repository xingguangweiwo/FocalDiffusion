import math
from types import SimpleNamespace

import torch
import pytest

from script.evaluate import compute_trace_metrics
from src.models.verification_trace import PhysicalVerificationTrace
from src.training.trainer import TraceMiningBuffer, migrate_verification_config


def make_trace():
    z = torch.tensor([[[[0.1, 0.7], [0.2, 0.4]]]])
    invalid = torch.tensor([[[[0.1, 0.2], [0.8, 0.1]]]])
    ones = torch.ones_like(z)
    return PhysicalVerificationTrace(
        focus_peak_confidence=ones,
        focus_peak_index=torch.zeros((1,1,2,2), dtype=torch.long),
        focus_peak_coordinate=torch.zeros_like(z),
        focus_margin=ones,
        focus_entropy=torch.zeros_like(z),
        operator_agreement=ones,
        texture_confidence=ones,
        depth_focus_discrepancy=z,
        stack_reprojection_residual=torch.zeros_like(z),
        focus_support=ones,
        generation_support=ones,
        conflict_score=z,
        invalid_score=invalid,
        verdict_scores=torch.cat([1-z, z, invalid], dim=1),
    )


def test_verification_config_migration_and_focus_operator_cleanup():
    cfg = {
        "validation": {"heldout_verifier": {"focus_operator": "laplacian_tenengrad_heldout", "confidence_threshold": 0.9}},
        "training": {"self_improvement": {"round_id": "M2"}},
    }
    migrated = migrate_verification_config(cfg)
    assert set(migrated["verification"]) == {"mining", "refinement", "evaluation"}
    assert migrated["verification"]["evaluation"]["focus_operator"] == "gradient_variance"
    assert migrated["verification"]["evaluation"]["thresholds"]["confidence"] == 0.9
    assert migrated["training"]["self_improvement"] == {"round_index": 2}


def test_invalid_focus_operator_raises():
    with pytest.raises(ValueError):
        migrate_verification_config({"verification": {"evaluation": {"focus_operator": "bad"}}})


def test_trace_metrics_are_slim_and_do_not_fabricate_uncertainty():
    output = SimpleNamespace(physical_verification_trace=make_trace(), uncertainty=None, uncertainty_final=None)
    metrics = compute_trace_metrics(output)
    assert metrics["uncertainty_available"] is False
    assert math.isnan(metrics["physical_risk_coverage_auc"])
    assert math.isnan(metrics["error_detection_auprc"])
    for removed in ("HCPVR", "Physical-AURC", "selective_physical_risk_at_coverage", "error_violation_detection_auroc", "uncertainty_error_spearman"):
        assert removed not in metrics


def test_trace_metrics_keep_only_canonical_reliability_keys_with_uncertainty():
    output = SimpleNamespace(
        physical_verification_trace=make_trace(),
        uncertainty_final=torch.tensor([[[0.1, 0.8], [0.7, 0.2]]]),
        uncertainty=None,
    )
    metrics = compute_trace_metrics(output)
    assert metrics["uncertainty_available"] is True
    assert "physical_risk_coverage_auc" in metrics
    assert "high_confidence_physical_violation_rate" in metrics
    assert "error_detection_auprc" in metrics
    assert math.isnan(metrics["error_detection_auprc"])


def test_manifest_migration_is_single_ingress_for_duplicate_fields(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"round_id":"M1","sample_path":"sample-a","crop":{"y0":0,"y1":1,"x0":0,"x1":1},"source_shape":{"height":1,"width":1},"conflict_mean":0.2,"invalid_mean":0.3,"support_target":0.7}\n')
    buffer = TraceMiningBuffer(max_items=2, manifest_path=path)
    item = buffer.items[0]
    assert item["round_index"] == 1
    assert item["sample_id"] == "sample-a"
    assert item["conflict_target"] == 0.2
    assert item["invalid_target"] == 0.3
    for removed in ("round_id", "sample_path", "conflict_mean", "invalid_mean"):
        assert removed not in item
