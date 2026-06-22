import pytest
import torch

from src.models import FocalStackProcessor
from src.models.focal_evidence_encoder import FocalEvidenceEncoder, PhysicalEvidenceEstimator, build_physical_evidence_features
from src.training.losses import FocalStackGenerationLoss, build_focal_axis_soft_targets


def test_focal_evidence_shapes_probability_and_ranges():
    B, N, C, H, W = 2, 5, 3, 32, 48
    stack = torch.randn(B, N, C, H, W)
    focal_plane_distances = torch.linspace(0.3, 10.0, N).repeat(B, 1)

    out = FocalEvidenceEncoder()(stack, focal_plane_distances[0])

    assert out["focal_posterior"].shape == (B, N, H, W)
    assert out["focal_depth_canonical"].shape == (B, 1, H, W)
    assert out["focal_entropy"].shape == (B, 1, H, W)
    assert out["focal_peak_confidence"].shape == (B, 1, H, W)
    assert torch.allclose(out["focal_posterior"].sum(dim=1), torch.ones(B, H, W), atol=1e-4)
    assert out["focal_depth_canonical"].min() >= 0
    assert out["focal_depth_canonical"].max() <= 1
    assert out["focal_entropy"].min() >= 0
    assert out["focal_entropy"].max() <= 1
    assert out["focal_peak_confidence"].min() >= 0
    assert out["focal_peak_confidence"].max() <= 1


def test_focal_evidence_supports_one_hundred_slices():
    B, N, C, H, W = 1, 100, 3, 16, 16
    stack = torch.randn(B, N, C, H, W)
    focal_plane_distances = torch.linspace(0.3, 10.0, N).repeat(B, 1)

    out = FocalEvidenceEncoder(hidden=16)(stack, focal_plane_distances)

    assert out["focal_posterior"].shape == (B, N, H, W)
    assert out["focal_depth_canonical"].shape == (B, 1, H, W)


def test_focal_processor_rejects_more_than_one_hundred_slices():
    processor = FocalStackProcessor(
        feature_dim=16,
        max_sequence_length=100,
        patch_size=8,
        focal_attention_heads=4,
        focal_attention_depth=1,
    )
    stack = torch.randn(1, 101, 3, 16, 16)
    focal_plane_distances = torch.linspace(0.0, 1.0, 101).unsqueeze(0)

    with pytest.raises(ValueError, match="Sequence length 101 exceeds maximum 100"):
        processor(stack, focal_plane_distances)


def test_pipeline_output_dataclass_exposes_focal_evidence_fields():
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationOutput

    history = [
        {
            "step": 0,
            "final_depth_canonical": torch.zeros(1, 1, 8, 8),
            "uncertainty_final": torch.ones(1, 1, 8, 8),
            "mean_conflict_score": 0.1,
            "mean_invalid_score": 0.2,
            "mean_focus_support": 0.3,
            "mean_generation_support": 0.4,
        }
    ]
    out = FocalStackGenerationOutput(
        depth_map=torch.zeros(1, 8, 8),
        all_in_focus_image=torch.zeros(1, 3, 8, 8),
        refinement_history=history,
    )
    for name in (
        "generated_depth_canonical",
        "focal_depth_canonical",
        "final_depth_canonical",
        "focal_posterior",
        "focal_entropy",
        "focal_peak_confidence",
        "physical_evidence_support",
        "focal_evidence_weight",
        "generative_prior_weight",
        "abstention_weight",
        "posterior_margin",
        "depth_disagreement",
        "uncertainty_disagreement",
        "uncertainty_final",
        "refinement_history",
    ):
        assert hasattr(out, name)
    assert out.refinement_history is history


def test_focal_posterior_kl_loss_is_finite():
    B, N, C, H, W = 2, 5, 3, 16, 16
    stack = torch.randn(B, N, C, H, W)
    focal_plane_distances = torch.linspace(0.3, 10.0, N).repeat(B, 1)
    evidence = FocalEvidenceEncoder(hidden=16)(stack, focal_plane_distances)
    depth_norm = torch.rand(B, 1, H, W)
    focus_target, _ = build_focal_axis_soft_targets(depth_norm, focal_plane_distances)

    loss_fn = FocalStackGenerationLoss(
        diffusion_weight=0.0,
        depth_weight=1.0,
        rgb_weight=0.0,
        focal_posterior_kl_weight=0.2,
        focus_depth_weight=0.2,
        prior_depth_weight=0.05,
    )
    loss_dict = loss_fn(
        diffusion_pred=torch.zeros(B, 4),
        diffusion_target=torch.zeros(B, 4),
        depth_target=depth_norm,
        depth_range=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        generated_depth_canonical=depth_norm,
        focal_depth_canonical=evidence["focal_depth_canonical"],
        final_depth_canonical=depth_norm,
        focal_posterior=evidence["focal_posterior"],
        focal_entropy=evidence["focal_entropy"],
        focal_plane_distances=focal_plane_distances,
        focal_evidence_weight=torch.rand(B, 1, H, W),
        focal_stack=stack,
        rgb_pred=torch.randn(B, C, H, W),
    )

    assert torch.isfinite(focus_target).all()
    assert "loss_focal_posterior_kl" in loss_dict
    assert torch.isfinite(loss_dict["loss_focal_posterior_kl"])
    assert "loss_focal_evidence_weight" in loss_dict
    assert torch.isfinite(loss_dict["total"])


def test_physical_evidence_feature_validation_rejects_shape_mismatch():
    focal_posterior = torch.softmax(torch.randn(2, 4, 8, 8), dim=1)
    bad_depth = torch.rand(2, 1, 4, 4)

    with pytest.raises(ValueError, match="focal_depth_canonical must have shape"):
        build_physical_evidence_features(
            focal_posterior=focal_posterior,
            focal_entropy=torch.rand(2, 1, 8, 8),
            focal_depth_canonical=bad_depth,
            generated_depth_canonical=torch.rand(2, 1, 8, 8),
            generative_uncertainty=torch.rand(2, 1, 8, 8),
        )


def test_physical_evidence_estimator_rejects_wrong_channel_count():
    head = PhysicalEvidenceEstimator(in_channels=5, hidden=8)

    with pytest.raises(ValueError, match="support_inputs must have 5 channels"):
        head(torch.rand(1, 4, 8, 8))


def test_physical_evidence_support_head_shapes_and_gate_normalization():
    support_inputs = torch.randn(2, 5, 32, 48)
    head = PhysicalEvidenceEstimator(in_channels=5, hidden=16)
    out = head(support_inputs)

    assert out["gate_logits"].shape == (2, 3, 32, 48)
    assert out["focal_evidence_weight"].shape == (2, 1, 32, 48)
    assert out["generative_prior_weight"].shape == (2, 1, 32, 48)
    assert out["abstention_weight"].shape == (2, 1, 32, 48)
    assert out["uncertainty_final"].shape == (2, 1, 32, 48)
    assert out["physical_evidence_support"].shape == (2, 1, 32, 48)
    gate_sum = out["focal_evidence_weight"] + out["generative_prior_weight"] + out["abstention_weight"]
    assert torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-5)


def test_build_physical_evidence_features_shapes():
    B, N, H, W = 2, 5, 32, 48
    focal_posterior = torch.softmax(torch.randn(B, N, H, W), dim=1)
    focal_entropy = torch.rand(B, 1, H, W)
    depth_focus = torch.rand(B, 1, H, W)
    depth_prior = torch.rand(B, 1, H, W)
    generative_uncertainty = torch.rand(B, 1, H, W)

    support_inputs, support_maps = build_physical_evidence_features(
        focal_posterior,
        focal_entropy,
        depth_focus,
        depth_prior,
        generative_uncertainty,
    )

    assert support_inputs.shape == (B, 5, H, W)
    assert support_maps["focal_peak_confidence"].shape == (B, 1, H, W)
    assert support_maps["posterior_margin"].shape == (B, 1, H, W)
    assert support_maps["depth_disagreement"].shape == (B, 1, H, W)


def test_local_affinity_shapes_and_range():
    from src.training.losses import _compute_local_affinity

    B, C, H, W = 2, 3, 12, 10
    evidence_image = torch.randn(B, C, H, W)

    affinity = _compute_local_affinity(evidence_image)

    assert affinity["x"].shape == (B, 1, H, W - 1)
    assert affinity["y"].shape == (B, 1, H - 1, W)
    assert torch.isfinite(affinity["x"]).all()
    assert torch.isfinite(affinity["y"]).all()
    assert affinity["x"].min() >= 0
    assert affinity["x"].max() <= 1
    assert affinity["y"].min() >= 0
    assert affinity["y"].max() <= 1


def test_focal_consistency_regularizers_are_finite():
    from src.training.losses import (
        _compute_local_affinity,
        _depth_affinity_smoothness_loss,
        _focal_axis_smoothness_loss,
        _gate_consistency_loss,
        _posterior_consistency_loss,
    )

    B, N, H, W = 2, 5, 12, 10
    focal_posterior = torch.softmax(torch.randn(B, N, H, W), dim=1)
    depth_final = torch.rand(B, 1, H, W)
    focal_evidence_weight = torch.rand(B, 1, H, W)
    evidence_image = torch.randn(B, 3, H, W)
    affinity = _compute_local_affinity(evidence_image)

    losses = [
        _posterior_consistency_loss(focal_posterior, affinity),
        _depth_affinity_smoothness_loss(depth_final, affinity),
        _gate_consistency_loss(focal_evidence_weight, affinity),
        _focal_axis_smoothness_loss(focal_posterior),
    ]
    for loss in losses:
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    two_plane_posterior = torch.softmax(torch.randn(B, 2, H, W), dim=1)
    focal_axis_loss = _focal_axis_smoothness_loss(two_plane_posterior)
    assert focal_axis_loss.dim() == 0
    assert torch.isfinite(focal_axis_loss)
    assert focal_axis_loss.item() == 0.0


def test_focal_diffusion_loss_includes_consistency_terms():
    B, N, C, H, W = 2, 5, 3, 12, 10
    depth_target = torch.rand(B, 1, H, W)
    focal_posterior = torch.softmax(torch.randn(B, N, H, W), dim=1)

    loss_fn = FocalStackGenerationLoss(
        diffusion_weight=0.0,
        depth_weight=1.0,
        rgb_weight=0.0,
        posterior_consistency_weight=0.02,
        depth_affinity_smoothness_weight=0.01,
        gate_consistency_weight=0.005,
        focal_axis_smoothness_weight=0.002,
    )
    loss_dict = loss_fn(
        diffusion_pred=torch.zeros(B, 4),
        diffusion_target=torch.zeros(B, 4),
        depth_target=depth_target,
        depth_range=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        final_depth_canonical=torch.rand(B, 1, H, W),
        generated_depth_canonical=torch.rand(B, 1, H, W),
        focal_depth_canonical=torch.rand(B, 1, H, W),
        focal_posterior=focal_posterior,
        focal_entropy=torch.rand(B, 1, H, W),
        focal_plane_distances=torch.linspace(0.3, 10.0, N).repeat(B, 1),
        focal_stack=torch.randn(B, N, C, H, W),
        rgb_pred=torch.randn(B, C, H, W),
        rgb_target=torch.randn(B, C, H, W),
        focal_evidence_weight=torch.rand(B, 1, H, W),
    )

    assert "loss_posterior_consistency" in loss_dict
    assert "loss_depth_affinity_smoothness" in loss_dict
    assert "loss_gate_consistency" in loss_dict
    assert "loss_focal_axis_smoothness" in loss_dict
    assert torch.isfinite(loss_dict["total"])


def test_focal_physical_verifier_trace_shapes_for_batch_first_stack():
    from src.models.physics_modules import FocalPhysicalVerifier

    B, K, C, H, W = 2, 4, 3, 12, 10
    trace = FocalPhysicalVerifier()(torch.rand(B, K, C, H, W), torch.linspace(0, 1, K), torch.rand(B, 1, H, W), torch.rand(B, C, H, W))

    for name in (
        "focus_peak_confidence",
        "focus_peak_index",
        "focus_peak_coordinate",
        "focus_margin",
        "focus_entropy",
        "operator_agreement",
        "texture_confidence",
        "depth_focus_discrepancy",
        "stack_reprojection_residual",
        "focus_support",
        "generation_support",
        "conflict_score",
        "invalid_score",
    ):
        assert getattr(trace, name).shape == (B, 1, H, W)
    assert trace.focus_peak_index.dtype == torch.long
    assert trace.focus_peak_coordinate.min() >= 0
    assert trace.focus_peak_coordinate.max() <= 1
    assert trace.verdict_scores.shape == (B, 3, H, W)
    for value in trace.__dict__.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all()


def test_trace_refinement_preserves_output_fields():
    from src.models.physics_modules import FocalPhysicalVerifier
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationOutput, FocalStackGenerationPipeline

    B, K, C, H, W = 1, 4, 3, 12, 10
    stack = torch.rand(B, K, C, H, W)
    trace = FocalPhysicalVerifier()(stack, torch.linspace(0, 1, K), torch.rand(B, 1, H, W), torch.rand(B, C, H, W))
    depth, uncertainty = FocalStackGenerationPipeline._apply_trace_refinement(
        final_depth_canonical=torch.rand(B, 1, H, W),
        focal_depth_canonical=torch.rand(B, 1, H, W),
        generated_depth_canonical=torch.rand(B, 1, H, W),
        uncertainty_final=torch.zeros(B, 1, H, W),
        trace=trace,
    )
    out = FocalStackGenerationOutput(
        depth_map=depth.squeeze(1),
        all_in_focus_image=torch.rand(B, C, H, W),
        uncertainty_final=uncertainty.squeeze(1),
        final_depth_canonical=depth.squeeze(1),
        physical_verification_trace=trace,
    )

    assert out.depth_map.shape == (B, H, W)
    assert out.uncertainty_final.shape == (B, H, W)
    assert out.final_depth_canonical.shape == (B, H, W)
    assert out.physical_verification_trace is trace


def test_trace_refinement_accepts_only_improved_physical_risk():
    from dataclasses import replace

    from src.models.physics_modules import FocalPhysicalVerifier
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline

    B, K, C, H, W = 1, 3, 3, 8, 8
    trace = FocalPhysicalVerifier()(torch.rand(B, K, C, H, W), torch.linspace(0, 1, K), torch.rand(B, 1, H, W), torch.rand(B, C, H, W))
    improved = replace(
        trace,
        conflict_score=(trace.conflict_score * 0.1).clamp(0.0, 1.0),
        invalid_score=(trace.invalid_score * 0.1).clamp(0.0, 1.0),
    )
    worse = replace(
        trace,
        conflict_score=torch.ones_like(trace.conflict_score),
        invalid_score=torch.ones_like(trace.invalid_score),
    )

    accepted, before, after = FocalStackGenerationPipeline._accept_refinement_candidate(trace, improved, epsilon=1e-6)
    assert accepted
    assert after < before
    rejected, before_reject, after_reject = FocalStackGenerationPipeline._accept_refinement_candidate(trace, worse, epsilon=1e-6)
    assert not rejected
    assert after_reject > before_reject


def test_evaluate_trace_metrics_computes_hcpvr_selective_risk_at_coverage():
    from types import SimpleNamespace

    from script.evaluate import compute_trace_metrics

    trace = SimpleNamespace(
        focus_support=torch.ones(1, 1, 2, 2),
        conflict_score=torch.tensor([[[[0.1, 0.7], [0.2, 0.6]]]]),
        invalid_score=torch.tensor([[[[0.2, 0.1], [0.8, 0.4]]]]),
    )
    output = SimpleNamespace(
        uncertainty_final=torch.tensor([[[0.1, 0.2], [0.8, 0.3]]]),
        physical_verification_trace=trace,
    )

    metrics = compute_trace_metrics(output, confidence_threshold=0.5, violation_threshold=0.5, coverage=0.5)

    assert metrics["high_confidence_physical_violation_rate"] == pytest.approx(2 / 3)
    assert metrics["selective_physical_risk_at_coverage"] == pytest.approx(0.45)
    assert metrics["invalid_overconfidence_rate"] == pytest.approx(0.0)
    assert metrics["accepted_coverage"] == pytest.approx(0.75)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["mean_conflict_score"] == pytest.approx(0.4)
    assert metrics["mean_invalid_score"] == pytest.approx(0.375)
    assert "error_violation_detection_auroc" in metrics


def test_training_trace_loss_uses_detached_verifier_targets_and_backpropagates_proxies():
    from src.models.physics_modules import FocalPhysicalVerifier

    B, N, C, H, W = 2, 4, 3, 12, 10
    focal_stack = torch.rand(B, N, C, H, W) * 2.0 - 1.0
    focal_plane_distances = torch.linspace(0.2, 1.0, N).repeat(B, 1)
    final_depth_canonical = torch.rand(B, 1, H, W, requires_grad=True)
    generated_depth_canonical = torch.rand(B, 1, H, W, requires_grad=True)
    rgb_recon = torch.rand(B, C, H, W, requires_grad=True) * 2.0 - 1.0

    with torch.no_grad():
        trace = FocalPhysicalVerifier()(
            focal_stack,
            focal_plane_distances,
            final_depth_canonical.detach(),
            rgb_recon.detach(),
            generated_depth_canonical.detach(),
        )

    support_logits = torch.randn(B, 1, H, W, requires_grad=True)
    invalid_logits = torch.randn(B, 1, H, W, requires_grad=True)
    predicted_support = torch.sigmoid(support_logits)
    predicted_invalid = torch.sigmoid(invalid_logits)
    diffusion_pred = torch.zeros(B, 4, requires_grad=True)
    diffusion_target = torch.zeros(B, 4)

    from src.training.losses import FocalStackGenerationLoss

    loss_fn = FocalStackGenerationLoss(lambda_trace=0.5, lambda_violation=0.5, lambda_invalid=0.5)
    loss_dict = loss_fn(
        diffusion_pred=diffusion_pred,
        diffusion_target=diffusion_target,
        physical_verification_trace=trace,
        physical_evidence_support=predicted_support,
        predicted_verification_support=predicted_support,
        predicted_verification_invalid=predicted_invalid,
        uncertainty=predicted_invalid,
    )

    assert trace.conflict_score.shape == (B, 1, H, W)
    assert trace.invalid_score.shape == (B, 1, H, W)
    assert trace.verdict_scores.shape == (B, 3, H, W)
    assert not trace.conflict_score.requires_grad
    assert loss_dict["loss_trace"].item() > 0.0
    assert torch.isfinite(loss_dict["loss_violation"])
    assert torch.isfinite(loss_dict["mean_conflict_score"])
    assert torch.isfinite(loss_dict["false_confident_violation_rate"])

    loss_dict["total"].backward()

    assert support_logits.grad is not None
    assert invalid_logits.grad is not None
    assert torch.isfinite(support_logits.grad).all()
    assert torch.isfinite(invalid_logits.grad).all()


def test_evaluate_trace_metrics_returns_hcpvr_for_high_confidence_invalid():
    from types import SimpleNamespace

    from script.evaluate import compute_trace_metrics

    trace = SimpleNamespace(
        focus_support=torch.ones(1, 1, 2, 2),
        conflict_score=torch.zeros(1, 1, 2, 2),
        invalid_score=torch.ones(1, 1, 2, 2),
    )
    output = SimpleNamespace(
        uncertainty_final=torch.zeros(1, 2, 2),
        physical_verification_trace=trace,
    )

    metrics = compute_trace_metrics(output, confidence_threshold=0.5, violation_threshold=0.5, coverage=0.5)

    assert metrics["high_confidence_physical_violation_rate"] == pytest.approx(1.0)
    assert metrics["invalid_overconfidence_rate"] == pytest.approx(1.0)
    assert metrics["accepted_coverage"] == pytest.approx(0.0)


def test_evaluate_trace_metrics_selective_risk_is_per_sample_mean():
    from types import SimpleNamespace

    from script.evaluate import compute_trace_metrics

    trace = SimpleNamespace(
        focus_support=torch.ones(2, 1, 1, 4),
        conflict_score=torch.tensor([[[[0.0, 0.9, 0.0, 0.0]]], [[[0.9, 0.8, 0.0, 0.0]]]]),
        invalid_score=torch.zeros(2, 1, 1, 4),
    )
    output = SimpleNamespace(
        uncertainty_final=torch.tensor([[[0.1, 0.2, 0.9, 0.9]], [[0.1, 0.2, 0.9, 0.9]]]),
        physical_verification_trace=trace,
    )

    metrics = compute_trace_metrics(output, confidence_threshold=0.5, violation_threshold=0.5, coverage=0.5)

    assert metrics["selective_physical_risk_at_coverage"] == pytest.approx(0.65)
    assert "physical_risk_coverage_auc" in metrics


def test_serialize_refinement_history_compacts_tensors_for_json():
    from script.evaluate import _serialize_refinement_history

    curve = _serialize_refinement_history(
        [
            {
                "step": 2,
                "final_depth_canonical": torch.zeros(1, 1, 2, 2),
                "uncertainty_final": torch.ones(1, 1, 2, 2),
                "mean_conflict_score": 0.1,
                "mean_invalid_score": 0.2,
                "mean_focus_support": 0.3,
                "mean_generation_support": 0.4,
            }
        ]
    )
    assert curve[0]["step"] == 2
    assert curve[0]["final_depth_canonical"]["shape"] == [1, 1, 2, 2]
    assert curve[0]["uncertainty_final"]["mean"] == pytest.approx(1.0)


def test_trace_refinement_accepts_only_when_risk_improves_by_epsilon():
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline

    current_risk = torch.tensor(0.5)
    improved_risk = torch.tensor(0.45)
    marginal_risk = torch.tensor(0.499)

    assert FocalStackGenerationPipeline._should_accept_refinement(current_risk, improved_risk, epsilon=0.01)
    assert not FocalStackGenerationPipeline._should_accept_refinement(current_risk, marginal_risk, epsilon=0.01)

def test_canonical_focal_coordinates_are_shared_across_components():
    from src.models.focal_evidence_encoder import FocusLikelihoodEstimator
    from src.models.focal_processor import FocalSweepEncoder
    from src.models.physics_modules import DefocusConsistencyVerifier
    from src.training.losses import normalize_focal_coordinates
    from src.utils.image_utils import canonical_focal_coordinates

    distances = torch.tensor([[2.0, 4.0, 8.0], [1.0, 3.0, 5.0]])
    expected, valid = canonical_focal_coordinates(distances, coordinate_type="distance")
    assert valid.all()
    assert torch.allclose(FocusLikelihoodEstimator(hidden=8)._normalize_focal_plane_distances(distances), expected)
    assert torch.allclose(FocalSweepEncoder.normalize_focal_plane_distances(distances), expected)
    assert torch.allclose(
        DefocusConsistencyVerifier._normalize_focal_distances(distances, 2, 3, distances.device, distances.dtype),
        expected,
    )
    assert torch.allclose(normalize_focal_coordinates(distances), expected)

    inverse, _ = canonical_focal_coordinates(distances, coordinate_type="diopter")
    zpos, _ = canonical_focal_coordinates(distances, coordinate_type="z_position")
    index, _ = canonical_focal_coordinates(distances, coordinate_type="index")
    rank, _ = canonical_focal_coordinates(distances, coordinate_type="normalized_rank")
    metric, _ = canonical_focal_coordinates(distances, coordinate_type="distance", metric_coordinates=distances)
    assert inverse.shape == zpos.shape == index.shape == rank.shape == metric.shape == distances.shape
    assert torch.allclose(zpos, expected)
    assert torch.allclose(metric, expected)


def test_image_range_helpers_match_legacy_split_and_pipeline_tensor_ingest():
    from src.models.physics_modules import _split_unit_and_signed_ranges
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline
    from src.utils.image_utils import to_model_range, to_unit_range

    unit = torch.rand(1, 3, 3, 4, 4)
    signed = unit * 2.0 - 1.0
    assert torch.allclose(to_unit_range(signed), unit)
    assert torch.allclose(to_model_range(unit), signed)
    legacy_unit, legacy_signed = _split_unit_and_signed_ranges(signed)
    assert torch.allclose(legacy_unit, unit)
    assert torch.allclose(legacy_signed, signed)
    assert torch.allclose(FocalStackGenerationPipeline._ensure_tensor_stack(None, unit.squeeze(0)), signed)


def test_resize_probability_volume_masks_and_rejects_all_invalid():
    from src.utils.image_utils import resize_probability_volume

    posterior = torch.ones(2, 3, 2, 2)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    resized = resize_probability_volume(posterior, (4, 4), mask)
    assert resized.shape == (2, 3, 4, 4)
    assert torch.all(resized[:, 0] >= 0)
    assert torch.allclose(resized.sum(dim=1), torch.ones(2, 4, 4))
    assert torch.all(resized[0, 1] == 0)
    with pytest.raises(ValueError, match="all-invalid"):
        resize_probability_volume(posterior[:1], (2, 2), torch.tensor([[False, False, False]]))


def test_compatibility_aliases_and_checkpoint_keys_are_preserved():
    from src.models.focal_evidence import FocalEvidenceEncoder, FocusLikelihoodEstimator
    from src.pipelines import FocalDiffusionOutput, FocalDiffusionPipeline, FocalStackGenerationOutput, FocalStackGenerationPipeline

    assert FocalEvidenceEncoder is FocusLikelihoodEstimator
    assert issubclass(FocalDiffusionOutput, FocalStackGenerationOutput)
    assert issubclass(FocalDiffusionPipeline, FocalStackGenerationPipeline)

    checkpoint = {
        "focal_processor_state_dict": {},
        "focal_evidence_head_state_dict": {},
        "task_output_decoder_state_dict": {},
        "physical_evidence_support_head_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "epoch": 0,
        "global_step": 0,
    }
    assert "focal_evidence_head_state_dict" in checkpoint
    assert "physical_evidence_support_head_state_dict" in checkpoint


def test_legacy_checkpoint_keys_load_through_migration_aliases(tmp_path):
    from types import SimpleNamespace
    from src.training.checkpointing import load_checkpoint

    class EmptyModule(torch.nn.Module):
        pass

    class DummyState:
        def load_state_dict(self, state):
            self.state = state

    checkpoint = {
        "focal_processor_state_dict": {},
        "focal_evidence_head_state_dict": {"legacy.extra": torch.tensor(1.0)},
        "task_output_decoder_state_dict": {"legacy.decoder": torch.tensor(2.0)},
        "physical_evidence_support_head_state_dict": {"legacy.support": torch.tensor(3.0)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"last_epoch": 0},
        "epoch": 7,
        "global_step": 42,
    }
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(device="cpu"),
        focal_processor=EmptyModule(),
        focal_evidence_head=EmptyModule(),
        task_output_decoder=EmptyModule(),
        physical_evidence_support_head=EmptyModule(),
        optimizer=DummyState(),
        lr_scheduler=DummyState(),
        ema=None,
        pipeline=SimpleNamespace(transformer=EmptyModule()),
    )
    epoch, step = load_checkpoint(trainer, str(path))
    assert (epoch, step) == (7, 42)
