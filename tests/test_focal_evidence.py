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

    out = FocalStackGenerationOutput(depth_map=torch.zeros(1, 8, 8), all_in_focus_image=torch.zeros(1, 3, 8, 8))
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
    ):
        assert hasattr(out, name)


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
        "focus_peak",
        "focus_margin",
        "focus_entropy",
        "operator_agreement",
        "texture_confidence",
        "depth_focus_discrepancy",
        "defocus_residual",
        "refocus_residual",
        "focus_support",
        "generation_support",
        "conflict_score",
        "invalid_score",
    ):
        assert getattr(trace, name).shape == (B, 1, H, W)
    assert trace.verdict_logits.shape == (B, 3, H, W)


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


def test_evaluate_trace_metrics_computes_fcpv_pvpr_at_coverage():
    from src.models.physics_modules import FocalPhysicalVerifier
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationOutput
    from script.evaluate import compute_trace_metrics

    B, K, C, H, W = 1, 3, 3, 8, 8
    trace = FocalPhysicalVerifier()(torch.rand(B, K, C, H, W), torch.linspace(0, 1, K), torch.rand(B, 1, H, W), torch.rand(B, C, H, W))
    output = FocalStackGenerationOutput(
        depth_map=torch.rand(B, H, W),
        all_in_focus_image=torch.rand(B, C, H, W),
        uncertainty_final=torch.rand(B, H, W),
        physical_verification_trace=trace,
    )
    metrics = compute_trace_metrics(output)

    assert "FCPV" in metrics
    assert "FCPV_soft" in metrics
    assert "PVPR_at_coverage" in metrics
    assert "coverage" in metrics
    assert "mean_conflict_score" in metrics
    assert "mean_invalid_score" in metrics
    assert "uncertainty_violation_alignment" in metrics


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

    loss_fn = FocalStackGenerationLoss(
        diffusion_weight=0.0,
        depth_weight=0.0,
        rgb_weight=0.0,
        lambda_trace=0.05,
        lambda_violation=0.05,
        lambda_invalid=0.05,
    )
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
    assert trace.verdict_logits.shape == (B, 3, H, W)
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


def test_trace_refinement_accepts_only_when_risk_improves_by_epsilon():
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline

    current_risk = torch.tensor(0.5)
    improved_risk = torch.tensor(0.45)
    marginal_risk = torch.tensor(0.499)

    assert FocalStackGenerationPipeline._should_accept_refinement(current_risk, improved_risk, epsilon=0.01)
    assert not FocalStackGenerationPipeline._should_accept_refinement(current_risk, marginal_risk, epsilon=0.01)
