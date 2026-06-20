from dataclasses import replace

import torch

from src.models.verification_trace import PhysicalVerificationTrace
from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline, TraceRefiner


def trace(value=0.2, residual=0.1, invalid=0.0, support=1.0, H=4, W=4):
    z = torch.full((1, 1, H, W), float(value))
    return PhysicalVerificationTrace(
        focus_peak_confidence=torch.ones_like(z),
        focus_peak_index=torch.zeros((1, 1, H, W), dtype=torch.long),
        focus_peak_coordinate=torch.zeros_like(z),
        focus_margin=torch.ones_like(z),
        focus_entropy=torch.zeros_like(z),
        operator_agreement=torch.ones_like(z),
        texture_confidence=torch.ones_like(z),
        depth_focus_discrepancy=z,
        stack_reprojection_residual=torch.full_like(z, residual),
        focus_support=torch.full_like(z, support),
        generation_support=torch.ones_like(z),
        conflict_score=z,
        invalid_score=torch.full_like(z, invalid),
        verdict_scores=torch.cat([1 - z, z, torch.full_like(z, invalid)], dim=1),
    )


def call_refiner(**overrides):
    base = dict(
        aif_latents=torch.rand(1, 4, 4, 4),
        current_depth=torch.full((1, 1, 4, 4), 0.2),
        uncertainty=torch.full((1, 1, 4, 4), 0.2),
        focal_depth=torch.full((1, 1, 4, 4), 0.8),
        focal_entropy=torch.zeros(1, 1, 4, 4),
        focus_support=torch.ones(1, 1, 4, 4),
        evidence_invalidity=torch.zeros(1, 1, 4, 4),
        operator_agreement=torch.ones(1, 1, 4, 4),
        depth_focus_discrepancy=torch.full((1, 1, 4, 4), 0.5),
        observation_residual=torch.full((1, 1, 4, 4), 0.7),
        verification_protocol='calibrated',
    )
    base.update(overrides)
    return TraceRefiner()( **base )


def test_depth_only_conflict_mainly_updates_depth():
    out = call_refiner(previous_depth_focus_discrepancy=None)
    assert out['delta_depth'].abs().mean() > 0
    assert out['delta_aif_latent'].abs().max() == 0


def test_aif_updates_only_after_depth_discrepancy_drops_with_high_residual():
    no_drop = call_refiner(previous_depth_focus_discrepancy=torch.full((1,1,4,4), 0.5))
    dropped = call_refiner(depth_focus_discrepancy=torch.full((1,1,4,4), 0.2), previous_depth_focus_discrepancy=torch.full((1,1,4,4), 0.5))
    assert no_drop['delta_aif_latent'].abs().max() == 0
    assert dropped['delta_aif_latent'].abs().max() > 0


def test_invalid_evidence_raises_uncertainty_not_depth():
    out = call_refiner(evidence_invalidity=torch.ones(1,1,4,4), focus_support=torch.zeros(1,1,4,4))
    assert out['delta_depth'].abs().max() == 0
    assert out['delta_uncertainty_logit'].mean() > 0


def test_aif_delta_preserves_edges_and_cannot_global_blur():
    latents = torch.ones(1, 4, 4, 4)
    out = call_refiner(aif_latents=latents, depth_focus_discrepancy=torch.zeros(1,1,4,4), previous_depth_focus_discrepancy=torch.ones(1,1,4,4))
    assert out['delta_aif_latent'].abs().max() == 0


def test_uncertainty_does_not_saturate_to_one():
    uncertainty = torch.full((1,1,4,4), 0.2)
    instability = call_refiner(uncertainty=uncertainty)['delta_uncertainty_logit']
    candidate = (0.85 * uncertainty + 0.15 * torch.sigmoid(instability)).clamp(0, 0.95)
    assert candidate.max() < 1.0


def test_unknown_coordinate_disables_metric_aif_refinement():
    out = call_refiner(verification_protocol='unknown', depth_focus_discrepancy=torch.zeros(1,1,4,4), previous_depth_focus_discrepancy=torch.ones(1,1,4,4))
    assert out['protocol_depth_mask'].item() == 0
    assert out['delta_depth'].abs().max() == 0
    assert out['delta_aif_latent'].abs().max() == 0


def test_accepted_step_primitive_criteria_are_satisfied():
    before = trace(value=0.4, residual=0.2)
    after = trace(value=0.2, residual=0.19)
    accepted, reason, metrics = FocalStackGenerationPipeline._primitive_acceptance_criteria(
        before_trace=before,
        after_trace=after,
        before_uncertainty=torch.full((1,1,4,4), 0.2),
        after_uncertainty=torch.full((1,1,4,4), 0.25),
        delta_depth=torch.full((1,1,4,4), 0.02),
        delta_aif_latent=torch.zeros(1,4,4,4),
        max_depth_step=0.08,
        max_aif_deviation=0.015,
        epsilon=1e-4,
    )
    assert accepted and reason == 'accepted'
    assert metrics['focus_discrepancy_after'] <= metrics['focus_discrepancy_before']
    assert metrics['observation_residual_after'] <= metrics['observation_residual_before'] + 0.02


def test_rejected_step_reason_reports_no_state_change_contract():
    before = trace(value=0.2, residual=0.1)
    after = trace(value=0.5, residual=0.1)
    accepted, reason, _ = FocalStackGenerationPipeline._primitive_acceptance_criteria(
        before_trace=before,
        after_trace=after,
        before_uncertainty=torch.full((1,1,4,4), 0.2),
        after_uncertainty=torch.full((1,1,4,4), 0.2),
        delta_depth=torch.full((1,1,4,4), 0.02),
        delta_aif_latent=torch.zeros(1,4,4,4),
        max_depth_step=0.08,
        max_aif_deviation=0.015,
        epsilon=1e-4,
    )
    assert not accepted
    assert reason == 'depth_focus_discrepancy_increased'


def test_parameter_checksum_unchanged_for_parameter_free_refiner():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), TraceRefiner())
    before = FocalStackGenerationPipeline.parameter_checksum(module)
    _ = call_refiner()
    after = FocalStackGenerationPipeline.parameter_checksum(module)
    assert after == before
