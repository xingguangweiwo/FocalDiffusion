import pytest
import torch

from src.models import FocalStackProcessor
from src.models.focal_evidence import FocalEvidenceHead, PhysicalSupportHead, build_support_inputs
from src.training.losses import FocalDiffusionLoss, build_soft_focus_target_from_depth


def test_focal_evidence_shapes_probability_and_ranges():
    B, N, C, H, W = 2, 5, 3, 32, 48
    stack = torch.randn(B, N, C, H, W)
    focus_distances = torch.linspace(0.3, 10.0, N).repeat(B, 1)

    out = FocalEvidenceHead()(stack, focus_distances)

    assert out["focus_posterior"].shape == (B, N, H, W)
    assert out["depth_focus_norm"].shape == (B, 1, H, W)
    assert out["focus_entropy"].shape == (B, 1, H, W)
    assert out["focus_peakiness"].shape == (B, 1, H, W)
    assert out["focus_reliability"].shape == (B, 1, H, W)
    assert torch.allclose(out["focus_posterior"].sum(dim=1), torch.ones(B, H, W), atol=1e-4)
    assert out["depth_focus_norm"].min() >= 0
    assert out["depth_focus_norm"].max() <= 1
    assert out["focus_entropy"].min() >= 0
    assert out["focus_entropy"].max() <= 1
    assert out["focus_peakiness"].min() >= 0
    assert out["focus_peakiness"].max() <= 1
    assert torch.allclose(out["focus_reliability"], out["focus_peakiness"])


def test_focal_evidence_supports_one_hundred_slices():
    B, N, C, H, W = 1, 100, 3, 16, 16
    stack = torch.randn(B, N, C, H, W)
    focus_distances = torch.linspace(0.3, 10.0, N).repeat(B, 1)

    out = FocalEvidenceHead(hidden=16)(stack, focus_distances)

    assert out["focus_posterior"].shape == (B, N, H, W)
    assert out["depth_focus_norm"].shape == (B, 1, H, W)


def test_focal_processor_rejects_more_than_one_hundred_slices():
    processor = FocalStackProcessor(
        feature_dim=16,
        max_sequence_length=100,
        patch_size=8,
        focal_attention_heads=4,
        focal_attention_depth=1,
    )
    stack = torch.randn(1, 101, 3, 16, 16)
    focus_distances = torch.linspace(0.0, 1.0, 101).unsqueeze(0)

    with pytest.raises(ValueError, match="Sequence length 101 exceeds maximum 100"):
        processor(stack, focus_distances)


def test_pipeline_output_dataclass_exposes_focal_evidence_fields():
    from src.pipelines.focal_diffusion_pipeline import FocalDiffusionOutput

    out = FocalDiffusionOutput(depth_map=torch.zeros(1, 8, 8), all_in_focus_image=torch.zeros(1, 3, 8, 8))
    for name in (
        "depth_prior",
        "depth_focus",
        "depth_final",
        "focus_posterior",
        "focus_entropy",
        "focus_reliability",
        "focus_peakiness",
        "physical_support",
        "gate_focus",
        "gate_prior",
        "gate_abstain",
        "posterior_margin",
        "depth_disagreement",
        "uncertainty_disagreement",
        "uncertainty_final",
    ):
        assert hasattr(out, name)


def test_focus_posterior_kl_loss_is_finite():
    B, N, C, H, W = 2, 5, 3, 16, 16
    stack = torch.randn(B, N, C, H, W)
    focus_distances = torch.linspace(0.3, 10.0, N).repeat(B, 1)
    evidence = FocalEvidenceHead(hidden=16)(stack, focus_distances)
    depth_norm = torch.rand(B, 1, H, W)
    focus_target, _ = build_soft_focus_target_from_depth(depth_norm, focus_distances)

    loss_fn = FocalDiffusionLoss(
        diffusion_weight=0.0,
        depth_weight=1.0,
        rgb_weight=0.0,
        focus_posterior_kl_weight=0.2,
        focus_depth_weight=0.2,
        prior_depth_weight=0.05,
    )
    loss_dict = loss_fn(
        diffusion_pred=torch.zeros(B, 4),
        diffusion_target=torch.zeros(B, 4),
        depth_target=depth_norm,
        depth_range=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        depth_prior_norm=depth_norm,
        depth_focus_norm=evidence["depth_focus_norm"],
        depth_final_norm=depth_norm,
        focus_posterior=evidence["focus_posterior"],
        focus_entropy=evidence["focus_entropy"],
        focus_reliability=evidence["focus_reliability"],
        focus_distances=focus_distances,
        gate_focus=torch.rand(B, 1, H, W),
        focal_stack=stack,
        rgb_pred=torch.randn(B, C, H, W),
    )

    assert torch.isfinite(focus_target).all()
    assert "loss_focus_posterior_kl" in loss_dict
    assert torch.isfinite(loss_dict["loss_focus_posterior_kl"])
    assert "loss_gate_focus" in loss_dict
    assert torch.isfinite(loss_dict["total"])


def test_physical_support_head_shapes_and_gate_normalization():
    support_inputs = torch.randn(2, 5, 32, 48)
    head = PhysicalSupportHead(in_channels=5, hidden=16)
    out = head(support_inputs)

    assert out["gate_logits"].shape == (2, 3, 32, 48)
    assert out["gate_focus"].shape == (2, 1, 32, 48)
    assert out["gate_prior"].shape == (2, 1, 32, 48)
    assert out["gate_abstain"].shape == (2, 1, 32, 48)
    assert out["uncertainty_final"].shape == (2, 1, 32, 48)
    assert out["physical_support"].shape == (2, 1, 32, 48)
    gate_sum = out["gate_focus"] + out["gate_prior"] + out["gate_abstain"]
    assert torch.allclose(gate_sum, torch.ones_like(gate_sum), atol=1e-5)


def test_build_support_inputs_shapes():
    B, N, H, W = 2, 5, 32, 48
    focus_posterior = torch.softmax(torch.randn(B, N, H, W), dim=1)
    focus_entropy = torch.rand(B, 1, H, W)
    depth_focus = torch.rand(B, 1, H, W)
    depth_prior = torch.rand(B, 1, H, W)
    uncertainty_decoder = torch.rand(B, 1, H, W)

    support_inputs, support_maps = build_support_inputs(
        focus_posterior,
        focus_entropy,
        depth_focus,
        depth_prior,
        uncertainty_decoder,
    )

    assert support_inputs.shape == (B, 5, H, W)
    assert support_maps["focus_peakiness"].shape == (B, 1, H, W)
    assert support_maps["posterior_margin"].shape == (B, 1, H, W)
    assert support_maps["depth_disagreement"].shape == (B, 1, H, W)
