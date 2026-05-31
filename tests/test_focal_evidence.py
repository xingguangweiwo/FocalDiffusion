import torch

from src.models.focal_evidence import FocalEvidenceHead
from src.training.losses import FocalDiffusionLoss, build_focus_target_from_depth


def test_focal_evidence_shapes_probability_and_ranges():
    B, N, C, H, W = 2, 5, 3, 64, 80
    stack = torch.randn(B, N, C, H, W)
    tau = torch.linspace(0.3, 10.0, N).repeat(B, 1)

    out = FocalEvidenceHead()(stack, tau)

    assert out["focus_prob"].shape == (B, N, H, W)
    assert out["depth_focus"].shape == (B, 1, H, W)
    assert out["focus_entropy"].shape == (B, 1, H, W)
    assert out["focus_reliability"].shape == (B, 1, H, W)
    assert torch.allclose(out["focus_prob"].sum(dim=1), torch.ones(B, H, W), atol=1e-4)
    assert out["depth_focus"].min() >= 0
    assert out["depth_focus"].max() <= 1
    assert out["focus_entropy"].min() >= 0
    assert out["focus_entropy"].max() <= 1
    assert out["focus_reliability"].min() >= 0
    assert out["focus_reliability"].max() <= 1


def test_pipeline_output_dataclass_exposes_focal_evidence_fields():
    from src.pipelines.focal_diffusion_pipeline import FocalDiffusionOutput

    out = FocalDiffusionOutput(depth_map=torch.zeros(1, 8, 8), all_in_focus_image=torch.zeros(1, 3, 8, 8))
    for name in (
        "depth_prior",
        "depth_focus",
        "depth_final",
        "focus_entropy",
        "focus_reliability",
        "uncertainty_final",
    ):
        assert hasattr(out, name)


def test_focus_posterior_kl_loss_is_finite():
    B, N, C, H, W = 2, 5, 3, 16, 16
    stack = torch.randn(B, N, C, H, W)
    focus_distances = torch.linspace(0.3, 10.0, N).repeat(B, 1)
    evidence = FocalEvidenceHead(hidden=16)(stack, focus_distances)
    depth_norm = torch.rand(B, 1, H, W)
    focus_target, _ = build_focus_target_from_depth(depth_norm, focus_distances)

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
        shape_norm=depth_norm,
        depth_prior=depth_norm,
        depth_focus=evidence["depth_focus"],
        depth_final=depth_norm,
        focus_prob=evidence["focus_prob"],
        focus_entropy=evidence["focus_entropy"],
        focus_reliability=evidence["focus_reliability"],
        focus_distances=focus_distances,
        focal_stack=stack,
        rgb_pred=torch.randn(B, C, H, W),
    )

    assert torch.isfinite(focus_target).all()
    assert "loss_focus_posterior_kl" in loss_dict
    assert torch.isfinite(loss_dict["loss_focus_posterior_kl"])
    assert torch.isfinite(loss_dict["total"])
