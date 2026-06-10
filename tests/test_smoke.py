import torch
from PIL import Image

from script.inference import resize_or_pad_to_multiple
from src.models import (
    TaskOutputDecoder,
    FocalEvidenceEncoder,
    FocalStackProcessor,
    PhysicalEvidenceEstimator,
    build_physical_evidence_features,
)
from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline
from src.training.losses import FocalStackGenerationLoss


def test_core_smoke():
    focal_stack = torch.rand(1, 5, 3, 128, 128)
    focal_plane_distances = torch.linspace(0, 1, 5).unsqueeze(0)
    processor = FocalStackProcessor(
        feature_dim=64,
        max_sequence_length=100,
        focal_encoder_type="focal_sweep",
        patch_size=8,
        focal_attention_heads=8,
        focal_attention_depth=1,
    )
    features = processor(focal_stack * 2 - 1, focal_plane_distances)
    assert "fused_features" in features

    decoder = TaskOutputDecoder(in_channels=16, out_channels_rgb=16)
    decoder_out = decoder(torch.randn(1, 16, 16, 16))
    assert "generated_depth_canonical" in decoder_out and "uncertainty" in decoder_out and "all_in_focus_latents" in decoder_out

    loss_fn = FocalStackGenerationLoss(diffusion_weight=1.0, depth_weight=1.0, rgb_weight=0.0)
    loss_dict = loss_fn(
        diffusion_pred=torch.randn(1, 16, 16, 16),
        diffusion_target=torch.randn(1, 16, 16, 16),
        depth_target=torch.rand(1, 1, 16, 16),
        generated_depth_canonical=decoder_out["generated_depth_canonical"],
        final_depth_canonical=decoder_out["generated_depth_canonical"],
        uncertainty=decoder_out["uncertainty"],
        focal_stack=focal_stack * 2 - 1,
        depth_range=torch.tensor([[0.2, 4.0]], dtype=torch.float32),
        depth_mask=torch.ones(1, 16, 16),
    )
    assert torch.isfinite(loss_dict["total"])
    assert "loss_flow_matching" in loss_dict


def test_focal_sweep_n100_smoke():
    focal_stack = torch.rand(1, 100, 3, 16, 16)
    focal_plane_distances = torch.linspace(0, 1, 100).unsqueeze(0)
    processor = FocalStackProcessor(
        feature_dim=16,
        max_sequence_length=100,
        focal_encoder_type="focal_sweep",
        patch_size=8,
        focal_attention_heads=4,
        focal_attention_depth=1,
    )
    features = processor(focal_stack * 2 - 1, focal_plane_distances)
    assert "fused_features" in features


def test_pipeline_size_inference_helper():
    h, w = FocalStackGenerationPipeline._make_divisible_size(240, 320, divisor=16)
    assert (h, w) == (240, 320)
    assert h != w

    h2, w2 = FocalStackGenerationPipeline._make_divisible_size(375, 500, divisor=16)
    assert h2 % 16 == 0 and w2 % 16 == 0
    assert h2 >= 375 and w2 >= 500

    h3, w3 = FocalStackGenerationPipeline._make_divisible_size(512, 768, divisor=16)
    assert (h3, w3) == (512, 768)

    images = [Image.new("RGB", (500, 375), color=(128, 128, 128)) for _ in range(3)]
    padded, meta = resize_or_pad_to_multiple(images, divisor=16)
    assert meta["original_size"] == [375, 500]
    assert meta["inference_size"][0] % 16 == 0
    assert meta["inference_size"][1] % 16 == 0
    assert padded[0].size == (meta["inference_size"][1], meta["inference_size"][0])


def test_load_config_base_smoke():
    from script.train import load_config

    config = load_config("configs/base.yaml")
    assert config["model"]["feature_dim"] == 128
    assert config["data"]["dataset_kwargs"]["strict_data"] is True


def test_dataset_strict_data_raises_on_missing_files(tmp_path):
    import pytest

    from src.data.dataset import FocalStackDataset

    filelist = tmp_path / "missing.txt"
    filelist.write_text("missing_stack missing_depth.png\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Dropped 1 samples due to missing files"):
        FocalStackDataset(data_root=tmp_path, filelist_path=filelist, strict_data=True)


def test_decode_metric_depth_from_focal_posterior_shape_and_finite():
    from src.models.focal_evidence_encoder import decode_metric_depth_from_focal_posterior

    posterior = torch.softmax(torch.randn(2, 5, 4, 3), dim=1)
    distances = torch.tensor([0.3, 0.5, 0.8, 1.5, 3.0])
    depth = decode_metric_depth_from_focal_posterior(posterior, distances)
    assert depth.shape == (2, 1, 4, 3)
    assert torch.isfinite(depth).all()


def test_build_focal_modules_from_config_feature_dim_128():
    from types import SimpleNamespace

    from src.pipelines.pipeline_utils import _build_focal_modules_from_config

    vae = SimpleNamespace(config=SimpleNamespace(latent_channels=16))
    modules = _build_focal_modules_from_config(
        {
            "model": {
                "feature_dim": 128,
                "max_focal_stack_size": 20,
                "focal_encoder_type": "focal_sweep",
                "patch_size": 8,
                "focal_attention_heads": 8,
                "focal_attention_depth": 1,
                "focal_evidence_hidden": 32,
                "focal_evidence_temperature": 0.1,
                "physical_evidence_support_hidden": 12,
            }
        },
        vae,
    )
    assert modules["focal_processor"].feature_dim == 128
    assert modules["focal_processor"].max_sequence_length == 20
    assert modules["focal_evidence_head"].hidden == 32


def test_reliability_metric_helpers_smoke():
    import importlib.util
    import pytest

    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed")

    from script.evaluate_reliability import evaluate_scores

    depth_gt = torch.linspace(1.0, 2.0, 20).numpy()
    depth_pred = depth_gt.copy()
    depth_pred[-5:] += 0.5
    uncertainty = torch.linspace(0.0, 1.0, 20).numpy()
    results = evaluate_scores(
        {
            "depth_pred": depth_pred,
            "depth_gt": depth_gt,
            "valid_mask": torch.ones(20, dtype=torch.bool).numpy(),
            "uncertainty_final": uncertainty,
        },
        high_error_mode="top_percent",
        high_error_percent=20,
    )
    assert "uncertainty_final" in results
    assert "auroc" in results["uncertainty_final"]
    assert "ause_absrel" in results["uncertainty_final"]


def test_build_coc_posterior_targets_smoke():
    from src.training.losses import build_coc_posterior_targets

    batch, planes, height, width = 2, 4, 8, 8
    depth = torch.rand(batch, 1, height, width) * 3.0 + 0.3
    focal_plane_distances = torch.linspace(0.4, 3.0, planes).unsqueeze(0).expand(batch, -1)
    camera_params = {
        "focal_length": torch.full((batch,), 0.05),
        "f_number": torch.full((batch, 1), 2.8),
        "pixel_size": torch.tensor(5e-6),
    }

    posterior, coc_pixels = build_coc_posterior_targets(
        depth,
        focal_plane_distances,
        camera_params,
        temperature=1.0,
    )

    assert posterior.shape == (batch, planes, height, width)
    assert coc_pixels.shape == (batch, planes, height, width)
    assert torch.allclose(posterior.sum(dim=1), torch.ones(batch, height, width), atol=1e-5)
    assert torch.isfinite(posterior).all()
    assert torch.isfinite(coc_pixels).all()


def test_focal_diffusion_loss_coc_target_smoke():
    batch, planes, height, width = 2, 4, 8, 8
    loss_fn = FocalStackGenerationLoss(
        diffusion_weight=1.0,
        depth_weight=1.0,
        focal_posterior_kl_weight=0.2,
        focal_target_type="coc",
        coc_posterior_temperature=1.0,
    )
    focal_posterior = torch.softmax(torch.randn(batch, planes, height, width), dim=1)
    depth_target = torch.rand(batch, 1, height, width) * 3.0 + 0.3
    loss_dict = loss_fn(
        diffusion_pred=torch.randn(batch, 4, height, width),
        diffusion_target=torch.randn(batch, 4, height, width),
        depth_target=depth_target,
        final_depth_canonical=torch.rand(batch, 1, height, width),
        focal_posterior=focal_posterior,
        focal_plane_distances=torch.linspace(0.4, 3.0, planes).unsqueeze(0).expand(batch, -1),
        depth_range=torch.tensor([[0.3, 3.3], [0.3, 3.3]]),
        camera_params={
            "focal_length": torch.full((batch,), 0.05),
            "f_number": torch.full((batch,), 2.8),
            "pixel_size": torch.tensor(5e-6),
        },
    )

    assert torch.isfinite(loss_dict["total"])
    assert "loss_focal_posterior_kl" in loss_dict


def test_custom_module_half_inputs_match_half_weights_smoke():
    from src.models.focal_evidence_encoder import FocalEvidenceEncoder, PhysicalEvidenceEstimator, build_physical_evidence_features

    focal_stack = torch.rand(1, 3, 3, 16, 16, dtype=torch.float16)
    focal_plane_distances = torch.linspace(0.4, 2.0, 3, dtype=torch.float16).unsqueeze(0)
    evidence_head = FocalEvidenceEncoder(hidden=8).to(dtype=torch.float16)
    evidence = evidence_head(focal_stack, focal_plane_distances)
    assert evidence["focal_posterior"].dtype == torch.float16

    depth_prior = torch.rand(1, 1, 16, 16, dtype=torch.float16)
    support_inputs, _ = build_physical_evidence_features(
        focal_posterior=evidence["focal_posterior"],
        focal_entropy=evidence["focal_entropy"],
        focal_depth_canonical=evidence["focal_depth_canonical"],
        generated_depth_canonical=depth_prior,
        generative_uncertainty=torch.rand(1, 1, 16, 16, dtype=torch.float16),
    )
    support_head = PhysicalEvidenceEstimator(hidden=8).to(dtype=torch.float16)
    support = support_head(support_inputs.to(dtype=torch.float16))
    assert support["physical_evidence_support"].dtype == torch.float16


def test_register_focal_modules_updates_only_focal_modules():
    import torch.nn as nn

    from src.pipelines.pipeline_utils import _register_focal_modules

    class DummyPipeline:
        def __init__(self):
            self.focal_processor = nn.Conv2d(1, 1, 1)
            self.focal_evidence_head = nn.Conv2d(1, 1, 1)
            self.task_output_decoder = nn.Conv2d(1, 1, 1)
            self.physical_evidence_support_head = nn.Conv2d(1, 1, 1)
            self.text_encoder = nn.Linear(1, 1)
            self.vae = nn.Linear(1, 1)
            self.registered = None

        def register_modules(self, **modules):
            self.registered = modules

    pipeline = DummyPipeline()

    _register_focal_modules(pipeline, device="cpu", dtype=torch.float64)

    assert set(pipeline.registered) == {
        "focal_processor",
        "focal_evidence_head",
        "task_output_decoder",
        "physical_evidence_support_head",
    }
    for module in pipeline.registered.values():
        assert next(module.parameters()).dtype == torch.float64
    assert next(pipeline.text_encoder.parameters()).dtype == torch.float32
    assert next(pipeline.vae.parameters()).dtype == torch.float32


def test_pipeline_rejects_invalid_focal_distance_mode_before_inference():
    import pytest

    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline

    with pytest.raises(ValueError, match="focal_distance_mode"):
        FocalStackGenerationPipeline.__call__(
            object(),
            focal_stack=torch.empty(1, 2, 3, 8, 8),
            focal_plane_distances=torch.ones(1, 2),
            focal_distance_mode="calibrated",
        )


def test_run_validation_merges_teacher_forced_and_generative_metrics(monkeypatch):
    from src.training import validation

    def fake_teacher(trainer, epoch):
        assert trainer == "trainer"
        assert epoch == 3
        return {
            "teacher_forced_abs_rel": 1.0,
            "teacher_forced_rmse": 2.0,
            "teacher_forced_l1": 3.0,
        }

    def fake_generative(trainer, epoch):
        assert trainer == "trainer"
        assert epoch == 3
        return {
            "generative_abs_rel": 4.0,
            "generative_rmse": 5.0,
            "generative_l1": 6.0,
        }

    monkeypatch.setattr(validation, "run_teacher_forced_validation", fake_teacher)
    monkeypatch.setattr(validation, "run_generative_validation", fake_generative)

    metrics = validation.run_validation("trainer", 3)

    assert metrics["teacher_forced_abs_rel"] == 1.0
    assert metrics["generative_abs_rel"] == 4.0
    assert metrics["loss"] == metrics["generative_l1"]
    assert metrics["abs_rel"] == metrics["generative_abs_rel"]
    assert metrics["rmse"] == metrics["generative_rmse"]


# Core tensor-path smoke tests merged from tests/test_smoke_core.py.
def test_focal_evidence_encoder_forward():
    encoder = FocalEvidenceEncoder()
    focal_stack = torch.randn(2, 5, 3, 64, 64)
    focal_plane_distances = torch.linspace(0.2, 2.0, 5).repeat(2, 1)

    outputs = encoder(focal_stack, focal_plane_distances)

    for key in (
        "focal_logits",
        "focal_posterior",
        "focal_depth_canonical",
        "focal_entropy",
        "focal_peak_confidence",
        "focal_coordinates",
    ):
        assert key in outputs
    assert torch.allclose(
        outputs["focal_posterior"].sum(dim=1),
        torch.ones(2, 64, 64),
        atol=1e-5,
    )


def test_physical_evidence_estimator_forward():
    focal_posterior = torch.softmax(torch.randn(2, 5, 64, 64), dim=1)
    focal_entropy = torch.rand(2, 1, 64, 64)
    focal_depth_canonical = torch.rand(2, 1, 64, 64)
    generated_depth_canonical = torch.rand(2, 1, 64, 64)
    generative_uncertainty = torch.rand(2, 1, 64, 64)
    support_inputs, _ = build_physical_evidence_features(
        focal_posterior=focal_posterior,
        focal_entropy=focal_entropy,
        focal_depth_canonical=focal_depth_canonical,
        generated_depth_canonical=generated_depth_canonical,
        generative_uncertainty=generative_uncertainty,
    )
    estimator = PhysicalEvidenceEstimator()

    outputs = estimator(support_inputs)

    for key in (
        "focal_evidence_weight",
        "generative_prior_weight",
        "abstention_weight",
        "uncertainty_final",
        "physical_evidence_support",
    ):
        assert key in outputs
    gate_weights = torch.cat(
        [
            outputs["focal_evidence_weight"],
            outputs["generative_prior_weight"],
            outputs["abstention_weight"],
        ],
        dim=1,
    )
    assert torch.allclose(
        gate_weights.sum(dim=1),
        torch.ones(2, 64, 64),
        atol=1e-5,
    )


def test_task_output_decoder_forward():
    decoder = TaskOutputDecoder(in_channels=16)
    latent = torch.randn(2, 16, 16, 16)

    outputs = decoder(latent)

    for key in (
        "generated_depth_canonical",
        "uncertainty",
        "all_in_focus_latents",
    ):
        assert key in outputs
    assert outputs["generated_depth_canonical"].min() >= 0
    assert outputs["generated_depth_canonical"].max() <= 1


def test_loss_forward_minimal_supervised():
    loss_fn = FocalStackGenerationLoss()
    batch, planes, height, width = 2, 5, 16, 16
    focal_posterior = torch.softmax(torch.randn(batch, planes, height, width), dim=1)
    depth_target = torch.rand(batch, 1, height, width)

    loss_dict = loss_fn(
        diffusion_pred=torch.randn(batch, 4, height, width),
        diffusion_target=torch.randn(batch, 4, height, width),
        depth_target=depth_target,
        generated_depth_canonical=torch.rand(batch, 1, height, width),
        focal_depth_canonical=torch.rand(batch, 1, height, width),
        final_depth_canonical=torch.rand(batch, 1, height, width),
        uncertainty=torch.rand(batch, 1, height, width),
        focal_posterior=focal_posterior,
        focal_entropy=torch.rand(batch, 1, height, width),
        focal_plane_distances=torch.linspace(0.2, 2.0, planes).repeat(batch, 1),
        depth_range=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        focal_evidence_weight=torch.rand(batch, 1, height, width),
    )

    assert "total" in loss_dict
    assert torch.isfinite(loss_dict["total"])
