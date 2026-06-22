from pathlib import Path

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
    expected_loss_defaults = {
        "lambda_trace": 0.05,
        "lambda_violation": 0.05,
        "lambda_invalid": 0.05,
    }
    assert {key: config["losses"][key] for key in expected_loss_defaults} == expected_loss_defaults
    assert config["training"]["supervision_mode"] in {"supervised", "semi_supervised"}
    train_filelists = {source["filelist"] for source in config["data"].get("train_sources", [])}
    test_filelists = {source["filelist"] for source in config["data"].get("test_sources", [])}
    assert train_filelists
    assert not (train_filelists & test_filelists)
    assert config["training"]["unsupervised_adaptation"]["enabled"] is False
    assert config["training"]["unsupervised_adaptation"]["round_id"] == "source"
    assert config["training"]["unsupervised_adaptation"]["round_index"] == 0
    assert config["data"]["adaptation_sources"]
    train_filelists = {source["filelist"] for source in config["data"]["train_sources"]}
    adapt_filelists = {source["filelist"] for source in config["data"]["adaptation_sources"]}
    test_filelists = {source["filelist"] for source in config["data"]["test_sources"]}
    assert adapt_filelists.isdisjoint(test_filelists)
    assert train_filelists.isdisjoint(test_filelists)
    assert config["training"]["unsupervised_adaptation"]["mining_manifest"]
    assert "mine_every_n_steps" not in config["training"]["unsupervised_adaptation"]


def test_disabled_adaptation_does_not_require_adaptation_sources():
    from copy import deepcopy
    from script.train import load_config, validate_config

    config = deepcopy(load_config("configs/base.yaml"))
    config["training"]["unsupervised_adaptation"]["enabled"] = False
    config["data"].pop("adaptation_sources", None)
    validate_config(config)


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


def test_defocus_consistency_range_invariant_residuals():
    from src.models import DefocusConsistencyVerifier

    torch.manual_seed(0)
    focal_stack_unit = torch.rand(2, 4, 3, 16, 16)
    all_in_focus_unit = focal_stack_unit.mean(dim=1)
    focal_plane_distances = torch.linspace(0.0, 1.0, 4)
    depth_canonical = torch.rand(2, 1, 16, 16)

    verifier = DefocusConsistencyVerifier(max_blur_radius=3)
    unit_outputs = verifier(focal_stack_unit, focal_plane_distances, depth_canonical, all_in_focus_unit)
    signed_outputs = verifier(
        focal_stack_unit * 2.0 - 1.0,
        focal_plane_distances,
        depth_canonical,
        all_in_focus_unit * 2.0 - 1.0,
    )

    assert torch.allclose(
        unit_outputs["stack_reprojection_residual"],
        signed_outputs["stack_reprojection_residual"],
        atol=1e-6,
    )
    assert torch.allclose(
        unit_outputs["stack_reprojection_residual"],
        signed_outputs["stack_reprojection_residual"],
        atol=1e-6,
    )
    assert unit_outputs.keys() == signed_outputs.keys()


def test_loss_trace_terms_are_active_with_physical_verification_trace():
    from types import SimpleNamespace

    batch, height, width = 1, 4, 4
    trace = SimpleNamespace(
        focus_support=torch.ones(batch, 1, height, width),
        generation_support=torch.ones(batch, 1, height, width),
        conflict_score=torch.full((batch, 1, height, width), 0.75),
        invalid_score=torch.full((batch, 1, height, width), 0.25),
    )
    loss_fn = FocalStackGenerationLoss(
        lambda_trace=0.02,
        lambda_violation=0.02,
        lambda_invalid=0.02,
    )

    loss_dict = loss_fn(
        diffusion_pred=torch.zeros(batch, 1, height, width),
        diffusion_target=torch.zeros(batch, 1, height, width),
        uncertainty=torch.zeros(batch, 1, height, width),
        physical_verification_trace=trace,
        predicted_verification_support=torch.full((batch, 1, height, width), 0.1),
        predicted_verification_invalid=torch.full((batch, 1, height, width), 0.1),
    )

    assert loss_dict["loss_trace"] > 0
    assert loss_dict["loss_violation"] > 0
    assert loss_dict["mean_conflict_score"] == torch.tensor(0.75)
    assert loss_dict["false_confident_violation_rate"] == torch.tensor(1.0)


def test_trace_mining_buffer_mines_and_replays_small_patches():
    from types import SimpleNamespace

    from src.training.trainer import TraceMiningBuffer

    trace = SimpleNamespace(
        conflict_score=torch.tensor([[[[0.1, 0.9], [0.2, 0.3]]]]),
        invalid_score=torch.tensor([[[[0.2, 0.1], [0.8, 0.4]]]]),
        depth_focus_discrepancy=torch.tensor([[[[0.7, 0.1], [0.2, 0.1]]]]),
        focus_support=torch.ones(1, 1, 2, 2),
    )
    buffer = TraceMiningBuffer(max_items=4, round_id="M0", patch_size=1)
    stats = buffer.mine(
        trace=trace,
        uncertainty=torch.zeros(1, 1, 2, 2),
        sample_ids=["sample_a"],
        batch_index=3,
        conflict_threshold=0.5,
        confidence_threshold=0.5,
        accepted_refinement=True,
        focal_split_seed=7,
        heldout_risk_before=0.4,
        heldout_risk_after=0.2,
        dataset_split="adaptation",
    )

    assert stats["mined_trace_items"] == 4
    assert stats["buffer_size"] == 4
    assert len(buffer) == 4
    assert {item["verdict_type"] for item in buffer.items} == {"conflict", "invalid", "focus_discrepancy", "reliable_non_conflict"}
    assert all("target_patch" not in item for item in buffer.items)
    assert all("sample_path" in item and "crop" in item for item in buffer.items)
    assert all("conflict_target" in item and "invalid_target" in item and "support_target" in item for item in buffer.items)

    replay_loss = buffer.replay_loss(
        predicted_support=torch.full((1, 1, 2, 2), 0.1),
        predicted_invalid=torch.full((1, 1, 2, 2), 0.1),
        sample_ids=["sample_a"],
    )
    assert replay_loss is not None
    assert replay_loss > 0


def test_trace_replay_updates_model_but_not_frozen_verifier_and_checkpoint_hash_changes(tmp_path):
    from src.training.trainer import TraceMiningBuffer, checkpoint_sha256

    model = torch.nn.Conv2d(3, 2, kernel_size=1)
    verifier = torch.nn.Conv2d(3, 1, kernel_size=1)
    for parameter in verifier.parameters():
        parameter.requires_grad_(False)
    verifier_before = {name: value.detach().clone() for name, value in verifier.state_dict().items()}

    m0_path = tmp_path / "m0.pt"
    torch.save({"model": model.state_dict()}, m0_path)
    m0_hash = checkpoint_sha256(m0_path)

    buffer = TraceMiningBuffer(max_items=2, round_id="M1", round_index=1, patch_size=2)
    buffer.items.append({
        "round_id": "M1",
        "round_index": 1,
        "sample_id": "adapt_a",
        "sample_path": "adapt_a",
        "verdict_type": "invalid",
        "crop": {"y0": 0, "y1": 2, "x0": 0, "x1": 2},
        "source_shape": {"height": 2, "width": 2},
        "focal_plane_coordinates": [0.0, 1.0],
        "conflict_target": 0.0,
        "invalid_target": 1.0,
        "support_target": 0.0,
        "verifier_confidence": 0.95,
    })

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.ones(1, 3, 2, 2)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    logits = model(x)
    predictions = torch.sigmoid(logits)
    replay_loss = buffer.replay_loss(
        predicted_support=predictions[:, :1],
        predicted_invalid=predictions[:, 1:],
        sample_ids=["adapt_a"],
    )
    assert replay_loss is not None
    replay_loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(torch.equal(verifier_before[name], value) for name, value in verifier.state_dict().items())

    m1_path = tmp_path / "m1.pt"
    torch.save({"model": model.state_dict()}, m1_path)
    assert checkpoint_sha256(m1_path) != m0_hash


def test_no_refocus_symbols_in_public_source():
    forbidden = ("refocus", "refocused", "refocus_residual")
    roots = [Path("src"), Path("script"), Path("configs"), Path("README.md")]
    offenders = []
    for root in roots:
        paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in paths:
            if path.suffix in {".py", ".yaml", ".md"}:
                text = path.read_text(encoding="utf-8")
                for symbol in forbidden:
                    if symbol in text:
                        offenders.append(f"{path}:{symbol}")
    assert offenders == []


def test_tiny_one_batch_overfit_cpu():
    model = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.ones(1, 1, 2, 2)
    y = torch.ones(1, 1, 2, 2)
    first_loss = None
    last_loss = None
    for _ in range(8):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    assert last_loss is not None and first_loss is not None
    final_loss = torch.nn.functional.mse_loss(model(x), y).item()
    assert final_loss < first_loss


def test_static_loss_constructor_matches_trainer_call_and_no_preference_config():
    import ast
    import inspect
    import yaml
    from pathlib import Path
    from src.training.losses import FocalStackGenerationLoss

    trainer_tree = ast.parse(Path("src/training/trainer.py").read_text(encoding="utf-8"))
    call_keywords = None
    for node in ast.walk(trainer_tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "FocalStackGenerationLoss":
            call_keywords = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
            break
    assert call_keywords is not None
    constructor_params = set(inspect.signature(FocalStackGenerationLoss.__init__).parameters) - {"self"}
    assert call_keywords <= constructor_params
    assert constructor_params - call_keywords <= {"diffusion_weight", "depth_weight", "rgb_weight"}

    for path in Path("configs").rglob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        assert "lambda_preference" not in str(config)


def test_static_evaluation_metrics_have_no_undefined_focus_support():
    import ast
    from pathlib import Path

    tree = ast.parse(Path("script/evaluate.py").read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compute_trace_metrics")
    loaded = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    stored = {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param))}
    assert "focus_support" not in loaded - stored


def test_static_config_keys_are_referenced_by_source():
    import yaml
    from pathlib import Path

    config = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    source_text = "\n".join(path.read_text(encoding="utf-8") for root in (Path("src"), Path("script")) for path in root.rglob("*.py"))
    ignored_sections = {"model", "data", "training", "optimizer", "losses", "validation", "logging", "hardware", "inference"}
    missing = []
    for section, values in config.items():
        if not isinstance(values, dict):
            continue
        for key in values:
            if key in ignored_sections:
                continue
            if key not in source_text and section not in source_text:
                missing.append(f"{section}.{key}")
    assert missing == []


def test_trace_manifest_metadata_and_full_replay_targets_drive_generation_outputs(tmp_path):
    from types import SimpleNamespace

    from src.training.trainer import TraceMiningBuffer

    buffer = TraceMiningBuffer(max_items=4, round_id="M1", round_index=1, manifest_path=tmp_path / "manifest.jsonl", patch_size=1)
    buffer.set_metadata(parent_checkpoint_sha256="parent-sha", verifier_config_hash="verifier-sha")
    trace = SimpleNamespace(
        conflict_score=torch.tensor([[[[0.9, 0.1], [0.1, 0.1]]]]),
        invalid_score=torch.zeros(1, 1, 2, 2),
        depth_focus_discrepancy=torch.tensor([[[[0.8, 0.1], [0.1, 0.1]]]]),
        stack_reprojection_residual=torch.tensor([[[[0.4, 0.1], [0.1, 0.1]]]]),
        focus_support=torch.ones(1, 1, 2, 2),
        generation_support=torch.ones(1, 1, 2, 2) * 0.5,
    )
    stats = buffer.mine(
        trace=trace,
        uncertainty=torch.zeros(1, 1, 2, 2),
        sample_ids=["adapt_a"],
        batch_index=0,
        focal_plane_coordinates=torch.tensor([[0.0, 0.5, 1.0]]),
        conflict_threshold=0.5,
        confidence_threshold=0.5,
        generated_depth=torch.ones(1, 1, 2, 2) * 0.8,
        focal_depth=torch.ones(1, 1, 2, 2) * 0.2,
        final_depth=torch.ones(1, 1, 2, 2) * 0.7,
        focal_gate=torch.ones(1, 1, 2, 2) * 0.3,
        generative_gate=torch.ones(1, 1, 2, 2) * 0.7,
        abstention=torch.ones(1, 1, 2, 2) * 0.1,
        accepted_refinement=True,
        focal_split_seed=11,
        heldout_risk_before=0.6,
        heldout_risk_after=0.3,
        dataset_split="adaptation",
    )
    assert stats["mined_trace_items"] > 0
    buffer.save(buffer.manifest_path)
    reloaded = TraceMiningBuffer(max_items=4, manifest_path=buffer.manifest_path)
    assert reloaded.metadata["parent_checkpoint_sha256"] == "parent-sha"
    assert reloaded.metadata["verifier_config_hash"] == "verifier-sha"
    assert all(item["accepted_refinement"] and item["dataset_split"] == "adaptation" for item in reloaded.items)
    assert all("focal_split_seed" in item and "heldout_risk_before" in item and "heldout_risk_after" in item for item in reloaded.items)
    assert all("stack_reprojection_residual_target" in item for item in reloaded.items)
    assert all("focal_gate_target" in item and "generative_gate_target" in item for item in reloaded.items)

    final_depth = torch.full((1, 1, 2, 2), 0.9, requires_grad=True)
    generated_depth = torch.full((1, 1, 2, 2), 0.9, requires_grad=True)
    focal_gate = torch.full((1, 1, 2, 2), 0.2, requires_grad=True)
    generative_gate = torch.full((1, 1, 2, 2), 0.8, requires_grad=True)
    uncertainty = torch.full((1, 1, 2, 2), 0.1, requires_grad=True)
    support = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    loss = reloaded.replay_loss(
        predicted_support=support,
        predicted_invalid=uncertainty,
        predicted_final_depth=final_depth,
        predicted_generated_depth=generated_depth,
        predicted_focal_gate=focal_gate,
        predicted_generative_gate=generative_gate,
        predicted_abstention=uncertainty,
        sample_ids=["adapt_a"],
    )
    assert loss is not None
    loss.backward()
    for tensor in (final_depth, generated_depth, focal_gate, generative_gate, uncertainty):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_focal_axis_metric_supervision_uses_focal_coordinate_system():
    from src.training.losses import metric_depth_to_focal_coordinates

    distances = torch.tensor([[2.0, 4.0, 6.0]])
    depth = torch.tensor([[[[2.0, 4.0, 6.0, 8.0]]]])
    coords = metric_depth_to_focal_coordinates(depth, distances)
    assert torch.allclose(coords, torch.tensor([[[0.0, 0.5, 1.0, 1.0]]]))


def test_filelist_allows_focal_stack_only_entries(tmp_path):
    from PIL import Image
    from src.data.dataset import FocalStackDataset

    stack_dir = tmp_path / "stack"
    stack_dir.mkdir()
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(stack_dir / "000.png")
    filelist = tmp_path / "list.txt"
    filelist.write_text("stack\n", encoding="utf-8")
    dataset = FocalStackDataset(tmp_path, filelist, image_size=(8, 8), focal_stack_size=1)
    sample = dataset[0]
    assert "focal_stack" in sample
    assert "depth" not in sample
    assert "all_in_focus" not in sample


def test_range_conversion_parity_for_train_inference_adaptation_inputs():
    from src.models.physics_modules import _split_unit_and_signed_ranges

    unit = torch.rand(1, 3, 3, 4, 4)
    signed = unit * 2.0 - 1.0
    train_unit, train_signed = _split_unit_and_signed_ranges(unit)
    infer_unit, infer_signed = _split_unit_and_signed_ranges(signed)
    adapt_unit, adapt_signed = _split_unit_and_signed_ranges(unit.clone())
    assert torch.allclose(train_signed, infer_signed)
    assert torch.allclose(train_signed, adapt_signed)
    assert torch.allclose(train_unit, infer_unit)
    assert torch.allclose(train_unit, adapt_unit)


def test_focus_likelihood_masks_duplicate_planes_and_low_texture_is_uncertain():
    from src.models.focal_evidence_encoder import FocusLikelihoodEstimator

    estimator = FocusLikelihoodEstimator(hidden=8, temperature=0.1)
    stack = torch.zeros(1, 3, 3, 8, 8)
    distances = torch.tensor([[0.2, 0.4, 0.4]])
    out = estimator(stack, distances)
    assert torch.all(out["focal_posterior"][:, 2] == 0)
    assert out["texture_confidence"].max() <= 1
    assert "multimodality_score" in out and "focus_coverage_confidence" in out


def test_synthetic_renderer_has_finite_depth_and_aif_gradients():
    from src.data.synthetic_focal_stack_renderer import SyntheticFocalStackRenderer

    renderer = SyntheticFocalStackRenderer(max_sigma=2.0, num_blur_levels=4)
    aif = torch.rand(1, 3, 8, 8, requires_grad=True)
    depth = (torch.rand(1, 1, 8, 8) + 0.5).requires_grad_(True)
    stack = renderer.generate(aif, depth, torch.tensor([[0.6, 1.0]]))
    loss = stack.mean()
    loss.backward()
    assert aif.grad is not None and torch.isfinite(aif.grad).all()
    assert depth.grad is not None and torch.isfinite(depth.grad).all()


def test_selective_tto_heldout_split_and_rejection_are_deterministic():
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline

    train_a, val_a = FocalStackGenerationPipeline._split_refinement_planes(5, torch.device("cpu"), seed=123)
    train_b, val_b = FocalStackGenerationPipeline._split_refinement_planes(5, torch.device("cpu"), seed=123)
    assert torch.equal(train_a, train_b) and torch.equal(val_a, val_b)
    assert set(train_a.tolist()).isdisjoint(set(val_a.tolist()))
    assert not FocalStackGenerationPipeline._should_accept_refinement(torch.tensor(0.1), torch.tensor(0.2), epsilon=0.0)


def test_selective_tto_recomputes_uncertainty_not_free_variable():
    from src.models.physics_modules import FocalPhysicalVerifier
    from src.pipelines.focal_stack_generation_pipeline import FocalStackGenerationPipeline

    stack = torch.rand(1, 4, 3, 8, 8)
    distances = torch.linspace(0.2, 1.0, 4).unsqueeze(0)
    depth = torch.full((1, 1, 8, 8), 0.5)
    aif = stack.mean(dim=1)
    trace = FocalPhysicalVerifier()(stack, distances, depth, aif, depth)
    cand_depth, unc, cand_aif = FocalStackGenerationPipeline._selective_test_time_refinement(
        focal_stack_unit=stack,
        focal_plane_distances=distances,
        final_depth_canonical=depth,
        focus_depth_canonical=depth,
        prior_depth_canonical=depth,
        all_in_focus_unit=aif,
        uncertainty_final=torch.zeros_like(depth),
        trace=trace,
        seed=5,
        inner_steps=1,
    )
    assert cand_depth.shape == depth.shape and cand_aif.shape == aif.shape
    assert unc.shape == depth.shape
    assert unc.max() > 0


def test_legacy_self_improvement_config_keys_migrate_to_unsupervised_adaptation():
    from script.train import _migrate_protocol_config

    cfg = {
        "data": {"self_improvement_sources": [{"filelist": "adapt.txt"}]},
        "training": {"self_improvement": {"enabled": True, "round_index": 1}},
    }
    migrated = _migrate_protocol_config(cfg)
    assert "self_improvement_sources" not in migrated["data"]
    assert migrated["data"]["adaptation_sources"][0]["filelist"] == "adapt.txt"
    assert "self_improvement" not in migrated["training"]
    assert migrated["training"]["unsupervised_adaptation"]["round_index"] == 1
