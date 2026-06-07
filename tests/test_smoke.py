import torch
from PIL import Image

from script.inference import resize_or_pad_to_multiple
from src.models import DualOutputDecoder, FocalStackProcessor
from src.pipelines.focal_diffusion_pipeline import FocalDiffusionPipeline
from src.training.losses import FocalDiffusionLoss


def test_core_smoke():
    focal_stack = torch.rand(1, 5, 3, 128, 128)
    focus_distances = torch.linspace(0, 1, 5).unsqueeze(0)
    processor = FocalStackProcessor(
        feature_dim=64,
        max_sequence_length=100,
        focal_encoder_type="focal_sweep",
        patch_size=8,
        focal_attention_heads=8,
        focal_attention_depth=1,
    )
    features = processor(focal_stack * 2 - 1, focus_distances)
    assert "fused_features" in features

    decoder = DualOutputDecoder(in_channels=16, out_channels_rgb=16)
    decoder_out = decoder(torch.randn(1, 16, 16, 16))
    assert "depth_prior_norm" in decoder_out and "uncertainty" in decoder_out and "aif_latents" in decoder_out

    loss_fn = FocalDiffusionLoss(diffusion_weight=1.0, depth_weight=1.0, rgb_weight=0.0)
    loss_dict = loss_fn(
        diffusion_pred=torch.randn(1, 16, 16, 16),
        diffusion_target=torch.randn(1, 16, 16, 16),
        depth_target=torch.rand(1, 1, 16, 16),
        depth_prior_norm=decoder_out["depth_prior_norm"],
        depth_final_norm=decoder_out["depth_prior_norm"],
        uncertainty=decoder_out["uncertainty"],
        focal_stack=focal_stack * 2 - 1,
        depth_range=torch.tensor([[0.2, 4.0]], dtype=torch.float32),
        depth_mask=torch.ones(1, 16, 16),
    )
    assert torch.isfinite(loss_dict["total"])
    assert "loss_flow_matching" in loss_dict


def test_focal_sweep_n100_smoke():
    focal_stack = torch.rand(1, 100, 3, 16, 16)
    focus_distances = torch.linspace(0, 1, 100).unsqueeze(0)
    processor = FocalStackProcessor(
        feature_dim=16,
        max_sequence_length=100,
        focal_encoder_type="focal_sweep",
        patch_size=8,
        focal_attention_heads=4,
        focal_attention_depth=1,
    )
    features = processor(focal_stack * 2 - 1, focus_distances)
    assert "fused_features" in features


def test_pipeline_size_inference_helper():
    h, w = FocalDiffusionPipeline._make_divisible_size(240, 320, divisor=16)
    assert (h, w) == (240, 320)
    assert h != w

    h2, w2 = FocalDiffusionPipeline._make_divisible_size(375, 500, divisor=16)
    assert h2 % 16 == 0 and w2 % 16 == 0
    assert h2 >= 375 and w2 >= 500

    h3, w3 = FocalDiffusionPipeline._make_divisible_size(512, 768, divisor=16)
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


def test_expected_metric_depth_from_focus_posterior_shape_and_finite():
    from src.models.focal_evidence import expected_metric_depth_from_focus_posterior

    posterior = torch.softmax(torch.randn(2, 5, 4, 3), dim=1)
    distances = torch.tensor([0.3, 0.5, 0.8, 1.5, 3.0])
    depth = expected_metric_depth_from_focus_posterior(posterior, distances)
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
                "physical_support_hidden": 12,
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
