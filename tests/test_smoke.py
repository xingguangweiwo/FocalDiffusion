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
