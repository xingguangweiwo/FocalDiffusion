import torch
from PIL import Image
from src.training.losses import FocusConsistencyCritic, FocalDiffusionLoss
from src.models import FocalStackProcessor, DualOutputDecoder
from src.pipelines.focal_diffusion_pipeline import FocalDiffusionPipeline
from script.inference import resize_or_pad_to_multiple

def test_core_smoke():
    focal_stack = torch.rand(1, 5, 3, 256, 256)
    focus_distances = torch.linspace(0, 1, 5).unsqueeze(0)
    processor = FocalStackProcessor(feature_dim=128,max_sequence_length=20,focal_encoder_type="focal_sweep",patch_size=8,focal_attention_heads=8,focal_attention_depth=2)
    features = processor(focal_stack * 2 - 1, focus_distances)
    assert "fused_features" in features

    decoder = DualOutputDecoder(in_channels=16,out_channels_rgb=16)
    out = decoder(torch.randn(1, 16, 32, 32))
    assert "shape_norm" in out and "uncertainty" in out and "aif_latents" in out

    critic = FocusConsistencyCritic()
    critic_out = critic(focal_stack * 2 - 1, focus_distances, out["shape_norm"])
    assert "focus_energy" in critic_out and "tau_contrast" in critic_out and "stack_contrast" in critic_out and "shape_candidate_contrast" in critic_out

    loss_fn = FocalDiffusionLoss(
        diffusion_weight=1.0,
        depth_weight=1.0,
        rgb_weight=0.0,
        consistency_weight=0.1,
        perceptual_weight=0.1,
        depth_gradient_weight=0.1,
        edge_consistency_weight=0.1,
        confidence_regularization_weight=0.1,
    )
    loss_dict = loss_fn(
        diffusion_pred=torch.randn(1,16,32,32),
        diffusion_target=torch.randn(1,16,32,32),
        depth_target=torch.rand(1, 1, 32, 32),
        shape_norm=out["shape_norm"],
        uncertainty=out["uncertainty"],
        focal_stack=focal_stack * 2 - 1,
        critic_outputs=critic_out,
        depth_range=torch.tensor([[0.2, 4.0]], dtype=torch.float32),
        depth_mask=torch.ones(1, 32, 32),
        focal_features=features,
        confidence_map=out["uncertainty"],
    )
    assert "total" in loss_dict


def test_focal_sweep_n20_smoke():
    focal_stack = torch.rand(1, 20, 3, 128, 128)
    focus_distances = torch.linspace(0, 1, 20).unsqueeze(0)
    processor = FocalStackProcessor(
        feature_dim=128,
        max_sequence_length=20,
        focal_encoder_type="focal_sweep",
        patch_size=8,
        focal_attention_heads=8,
        focal_attention_depth=2,
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
