import torch
from src.training.losses import FocusConsistencyCritic, FocalDiffusionLoss
from src.models import FocalStackProcessor, DualOutputDecoder

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

    loss_fn = FocalDiffusionLoss(diffusion_weight=1.0, depth_weight=0.0, rgb_weight=0.0)
    loss_dict = loss_fn(diffusion_pred=torch.randn(1,16,32,32), diffusion_target=torch.randn(1,16,32,32), shape_norm=out["shape_norm"], uncertainty=out["uncertainty"], focal_stack=focal_stack * 2 - 1, critic_outputs=critic_out)
    assert "total" in loss_dict
