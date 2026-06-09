import pytest

from diffusers.models.transformers import SD3Transformer2DModel

from src.pipelines.focal_stack_generation_pipeline import FocalInjectedSD3Transformer


def _tiny_sd3_transformer() -> SD3Transformer2DModel:
    return SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=1,
        attention_head_dim=4,
        num_attention_heads=1,
        joint_attention_dim=8,
        caption_projection_dim=8,
        pooled_projection_dim=8,
        out_channels=4,
        pos_embed_max_size=8,
    )


def test_getattr_preserves_wrapper_registered_modules():
    wrapper = FocalInjectedSD3Transformer(_tiny_sd3_transformer())

    assert wrapper.pre_focal_attn is wrapper._modules["pre_focal_attn"]
    assert wrapper.focal_attn is wrapper._modules["focal_attn"]
    assert wrapper.pre_norm is wrapper._modules["pre_norm"]
    assert wrapper.post_norm is wrapper._modules["post_norm"]
    assert wrapper.condition_scale is wrapper._parameters["condition_scale"]


def test_getattr_falls_back_to_base_transformer_attributes():
    base = _tiny_sd3_transformer()
    base.custom_marker = object()
    wrapper = FocalInjectedSD3Transformer(base)

    assert wrapper.custom_marker is base.custom_marker

    with pytest.raises(AttributeError):
        _ = wrapper.attribute_that_does_not_exist
