# Focal Stack Understanding as Image Generation

A task-conditioned diffusion framework for focal-stack understanding, including all-in-focus reconstruction, depth generation, uncertainty estimation, focal evidence estimation, and refocused focal-plane generation.

FocalStackGeneration formulates focal-stack understanding as task-conditioned image generation. It uses focal stacks as physical evidence, estimates a per-pixel focal-plane posterior, generates canonical depth and uncertainty outputs, and estimates focal evidence weighting for physically grounded outputs while preserving the Stable Diffusion 3.5 backbone.

## Highlights

* Task-conditioned focal-stack image generation for all-in-focus reconstruction, depth, uncertainty, focal evidence, and refocus outputs.
* Focal evidence posterior over focal planes.
* All-in-focus reconstruction and depth generation from focal-stack conditioning.
* Uncertainty-related outputs from focal entropy, generated-depth disagreement, and abstention weighting.
* Refocused focal-plane generation support for held-out focal-plane evaluation.

## Important Notes

* A trained checkpoint is required for meaningful inference.
* “Zero-shot” means evaluation on unseen datasets or scenes without test-time fine-tuning.
* Depth is normalized by default.
* Metric depth requires calibrated focal-plane distances and camera parameters.
* If focal-plane distances are omitted, depth should be interpreted as relative or normalized depth.

## Installation

```bash
pip install -e .
```

Install the runtime dependencies required by your environment, including PyTorch, diffusers, transformers, accelerate, and optional PEFT/LoRA packages.

## Data Preparation

Configure dataset roots and file lists in `configs/*.yaml`.

Typical file-list entries may include:

* `focal_stack_dir`
* `all_in_focus`
* `depth_path`
* `focal_plane_distances`
* optional camera metadata

Legacy file lists may still use `focus_distances`; loaders keep compatibility where possible.

## Training

```bash
python -m script.train --config configs/base.yaml
```

Before full training, replace placeholder dataset paths in `configs/base.yaml` with real dataset roots and file lists.

## Inference

```bash
python -m script.inference \
  --model-path outputs/experiments/base/checkpoints/best.pt \
  --input path/to/focal_stack \
  --output outputs/demo \
  --focus-distances 0.3,0.5,0.8,1.5,3.0 \
  --focal-distance-mode metric \
  --num-inference-steps 30 \
  --guidance-scale 1.0
```

Use `--focal-distance-mode metric` only when `--focus-distances` / `--focal-plane-distances` are calibrated metric distances. The default mode is `normalized`, which still uses focal-plane values for canonical model conditioning but does not expose `depth_focus_metric`.

If `--focus-distances` is omitted, index-spaced focal positions are used for backward-compatible CLI behavior. In that case, keep `--focal-distance-mode normalized`, and output depth should not be interpreted as metric depth.

## Generation Tasks

Canonical task names are centralized in `src/generation_tasks.py`:

* `all_in_focus`
* `depth`
* `uncertainty`
* `focal_evidence`
* `refocus`

The legacy task alias `aif` is accepted only as a compatibility alias for `all_in_focus`.

## Method

FocalStackGeneration consists of three main components:

1. **Focal Evidence Encoder**
   Predicts a per-pixel posterior distribution over focal planes from the input focal stack.

2. **Task-Conditioned Generative Decoder**
   Uses latent diffusion features to reconstruct all-in-focus images and generate canonical depth.

3. **Physical Evidence Estimator**
   Estimates focal evidence weighting and uncertainty from compact focal evidence diagnostics, including focal entropy, posterior margin, generated-depth disagreement, and generative uncertainty.

## Limitations

* Metric depth requires calibrated focal-plane distances and camera parameters.
* Without valid focal-plane distances, output depth is relative or normalized.
* Reliability claims should be supported by high-error detection, sparsification, and calibration evaluation.
* Large diffusion backbones should be separated from focal-evidence contributions through ablation studies.

## Repository Structure

```text
src/models/       Focal evidence encoder, focal processor, attention blocks, decoder
src/pipelines/    FocalStackGeneration pipeline and injected SD3 transformer
src/training/     Trainer, losses, validation, optimization
src/data/         Dataset and synthetic focal-stack rendering utilities
script/           Training and inference entry points
configs/          Experiment configurations
tests/            Smoke and module tests
```

## License

See the repository license.
