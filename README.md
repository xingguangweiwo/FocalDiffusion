# FSDiffusion: Zero-Shot Focal-Stack Diffusion via Focal Evidence

A focal-stack-conditioned diffusion framework for joint all-in-focus reconstruction and depth estimation.

FSDiffusion uses focal stacks as physical evidence for depth prediction. It estimates a per-pixel focal-plane posterior, converts focal evidence into focus-derived depth, and fuses it with a diffusion-prior depth prediction through a lightweight physical-support gate.

## Highlights

* Focal Evidence Posterior over focus planes.
* Joint all-in-focus reconstruction and depth estimation.
* Fusion of focus-derived depth and diffusion-prior depth.
* Uncertainty-related outputs from focus entropy, prior-focus disagreement, and gate abstention.
* Zero-shot evaluation on unseen scenes or datasets after training.

## Important Notes

* A trained checkpoint is required for meaningful inference.
* “Zero-shot” means evaluation on unseen datasets or scenes without test-time fine-tuning.
* Depth is normalized by default.
* Metric depth requires calibrated focus distances and camera parameters.
* If focus distances are omitted, depth should be interpreted as relative or normalized depth.

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
* `focus_distances`
* optional camera metadata

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
  --num-inference-steps 30 \
  --guidance-scale 1.0
```

If `--focus-distances` is omitted, index-spaced focal positions are used. In that case, the output depth should not be interpreted as metric depth.

## Method

FSDiffusion consists of three main components:

1. **Focal Evidence Posterior**
   Predicts a per-pixel posterior distribution over focus planes from the input focal stack.

2. **Diffusion Prior Decoder**
   Uses latent diffusion features to reconstruct all-in-focus images and predict prior depth.

3. **Physical Support Gate**
   Fuses focus-derived depth and diffusion-prior depth using compact focal evidence diagnostics, including focus entropy, posterior margin, prior-focus disagreement, and decoder uncertainty.

## Limitations

* Metric depth requires calibrated focus distances and camera parameters.
* Without valid focus distances, output depth is relative or normalized.
* Reliability claims should be supported by high-error detection, sparsification, and calibration evaluation.
* Large diffusion backbones should be separated from focal-evidence contributions through ablation studies.

## Repository Structure

```text
src/models/       Focal evidence, focal processor, attention blocks, decoder
src/pipelines/    FSDiffusion pipeline and injected SD3 transformer
src/training/     Trainer, losses, validation, optimization
src/data/         Dataset and focal-stack simulation utilities
script/           Training and inference entry points
configs/          Experiment configurations
tests/            Smoke and module tests
```

## License

See the repository license.
