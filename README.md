# FSDiffusion: Reliable Zero-Shot Focal-Stack Diffusion via Focal Evidence

## Short Introduction
FSDiffusion is a focal-stack-conditioned latent diffusion framework for reliable zero-shot all-in-focus reconstruction and depth estimation. FSDiffusion uses a Focal Evidence Posterior to explicitly predict a per-pixel posterior distribution over focus planes. The posterior is converted into focus-derived depth by soft-argmax, while focus entropy is treated as posterior sharpness rather than calibrated reliability. A lightweight Physical Support Head learns selective focus/prior gates and uncertainty from compact focal evidence diagnostics.

## Highlights
- Local **Focal Evidence Posterior** (`focus_posterior`) over focus planes.
- Learned physical-support gating of focus-derived depth and diffusion-prior depth.
- Physical-support uncertainty from focus entropy, posterior margin, prior-focus disagreement, and decoder uncertainty.
- AIF-focus high-pass consistency for local evidence alignment.
- Default training/inference path does **not** require PSF/NA/camera metadata.

## Important Notes
- A trained checkpoint is required for practical inference quality.
- Here, zero-shot means inference on unseen focal-stack datasets without test-set fine-tuning, not training-free inference without a learned checkpoint.
- Depth output is normalized by default.
- Metric depth/height requires dataset/device calibration (e.g., depth range).
- N=100 may require reducing feature_dim, patch resolution, or batch size.

## Installation
```bash
pip install -e .
```

## Data Preparation
- Prepare dataset file lists under `data/filelists/`.
- Configure dataset roots, filelists, and modality options in `configs/*.yaml`.
- Ensure `focal_stack`, `focus_distances`, and (for supervised/semi-supervised) labels are available according to your mode.

## Training
```bash
python script/train.py --config configs/base.yaml
```

## Inference
```bash
python script/inference.py --model-path <path_to_checkpoint> --input <focal_stack_dir_or_images> --output <output_dir>
```

## Method Overview
FSDiffusion uses a Focal Evidence Posterior to explicitly predict a per-pixel posterior distribution over focus planes. Focus entropy measures posterior sharpness rather than calibrated reliability. FSDiffusion therefore uses a lightweight Physical Support Head that calibrates the final gate and uncertainty from focus entropy, posterior margin, focus-prior disagreement, and decoder uncertainty. Reliable means evidence-aware uncertainty calibration and selective trust, not guaranteed correctness for all pixels or all scenes.

1. Estimate `focus_posterior`, `depth_focus`, `focus_entropy`, and `focus_peakiness` with a local Focal Evidence Posterior head.
2. Decode SD3/FSDiffusion latents into `depth_prior`, AIF latents, and decoder uncertainty.
3. Build compact physical-support inputs from focus peakiness / posterior sharpness, posterior margin, focus-prior disagreement, and decoder uncertainty.
4. Fuse `depth_focus` and `depth_prior` with learned focus/prior gates, and report calibrated physical support plus final uncertainty.

## Repository Structure
- `src/models/`: focal-sweep processor, focal evidence head, attention blocks, and dual-output decoder.
- `src/pipelines/`: FSDiffusion pipeline and injected SD3 transformer.
- `src/training/`: trainer, FEP losses, validation, and optimization utilities.
- `src/data/`: datasets, augmentations, simulation helpers.
- `script/`: train / inference / evaluate entry points.
- `configs/`: base and dataset-specific configs.
- `tests/`: smoke and module tests.

## License
See project license files and repository policy.
