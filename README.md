# FSDiffusion: Reliable Zero-Shot Focal-Stack Diffusion via Focal Evidence

## Short Introduction
FSDiffusion is a focal-stack-conditioned latent diffusion framework for joint all-in-focus reconstruction and depth estimation. It uses images captured at different focus positions to predict a normalized shape/depth map, recover an all-in-focus image, and estimate uncertainty.

## Highlights
- SD3-based latent diffusion training with an **SD3 flow-matching objective**.
- Joint outputs: normalized shape/depth, AIF reconstruction, and uncertainty.
- Focus-consistency critic for focus-energy and contrast diagnostics during training/validation.
- Default training/inference path does **not** require PSF/NA/camera metadata.

## Important Notes
- A trained checkpoint is required for practical inference quality.
- Here, zero-shot means inference on unseen focal-stack datasets without test-set fine-tuning, not training-free inference without a learned checkpoint.
- Depth output is normalized by default.
- Metric depth/height requires dataset/device calibration (e.g., depth range).

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
FSDiffusion now uses a Focal Evidence Posterior module that explicitly predicts a per-pixel distribution over focus planes. The posterior is converted into a focus-derived depth by soft-argmax and into observability by entropy. The final depth is a physics-gated fusion of the focus-derived depth and the diffusion decoder prior. This differs from direct stack-attention depth prediction because the model exposes the focal response curve and uncertainty, preserving the algorithmic structure of classical focus measurement.

1. Encode focal stack evidence with a local Focal Evidence Posterior head and focal-sweep conditioning modules.
2. Condition SD3 transformer denoising with focal features.
3. Decode latent outputs into a normalized depth prior, AIF latents, and decoder uncertainty.
4. Fuse the focus-derived depth and decoder prior with entropy-based reliability, while keeping the old focus critic only as an optional ablation.

## Repository Structure
- `src/models/`: focal processors, attention blocks, decoders, camera modules.
- `src/pipelines/`: FSDiffusion pipeline and injected SD3 transformer.
- `src/training/`: trainer, losses, validation, optimization utilities.
- `src/data/`: datasets, augmentations, simulation helpers.
- `script/`: train / inference / evaluate entry points.
- `configs/`: base and dataset-specific configs.
- `tests/`: smoke and module tests.

## License
See project license files and repository policy.
