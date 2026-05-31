# FSDiffusion: Zero-Shot Focal-Stack-Conditioned Latent Diffusion for Joint All-in-Focus Reconstruction and Depth Estimation

## Short Introduction
FSDiffusion is a focal-stack-conditioned latent diffusion framework for joint all-in-focus reconstruction and depth estimation. It uses images captured at different focus positions to predict a normalized shape/depth map, recover an all-in-focus image, and estimate uncertainty.

## Highlights
- SD3-based latent diffusion training with an **SD3 flow-matching objective**.
- Joint outputs: normalized shape/depth, AIF reconstruction, and uncertainty.
- Focus-consistency critic for focus-energy and contrast diagnostics during training/validation.
- Default training/inference path does **not** require PSF/NA/camera metadata.

## Important Notes
- A trained checkpoint is required for practical inference quality.
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
1. Encode focal stack features with focal processor modules.
2. Condition SD3 transformer denoising with focal features.
3. Decode latent outputs into normalized shape, AIF latents, and uncertainty.
4. Apply supervised/semi-supervised objectives plus focus-consistency diagnostics.

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
