# FocalDiffusion: Focal-Stack-Conditioned Latent Diffusion for Joint All-in-Focus Reconstruction and Depth Estimation

FocalDiffusion is a focal-stack-conditioned SD3/SD3.5 latent diffusion project for joint all-in-focus (AIF) reconstruction and depth/shape prediction.

- **Input:** focal stack + focus positions (`focus_distances`)
- **Output:** AIF image + normalized depth/shape + uncertainty
- **Metric depth:** requires calibration (`depth_range`, z-step, or equivalent metadata)
- **Default path:** does **not** require PSF/NA/lens type/camera metadata
- **Inference:** a trained FocalDiffusion checkpoint is required

## Highlights

- SD3/SD3.5 backbone with focal-stack conditioning.
- Flow-matching training objective.
- Normalized-shape supervision when `depth_range` is available.

## Installation

```bash
git clone https://github.com/xingguangweiwo/FocalDiffusion.git
cd FocalDiffusion
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For SD3.5 access:

```bash
huggingface-cli login
```

## Data Preparation

Training uses plain-text file lists:

```text
<relative_rgb_or_stack_path> <relative_depth_path> [optional_extra_tokens]
```

- The first token can be an AIF image path or focal-stack directory.
- If RGB + depth are provided, focal stacks can be synthesized by the built-in simulator.
- Paths are resolved relative to `data_root` in config.

## Training

```bash
python -m script.train --config configs/hypersim.yaml
```

Mixed dataset training:

```bash
python -m script.train --config configs/mixed.yaml
```

## Inference

```bash
python -m script.inference \
  --input /path/to/focal_stack \
  --output outputs/inference/example \
  --model-path /path/to/focaldiffusion_checkpoint.pt \
  --base-model stabilityai/stable-diffusion-3.5-large
```

## Method Overview

Current training objective combines:

- SD3/SD3.5 flow-matching loss
- focus-consistency critic contrast losses
- normalized shape supervision (if `depth_range` exists)
- AIF high-pass consistency
- uncertainty reliability loss

## Important Notes

- Model outputs are normalized by default.
- Metric depth / microscope height needs calibration information.
- A trained checkpoint is required for meaningful inference.

## Repository Structure

```text
configs/          YAML experiment configurations
data/filelists/   Example dataset file lists
script/           Training / evaluation / inference scripts
src/data/         Dataset loading and focal-stack simulation
src/models/       Focal encoders and decoders
src/pipelines/    FocalDiffusion SD3.5 pipeline
src/training/     Trainer, losses, optimization, validation
```

## License

Please add a `LICENSE` file before public release.
