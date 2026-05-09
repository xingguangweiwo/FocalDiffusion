# FocalDiffusion: Efficient Zero-Shot Focal-Stack Diffusion for Joint Appearance and Depth Recovery

**FocalDiffusion** is a focal-stack-conditioned diffusion framework for joint all-in-focus image reconstruction and depth estimation. It adapts a Stable Diffusion 3.5 MMDiT backbone with focal-stack conditioning modules to recover appearance and depth from focal stacks.

> **Note:** A trained FocalDiffusion checkpoint is required for meaningful inference. Loading only the base Stable Diffusion 3.5 model does not provide trained focal-stack reconstruction or depth-estimation capability.

## Highlights

- **Focal-stack conditioning:** extracts focus-dependent cues from a sequence of differently focused images.
- **Diffusion-based reconstruction:** adapts a Stable Diffusion 3.5 MMDiT backbone for focal-stack computational imaging.
- **Dual-output prediction:** predicts both all-in-focus RGB reconstruction and depth.
- **Parameter-efficient adaptation:** supports efficient tuning strategies such as attention-only tuning or LoRA-style adaptation.
- **Synthetic or pre-rendered stacks:** supports focal-stack simulation from RGB-depth pairs and precomputed focal-stack inputs.
- **Config-driven experiments:** provides YAML configurations for HyperSim, Virtual KITTI, and mixed-dataset training.

## Installation

```bash
git clone https://github.com/xingguangweiwo/FocalDiffusion.git
cd FocalDiffusion

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Stable Diffusion 3.5 requires Hugging Face access:

```bash
huggingface-cli login
```

Please make sure that you have accepted the Stable Diffusion 3.5 license terms on Hugging Face before running training or inference.

## Data Preparation

Training uses plain-text file lists. Each non-comment line follows:

```text
<relative_rgb_or_stack_path> <relative_depth_path> [optional_extra_tokens]
```

The first token can point to either an all-in-focus RGB image or a focal-stack directory. If only an RGB image and a depth map are provided, the dataloader can synthesize a focal stack using the built-in circle-of-confusion simulator.

Example:

```text
ai_001_001/ai_001_001/images/scene_cam_00_final_preview/frame.0000.color.jpg ai_001_001/ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.depth_meters.hdf5
Scene01/clone/frames/rgb/Camera_0/rgb_00042.jpg Scene01/clone/frames/depth/Camera_0/depth_00042.png
```

Additional tokens can be used to specify options such as focal-stack generation, stack directories, camera parameters, or the number of focal slices. Paths are resolved relative to the `data_root` specified in the configuration file.

## Configuration

Experiment settings are defined in `configs/`.

```text
configs/base.yaml           Common model, optimizer, loss, and logging settings
configs/hypersim.yaml       HyperSim-specific training configuration
configs/virtual_kitti.yaml  Virtual KITTI-specific training configuration
configs/mixed.yaml          Mixed HyperSim and Virtual KITTI training configuration
```

Dataset-specific configuration files inherit from `configs/base.yaml` and override dataset paths, focal-stack settings, and selected training options.

Validate a configuration without starting training:

```bash
python -m script.train --config configs/hypersim.yaml --dry-run
```

## Training

Train on HyperSim:

```bash
python -m script.train --config configs/hypersim.yaml
```

Train with custom paths:

```bash
python -m script.train \
  --config configs/hypersim.yaml \
  --data-root /path/to/hypersim \
  --output-dir outputs/experiments/hypersim_run
```

Train on Virtual KITTI:

```bash
python -m script.train --config configs/virtual_kitti.yaml
```

Train on mixed datasets:

```bash
python -m script.train --config configs/mixed.yaml
```

Checkpoints, logs, and visualizations are saved to the output directory specified in the configuration.

## Inference

Run inference with a trained FocalDiffusion checkpoint:

```bash
python -m script.inference \
  --input /path/to/focal_stack \
  --output outputs/inference/example \
  --config configs/hypersim.yaml \
  --base-model stabilityai/stable-diffusion-3.5-large \
  --model-path /path/to/focaldiffusion_checkpoint.pt
```

The inference script exports:

```text
all-in-focus RGB reconstruction
predicted depth map
optional visualizations and metadata
```

## Training Objective

The training objective combines diffusion supervision with task-specific reconstruction losses:

```text
diffusion noise prediction loss
RGB reconstruction loss
depth supervision loss
focus-consistency loss
optional edge-aware or perceptual losses
```

## Important Notes

- Stable Diffusion 3.5 uses an MMDiT transformer backbone, not a traditional UNet.
- A trained FocalDiffusion checkpoint is required for meaningful focal-stack inference.
- Metric-scale depth prediction requires metric supervision and valid depth calibration.
- The circle-of-confusion simulator is a simplified approximation of real optical defocus.
- Zero-shot performance should be reported only for explicitly validated datasets and settings.

## Repository Structure

```text
configs/          YAML experiment configurations
data/filelists/   Example dataset file lists
script/           Training, evaluation, and inference scripts
src/data/         Dataset loading and focal-stack simulation
src/models/       Focal encoders, camera encoders, and decoders
src/pipelines/    FocalDiffusion pipeline based on SD3.5
src/training/     Trainer, losses, optimization, and validation utilities
```

## Limitations

FocalDiffusion is currently an early research prototype. The main limitations are:

- The public implementation focuses on HyperSim and Virtual KITTI.
- Real focal-stack benchmarks should be added for stronger validation.
- The synthetic defocus simulator does not fully model real lens aberrations, sensor noise, or ISP effects.
- Zero-shot transfer should be claimed only for explicitly tested datasets and settings.
- Pretrained FocalDiffusion weights are required for meaningful inference and should be released with the paper or project update.

## License

Please add a `LICENSE` file before public release.

The source code license should be specified by the authors. The Stable Diffusion 3.5 backbone is subject to the Stability AI license terms.
