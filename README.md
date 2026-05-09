# FocalDiffusion: Efficient Zero-Shot Focal-Stack Diffusion for Joint Appearance and Depth Recovery

**FocalDiffusion** is a focal-stack-conditioned diffusion framework for joint all-in-focus reconstruction and depth estimation. It adapts a Stable Diffusion 3.5 MMDiT backbone with focal-stack conditioning modules, camera-aware extensions, and a dual-output decoder for predicting all-in-focus appearance and depth from focal stacks.

> Note: A trained FocalDiffusion checkpoint is required for meaningful focal-stack inference. Loading only the base SD3.5 model initializes focal-specific modules without trained weights.

## Highlights

- **Focal-stack conditioning:** extracts multi-scale focus cues from a sequence of differently focused images.
- **SD3.5 MMDiT backbone:** adapts Stable Diffusion 3.5 through focal cross-attention and parameter-efficient tuning.
- **Dual-output prediction:** jointly estimates all-in-focus RGB reconstruction and depth.
- **Synthetic and pre-rendered stacks:** supports file-list based datasets and optional circle-of-confusion stack simulation.
- **Config-driven training:** YAML presets are provided for HyperSim, Virtual KITTI, and mixed-dataset experiments.

## Installation

```bash
git clone https://github.com/xingguangweiwo/FocalDiffusion.git
cd FocalDiffusion

python -m venv .venv
source .venv/bin/activate
pip install -e .

Stable Diffusion 3.5 requires access through Hugging Face:

huggingface-cli login

You may need to accept the Stable Diffusion 3.5 license terms on Hugging Face before downloading the model weights.

Data preparation

Training uses plain-text file lists. Each non-comment line follows:

<relative_rgb_or_stack_path> <relative_depth_path> [optional_extra_tokens]

The first token can point to an all-in-focus RGB image or a focal-stack directory. If only an RGB image and a depth map are provided, the loader can synthesize a focal stack on the fly using the built-in circle-of-confusion simulator. Optional tokens may include key=value pairs, an explicit stack directory, or an integer indicating the number of focal slices.

Example:

ai_001_001/images/frame.0000.color.jpg ai_001_001/geometry/frame.0000.depth_meters.hdf5
Scene01/clone/frames/rgb/Camera_0/rgb_00042.jpg Scene01/clone/frames/depth/Camera_0/depth_00042.png

Paths are resolved relative to the data_root defined in the training configuration.

Configuration

Experiments are controlled by YAML files in configs/:

configs/base.yaml: common model, optimizer, loss, logging, and output settings.
configs/hypersim.yaml: HyperSim-specific dataset paths and focal-stack settings.
configs/virtual_kitti.yaml: Virtual KITTI-specific dataset paths and focal-stack settings.
configs/mixed.yaml: mixed HyperSim + Virtual KITTI training.

Dataset-specific configs inherit from configs/base.yaml and override dataset paths and selected training options.

Validate a configuration without training:

python -m script.train --config configs/hypersim.yaml --dry-run
Training
python -m script.train --config configs/hypersim.yaml

Common overrides:

python -m script.train \
  --config configs/hypersim.yaml \
  --data-root /path/to/hypersim \
  --output-dir outputs/experiments/hypersim_run

Checkpoints and logs are written to output.save_dir.

Inference
python -m script.inference \
  --input /path/to/focal_stack \
  --output outputs/inference/example \
  --config configs/hypersim.yaml \
  --base-model stabilityai/stable-diffusion-3.5-large \
  --model-path /path/to/focaldiffusion_checkpoint.pt

The script exports:

all-in-focus RGB reconstruction
depth prediction
optional visualizations and metadata
