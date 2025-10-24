# FocalDiffusion

FocalDiffusion adapts a pre-trained Stable Diffusion 3.5 backbone to focal-stack
inputs in order to predict all-in-focus RGB images together with metric depth
maps.  The repository provides the training and inference code used in the
paper *FocalDiffusion: Affordable Zero-shot Diffusion-Based Image and Depth
Generators from Focal Stack*.

## Prerequisites

The project targets Python 3.10+ with CUDA-enabled PyTorch.  Install the core
dependencies with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.31.0 transformers accelerate safetensors lpips einops
```

Authenticate with Hugging Face before training so the Stable Diffusion 3.5
weights can be downloaded:

```bash
huggingface-cli login
```

## Datasets

Training relies on focal stacks paired with ground-truth depth.  The
`data/filelists/` directory documents the whitespace separated format adopted
from **Marigold**:

```
<relative_rgb_path> <relative_depth_path> [optional_tokens]
```

Providing just the RGB and depth paths lets the loader reuse the same
all-in-focus RGB that Marigold expects while generating the focal stack
on-the-fly via the thin-lens simulator.  Additional `key=value` tokens or JSON
lines can still reference pre-rendered stacks, specify HyperSim HDF5 dataset
names, override camera parameters, or disable on-the-fly synthesis per sample.

Paths are resolved relative to each source's `data_root` as declared in the
configuration.  Create separate file lists for your `train`, `val`, and `test`
splits – you can combine HyperSim and Virtual KITTI by listing both datasets
under `train_sources` / `val_sources`.

Download the raw datasets before launching training:

- **HyperSim** – request access and download from the official project page
  (<https://github.com/apple/ml-hypersim>).  Each archive expands to
  `<scene_id>/<scene_id>/images/` with RGB frames located in
  `scene_cam_00_final_preview/frame.XXXX.color.jpg` and depth delivered as
  `scene_cam_00_geometry_hdf5/frame.XXXX.depth_meters.hdf5`.  Scene
  `ai_001_001` ships frames `frame.0000`–`frame.0099`; the sample file lists only
  reference indices within that range so they work out of the box after you set
  `data_root` to the folder that contains `ai_001_001/`.
- **Virtual KITTI 2** – download the RGB+depth release from the official web
  page (<https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/>)
  or the direct archive linked therein (`vkitti_2.0.3_depthgt.zip`).  After
  extraction you should see `SceneXX/clone/frames/`.  RGB frames live under
  `frames/rgb/Camera_0/rgb_XXXXX.jpg` while depth targets use
  `frames/depth/Camera_0/depth_XXXXX.png`.  The provided lists reference
  `rgb_00001`–`rgb_00110`, all of which are present in the standard download.

Referencing those relative paths in the file lists lets you operate directly on
the unmodified releases without converting depth to PNG first.  Point the
configuration's `data_root` entries (e.g. `configs/base.yaml:data.train_sources`
→ `data_root`) at the folder that contains the extracted scene directories, such
as `D:\Files\paper\DepthEstimation\Dataset\hypersim` or
`/mnt/datasets/virtual_kitti`.

*Virtual KITTI 2 layout.*  Place `data_root` at the top-level directory that
contains `SceneXX/clone/frames/`.  RGB frames live under
`frames/rgb/Camera_0/rgb_XXXXX.jpg` and depth targets under
`frames/depth/Camera_0/depth_XXXXX.png`.

## Configuration

All experiments are described with YAML files located in `configs/`:

- `configs/base.yaml` collects the default optimisation, model, and logging
  settings.
- `configs/hypersim.yaml`, `configs/virtual_kitti.yaml`, and
  `configs/mixed.yaml` inherit from the base recipe and only override the dataset
  section.

You can start from one of the presets and edit the following keys:

- `model.base_model_id` – the Stable Diffusion 3.5 checkpoint to adapt.
- `data.train_sources` / `data.val_sources` – lists of `{data_root, filelist}`
  mappings.  Provide one entry per dataset (e.g. HyperSim and Virtual KITTI) to
  train on a concatenated mixture.  Use `data.dataset_kwargs` to pass camera
  defaults, simulator hyper-parameters, and per-split overrides such as
  `generate_focal_stack`.
- `training.batch_size`, `training.gradient_accumulation_steps`,
  `optimizer.learning_rate` – adjusted to your hardware budget.

Validate any configuration without starting optimisation via:

```bash
python -m script.train --config configs/hypersim.yaml --dry-run
```

## Training

Launch full training once the datasets and configuration are in place:

```bash
python -m script.train --config configs/hypersim.yaml
```

Checkpoints and logs are written under `output.save_dir`.  Set
`logging.use_wandb` to `true` to stream metrics to Weights & Biases via
`accelerate`.

## Inference

After training, export predictions for a focal stack directory with:

```bash
python -m script.inference --input /path/to/focal_stack_dir \
    --output outputs/inference/example \
    --config configs/hypersim.yaml \
    --model-path /path/to/checkpoint
```

The script saves the all-in-focus reconstruction, depth map, and optional
visualisations.  Refer to `python -m script.inference --help` for the complete
set of arguments.

## Project structure

- `configs/`: training and dataset presets.
- `data/filelists/`: file list templates for common benchmarks.
- `script/`: entry points for training, evaluation, and utilities.
- `src/`: library code implementing the pipeline, models, and trainer.

