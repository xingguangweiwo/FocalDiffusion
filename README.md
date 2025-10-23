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
`data/filelists/` directory documents the supported text formats:

- **CSV** entries describe pre-rendered stacks with
  `<stack_directory>,<depth_map_path>,<num_images>`.
- **JSON** entries can point to HyperSim HDF5 files and optionally include an
  all-in-focus RGB frame.  When `generate_focal_stack` is enabled the loader will
  synthesise the stack on-the-fly using the built-in circle-of-confusion
  simulator (mirroring the MATLAB reference shared above).  Camera parameters,
  focus distances, depth scaling factors, and orientation fixes can be provided
  per sample.

Paths are resolved relative to `data.data_root` in the configuration.  Generate
file lists for your `train`, `val`, and `test` splits after preparing the
datasets (e.g. HyperSim, Virtual KITTI) or your own focal-stack generator.

## Configuration

All experiments are described with YAML files located in `configs/`:

- `configs/base.yaml` collects the default optimisation, model, and logging
  settings.
- `configs/hypersim.yaml`, `configs/virtual_kitti.yaml`, and
  `configs/mixed.yaml` inherit from the base recipe and only override the dataset
  section.

You can start from one of the presets and edit the following keys:

- `model.base_model_id` – the Stable Diffusion 3.5 checkpoint to adapt.
- `data.data_root` and the `*_filelist` entries – absolute paths to your focal
  stack datasets.  Use `data.dataset_kwargs` to pass camera defaults,
  `simulator_kwargs`, and per-split overrides such as `generate_focal_stack`.
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

