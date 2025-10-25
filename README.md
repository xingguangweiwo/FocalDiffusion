# FocalDiffusion

FocalDiffusion fine-tunes Stable Diffusion 3.5 to recover all-in-focus RGB and
metric depth from focal stacks. The repository hosts the training, evaluation,
and inference code released with *FocalDiffusion: Affordable Zero-shot
Diffusion-Based Image and Depth Generators from Focal Stack*.

## Environment

- Python 3.10+
- CUDA-enabled PyTorch 2.2+

Install the core dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.31.0 transformers accelerate safetensors lpips einops
```

Authenticate with Hugging Face so diffusers can download Stable Diffusion 3.5
weights on demand:

```bash
huggingface-cli login
```

Optional accelerators (FlashAttention/xFormers) can be installed separately.

## Data

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

YAML presets in `configs/` describe each experiment. `configs/base.yaml` provides
common optimisation, logging, and dataloader defaults, while
`configs/hypersim.yaml`, `configs/virtual_kitti.yaml`, and `configs/mixed.yaml`
override only the dataset section.

> **Important:** Edit the configuration you actually launch (or pass overrides on
> the CLI). Tweaking `configs/base.yaml` alone will not change
> `configs/hypersim.yaml` because that file sets its own `data_root`
> placeholder.

Important knobs:

- `model.base_model_id` – the Stable Diffusion 3.5 checkpoint to adapt.
- `data.data_root` and the `*_filelist` entries – absolute paths to your focal
  stack datasets.  Use `data.dataset_kwargs` to pass camera defaults,
  `simulator_kwargs`, and per-split overrides such as `generate_focal_stack`.
- `training.batch_size`, `training.gradient_accumulation_steps`,
  `optimizer.learning_rate` – adjusted to your hardware budget.

Validate a configuration without starting optimisation:

```bash
python -m script.train --config configs/hypersim.yaml --dry-run
```

## Training

Launch optimisation once datasets and configs are in place:

```bash
python -m script.train --config configs/hypersim.yaml
```

Checkpoints and logs are written to `output.save_dir`. Enable Weights & Biases
logging by setting `logging.use_wandb: true`.

## Inference

Export predictions for a focal stack directory:

```bash
python -m script.inference \
    --input /path/to/focal_stack \
    --output outputs/inference/example \
    --config configs/hypersim.yaml \
    --model-path /path/to/checkpoint
```

The script produces the recovered all-in-focus RGB, the metric depth map, and
optional visualisations. Run `python -m script.inference --help` for the complete
argument list.

## Repository layout

- `configs/` — experiment presets.
- `data/filelists/` — sample lists for HyperSim, Virtual KITTI,
  and mixed splits.
- `script/` — CLI entry points for training, evaluation, and utilities.
- `src/` — dataset, simulator, pipeline, and trainer implementations.
