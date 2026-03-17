# FocalDiffusion

All-in-focus image and metric depth recovery from focal stacks, powered by fine-tuning Stable Diffusion 3.5. This repository contains the training, evaluation, and inference pipelines released with *FocalDiffusion* for focal-stack-based all-in-focus reconstruction and depth estimation.

## At a glance

- **Two-view outputs:** reconstruct all-in-focus RGB and metric depth from focal stacks.
- **Stable Diffusion 3.5 backbone:** load weights directly from Hugging Face with `diffusers`.
- **Synthetic or real stacks:** supports pre-rendered sequences or on-the-fly simulation via a circle-of-confusion model.
- **Config-driven:** reproducible YAML presets for HyperSim, Virtual KITTI, and mixed datasets.
- **Trainer CLI:** unified entry points for dry-runs, training, evaluation, and inference.

## Contents

1. [Environment setup](#environment-setup)
2. [Data preparation](#data-preparation)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Inference](#inference)
6. [Repository layout](#repository-layout)
7. [Troubleshooting](#troubleshooting)
8. [Paper-facing method notes](#paper-facing-method-notes)
9. [Reviewer-readiness checklist](#reviewer-readiness-checklist)
10. [Slide-friendly sketch](#slide-friendly-sketch)

## Environment setup

**Requirements**

- Python 3.10+
- CUDA-enabled PyTorch 2.2+
- [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub) credentials to pull Stable Diffusion 3.5 checkpoints.

Install dependencies (recommended inside a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Authenticate with Hugging Face so `diffusers` can download model weights on demand:

```bash
huggingface-cli login
```

## Data preparation

Training consumes focal stacks paired with ground-truth depth. File lists under `data/filelists/` document the supported formats:

- **CSV** — rows look like `<stack_directory>,<depth_map_path>,<num_images>` for pre-rendered stacks.
- **JSON** — can reference HyperSim HDF5 files and optionally an all-in-focus RGB frame. When `generate_focal_stack` is enabled, the loader synthesises the stack using the built-in circle-of-confusion simulator. Camera parameters, focus distances, depth scaling factors, and orientation fixes can be specified per sample.

Paths are resolved relative to `data.data_root` in your configuration. Prepare file lists for `train`, `val`, and `test` splits after downloading datasets (e.g., HyperSim, Virtual KITTI) or generating your own stacks.

## Configuration

Experiments are driven by YAML presets in `configs/`:

- `configs/base.yaml` — common optimisation, logging, and dataloader defaults.
- `configs/hypersim.yaml`, `configs/virtual_kitti.yaml`, `configs/mixed.yaml` — dataset-specific overrides.

> **Tip:** Edit the configuration you actually launch (or pass CLI overrides). Changing `configs/base.yaml` alone will not affect `configs/hypersim.yaml` because it sets its own `data_root` placeholder.

Key options:

- `model.base_model_id` — Stable Diffusion 3.5 checkpoint to adapt.
- `data.train_sources`/`val_sources`/`test_sources` to mix datasets, or `data.data_root` + `*_filelist` for a single dataset. Per-source overrides such as `focal_range`, `focal_stack_size`, and `image_size` can be specified inside each source block. Use `data.dataset_kwargs` for camera defaults, `simulator_kwargs`, and per-split overrides (e.g., `generate_focal_stack`).
- `training.batch_size`, `training.gradient_accumulation_steps`, `optimizer.learning_rate` — tune for your hardware budget.

Validate a configuration without starting optimisation:

```bash
python -m script.train --config configs/hypersim.yaml --dry-run
```

## Training

Launch optimisation once datasets and configs are in place:

```bash
python -m script.train --config configs/hypersim.yaml
```

Checkpoints and logs are written to `output.save_dir`. Enable Weights & Biases logging by setting `logging.use_wandb: true` in the config.

## Inference

Export predictions for a focal stack directory:

```bash
python -m script.inference \
    --input /path/to/focal_stack \
    --output outputs/inference/example \
    --config configs/hypersim.yaml \
    --model-path /path/to/checkpoint
```

The script produces the recovered all-in-focus RGB, the metric depth map, and optional visualisations. Run `python -m script.inference --help` for the complete argument list.

## Paper-facing method notes

If you are preparing a manuscript, here is the implementation-grounded summary of what is actually modified:

- **Focal conditioning path:** focal-stack features are injected into the SD3.5 transformer **before and after** the backbone call through dedicated focal cross-attention blocks (`pre_focal_attn`, `focal_attn`) inside `FocalInjectedSD3Transformer`.
- **Feature extraction/fusion:** `FocalStackProcessor` aggregates stack-level cues into the conditioning features consumed by the transformer wrapper.
- **Camera metadata path:** `CameraInvariantEncoder` provides camera-parameter embeddings that can be fused with focal features.
- **Dual-task decoding:** `DualOutputDecoder` predicts both all-in-focus RGB and metric depth from latent features.
- **Objective terms:** training combines diffusion noise prediction with depth, RGB, consistency, and optional perceptual losses (`FocalDiffusionLoss`).

These modules are the primary places to cite when describing the method in equation form.

## Reviewer-readiness checklist

The following checklist is kept in this README (instead of a separate file) to make paper-facing guidance self-contained.

### 1) Clarify algorithmic novelty (must be unambiguous)

In the method section, explicitly specify:

- focal-stack tensor shape/order;
- whether and where camera metadata is encoded;
- where focal features interact with SD3.5;
- what is modified vs. frozen in SD3.5;
- which parameters are trainable (e.g., LoRA/focal modules/decoders);
- how RGB and depth heads share or split representations.

### 2) Prove "why diffusion" via ablations

Minimum ablation set:

1. w/ diffusion loss vs. w/o diffusion loss (same architecture);
2. w/ RGB branch vs. w/o RGB branch (measure depth-boundary impact);
3. w/ focal conditioning vs. weak aggregation baseline;
4. SD3.5 backbone vs. lightweight UNet baseline.

Report depth + RGB + efficiency together so gains are attributable and fair.

### 3) Stress optical realism

Evaluate robustness under unseen optics/capture settings:

- focal length / f-number / focus schedule / pixel size;
- sensor noise, exposure shifts, white-balance shifts, compression artifacts;
- synthetic-to-real transfer with explicit failure analysis.

### 4) Define "zero-shot" strictly

Use one precise claim and keep it consistent across title/abstract/main text:

- zero-shot optics transfer,
- zero-shot synthetic-to-real transfer, or
- cross-dataset generalization.

Avoid conflating these claims unless all are demonstrated.

### 5) Use DFF-expected benchmarks when possible

In addition to HyperSim / Virtual KITTI, prioritize:

- DDFF-12,
- FoD500,
- NYU-v2 focal-stack synthesis protocol,
- ARKitScenes or other real-capture sets.

### 6) Report cost-quality tradeoff

For each variant/baseline, include:

- total/trainable params,
- runtime (per sample and per stack size),
- peak VRAM,
- training GPU-hours,
- input resolution and focal-stack length.

### 7) Priority experiment roadmap

1. prove diffusion contributes non-trivial gains;
2. prove synthetic training transfers to real/unseen optics;
3. prove value per compute vs. traditional DFF methods;
4. prove RGB-depth multi-task coupling helps both tasks.

## Current limitations to acknowledge in a paper draft

- CoC-based blur simulation may still be simplified versus real lens aberrations/ISP pipelines.
- Present public configs focus on HyperSim/Virtual KITTI, so broader DFF benchmark coverage should be expanded.
- "Zero-shot" claims should be narrowed to the exact validated regime to avoid overstatement.

## Repository layout

- `configs/` — experiment presets.
- `data/filelists/` — example lists for HyperSim, Virtual KITTI, and mixed splits.
- `script/` — CLI entry points for training, evaluation, and utilities.
- `src/` — dataset, simulator, pipeline, and trainer implementations.
- `src/legacy/` — archived experimental implementations kept for reference only.

## Troubleshooting

- **Model downloads fail:** ensure `huggingface-cli login` succeeded and the token has access to the Stable Diffusion 3.5 repository.
- **OOM during training:** lower `training.batch_size`, increase `gradient_accumulation_steps`, or enable FlashAttention/xFormers.
- **Misaligned depth scale:** check `data.dataset_kwargs.depth_scale` and per-split overrides in your config; verify file list depth units.
- **Generated stacks look wrong:** confirm focus distances and camera parameters in your JSON file lists when `generate_focal_stack` is enabled.

## Slide-friendly sketch

If you want to redraw the architecture in PowerPoint/Keynote/Google Slides, mirror this layout:

1. **Inputs (left column):** box labelled “Focal stack (N images)” with arrows showing optional “camera + focus distances”.
2. **Preprocessing:** small box “resize/normalise; optional CoC simulator” feeding into the model.
3. **Model core:** central box “Stable Diffusion 3.5 UNet (focal-conditioned)”. Add an annotation bubble “fine-tune: RGB + depth”.
4. **Outputs (right column):** two parallel boxes “All-in-focus RGB” and “Metric depth”.
5. **Losses (feedback loop):** curved arrow from outputs back to the model labelled “RGB L1/SSIM + depth L1/scale-aware”.

Use consistent colours (e.g., blue for inputs, teal for model, orange for outputs, red for loss arrows) and minimal text so the flow is readable at a glance.
