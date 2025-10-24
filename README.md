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

Providing only RGB and depth paths triggers on-the-fly focal-stack synthesis via
the built-in thin-lens simulator. Optional tokens such as
`generate_focal_stack=false`, `focal_stack_dir=...`, or per-sample camera
parameters override the defaults.

Download and extract the datasets before training:

- **HyperSim** — official archives at <https://github.com/apple/ml-hypersim>.
  Files expand to `<scene>/<scene>/images/scene_cam_00_final_preview/`
  (`frame.XXXX.color.jpg`) with depth maps in
  `scene_cam_00_geometry_hdf5/frame.XXXX.depth_meters.hdf5`. Scene `ai_001_001`
  contains frames `frame.0000`–`frame.0099`; the provided file lists reference
  only that range.
- **Virtual KITTI 2** — download `vkitti_2.0.3_depthgt.zip` from
  <https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/>.
  After extraction, RGB frames live in
  `SceneXX/clone/frames/rgb/Camera_0/rgb_XXXXX.jpg` and depth targets in
  `SceneXX/clone/frames/depth/Camera_0/depth_XXXXX.png`.

Point each configuration's `data_root` to the directory that contains the
extracted scenes (e.g. `D:\Datasets\hypersim`, `/mnt/vkitti`). When mixing
sources, list one entry per dataset under `data.train_sources` and
`data.val_sources`.

## Configuration

YAML presets in `configs/` describe each experiment. `configs/base.yaml` provides
common optimisation, logging, and dataloader defaults, while
`configs/hypersim.yaml`, `configs/virtual_kitti.yaml`, and `configs/mixed.yaml`
override only the dataset section.

Important knobs:

- `model.base_model_id` — Hugging Face identifier for the Stable Diffusion 3.5
  checkpoint.
- `data.train_sources` / `data.val_sources` — lists of `{data_root, filelist}`
  dictionaries. Multiple entries concatenate datasets within an epoch.
- `data.dataset_kwargs` — defaults for the focal-stack simulator and camera
  metadata (e.g. `generate_focal_stack`, `num_slices`).
- `training.batch_size`, `training.gradient_accumulation_steps`, and
  `optimizer.learning_rate` — tune to match your hardware budget.

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
