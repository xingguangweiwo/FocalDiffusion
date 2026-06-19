# FocalTrace: Self-Refining Focal-Stack Understanding via Physical Verification Traces

FocalTrace is a task-conditioned diffusion framework for focal-stack understanding with Physical Verification Traces. It supports all-in-focus reconstruction, canonical depth generation, uncertainty estimation, focal evidence estimation, physical trace verification, and inference-time trace-guided self-refinement.

The system uses focal stacks as physical evidence, estimates a per-pixel focal-plane posterior, generates canonical depth and uncertainty outputs, and computes a Physical Verification Trace that summarizes focus support, defocus/refocus consistency, conflict, invalidity, and physical support signals while preserving the Stable Diffusion 3.5 backbone.

## Highlights

* Task-conditioned focal-stack image generation for all-in-focus reconstruction, depth, uncertainty, focal evidence, and refocus outputs.
* Physical Verification Trace outputs for focus confidence/index/coordinate, depth-focus discrepancy, defocus/refocus residuals, support, conflict, invalidity, and verdict logits.
* Inference-time self-refinement driven by physical verification traces.
* Evaluation metrics for physical hallucination, valid physical reconstruction at coverage, and conflict/invalid trace scores.
* Experimental training-time trace mining/replay hooks for an initial M0→M1 self-improvement workflow; this is not yet a complete automated M0→M1→M2 closed loop.

## Important Notes

* A trained checkpoint is required for meaningful inference.
* “Zero-shot” means evaluation on unseen datasets or scenes without test-time fine-tuning.
* Depth is normalized by default.
* `focal_plane_distances` are currently required by the pipeline and evaluation/inference scripts.
* Metric depth is interpretable only when focal-plane distances are calibrated metric distances and camera parameters are valid; otherwise outputs should be treated as relative/canonical depth.

## Installation

```bash
pip install -e .
```

Install the runtime dependencies required by your environment, including PyTorch, diffusers, transformers, accelerate, and optional PEFT/LoRA packages.

## Data Preparation

Configure dataset roots and file lists in `configs/*.yaml`.

Typical file-list entries may include:

* `focal_stack_dir`
* `all_in_focus`
* `depth_path`
* `focal_plane_distances`
* optional camera metadata


## Training

```bash
python -m script.train --config configs/base.yaml
```

Before full training, replace placeholder dataset paths in `configs/base.yaml` with real dataset roots and file lists.

## Inference

```bash
python -m script.inference \
  --model-path outputs/experiments/base/checkpoints/best.pt \
  --input path/to/focal_stack \
  --output outputs/demo \
  --focal-plane-distances 0.3,0.5,0.8,1.5,3.0 \
  --focal-distance-mode metric \
  --num-inference-steps 30 \
  --guidance-scale 1.0
```

`--focal-plane-distances` is required. Use `--focal-distance-mode metric` only when those distances are calibrated metric distances. The default mode is `normalized`, which still uses focal-plane values for canonical model conditioning but does not expose `depth_focus_metric`. Without calibrated focal distances and camera parameters, output depth should be interpreted as relative/canonical rather than metric depth.


## Evaluation

Use `script.evaluate` as the official evaluation entry point for FocalTrace/FocalStackGeneration checkpoints:

```bash
python -m script.evaluate \
  --config configs/base.yaml \
  --checkpoint outputs/experiments/base/checkpoints/best.pt \
  --dataset hypersim \
  --data_root /path/to/hypersim \
  --output_dir outputs/evaluation \
  --num_inference_steps 30 \
  --num_refinement_steps 0
```

The evaluator writes `metrics.json`, `trace_metrics.json`, and a `visualizations/` directory. When refinement is enabled with `--num_refinement_steps > 0`, it also requests refinement history from the pipeline and writes `refinement_curve.json`. Use `--confidence-threshold`, `--violation-threshold`, and `--coverage` to tune the reported trace metrics.

Trace metrics include:

* **Physical Hallucination Rate (PHR) / false-confident violation rate**: the fraction of high-confidence predictions whose physical violation score exceeds the configured threshold.
* **VPR@Coverage** (`VPR_at_coverage`): among the top-confidence coverage fraction, the proportion of pixels/patches whose violation is below threshold.
* **`mean_conflict_score`**: average physical conflict score from the trace.
* **`mean_invalid_score`**: average invalid/physically unreliable score from the trace.

## Generation Tasks

Canonical task names are centralized in `src/generation_tasks.py`:

* `all_in_focus`
* `depth`
* `uncertainty`
* `focal_evidence`
* `refocus`

The legacy task alias `aif` is accepted only as a compatibility alias for `all_in_focus`.

## Method

FocalTrace consists of four main components:

1. **Focal Evidence Encoder**
   Predicts a per-pixel posterior distribution over focal planes from the input focal stack.

2. **Task-Conditioned Generative Decoder**
   Uses latent diffusion features to reconstruct all-in-focus images and generate canonical depth.

3. **Physical Evidence Estimator**
   Estimates focal evidence weighting and uncertainty from compact focal evidence diagnostics, including focal entropy, posterior margin, generated-depth disagreement, and generative uncertainty.

4. **Physical Verification Trace**
   Computes fixed physical checks for focus confidence, focus peak index/coordinate, defocus/refocus residuals, support, conflict, and invalidity. This trace is implemented and can guide inference-time self-refinement. Training-time self-improvement is currently experimental via optional trace mining/replay, not a full multi-round automated loop.

## Limitations

* Metric depth requires calibrated focal-plane distances and valid camera parameters; otherwise output depth is relative/canonical.
* `focal_plane_distances` are required by the current pipeline.
* Reliability claims should be supported by high-error detection, sparsification, and calibration evaluation.
* Large diffusion backbones should be separated from focal-evidence contributions through ablation studies.

## Repository Structure

```text
src/models/       Focal evidence encoder, focal processor, attention blocks, decoder
src/pipelines/    FocalStackGeneration pipeline and injected SD3 transformer
src/training/     Trainer, losses, validation, optimization
src/data/         Dataset and synthetic focal-stack rendering utilities
script/           Training and inference entry points
configs/          Experiment configurations
tests/            Smoke and module tests
```

## License

See the repository license.
