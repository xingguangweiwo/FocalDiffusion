# FocalTrace: Test-Time Refinement for Focal-Stack Generation and Reconstruction via Physical Verification Trace

FocalTrace is a reliability-aware focal-stack reconstruction system for all-in-focus (AIF) image reconstruction, canonical depth estimation, focus-likelihood diagnostics, and focal-stack consistency evaluation. The active method is: focal stack → focal-sweep features → focus likelihood and focus-based depth → generative prior depth and AIF reconstruction → reliability-aware fusion → focal-consistency diagnostics → optional per-instance test-time optimization. Consistency traces describe agreement between focus cues and an image-formation model; they are not ground-truth correctness labels.

## Status Table

| Area | Implemented | Experimental | Planned |
| --- | --- | --- | --- |
| Focal-stack conditioning | Focal-sweep features condition the SD3 transformer and multi-output decoder | Attention/fusion ablations | Lightweight non-diffusion baseline |
| Focus likelihood | Per-plane posterior, entropy, margin, multimodality, texture and coverage diagnostics | Adaptive low-texture temperature and focal-axis unimodality regularization | Calibrated uncertainty for all camera models |
| Reliability | Reliability fusion head, abstention probability, consistency trace diagnostics | Selective test-time optimization (TTO) with held-out focal planes | Broader evaluator suite for transparent/specular masks |
| Data protocols | Explicit source/adaptation/test splits and coordinate protocol validation | Unlabeled target adaptation on disjoint file lists | Dataset cards with verified camera metadata |
| Evaluation | Depth metrics and internal/GT uncertainty detection hooks | AIF perceptual metrics and region-sliced reporting where labels/masks exist | Public benchmark tables after measured runs |

## Protocol Table

| Method | Uses target focal stacks? | Uses target labels? | Updates network parameters? | May be called zero-shot? |
| --- | --- | --- | --- | --- |
| **M0 feed-forward** frozen source model | Yes, for inference only | No | No | Yes |
| **M0 + per-instance TTO** | Yes, unlabeled focal stack at test time | No | No; optimizes detached per-sample state | No; report as test-time optimization |
| **M1 unsupervised adaptation** | Yes, adaptation split only | No | Yes, on unlabeled adaptation split | No; never label M1 as zero-shot |
| **Target-test evaluation** | Yes, target-test split | GT only inside evaluator metrics | No | Depends on method row above |

## Configuration Migration

| Legacy key/name | New key/name | Notes |
| --- | --- | --- |
| `data.self_improvement_sources` | `data.adaptation_sources` | Legacy configs are migrated on load. |
| `training.self_improvement` | `training.unsupervised_adaptation` | M0 keeps `enabled: false`; M1 uses `enabled: true` and `round_index >= 1`. |
| `FocalEvidenceEncoder` | `FocusLikelihoodEstimator` | Legacy import remains as an alias. |
| `PhysicalEvidenceEstimator` | `ReliabilityFusionHead` | Legacy import remains as an alias. |
| `build_physical_evidence_features` | `build_reliability_features` | Legacy wrapper remains. |
| `PhysicalVerificationTrace` | `FocalConsistencyTrace` | Legacy alias remains for checkpoints/imports. |
| `FocalPhysicalVerifier` | `FocalConsistencyEvaluator` | Legacy alias remains. |
| `physical_evidence_support` | `reliability_score` | Both output keys are currently emitted. |
| `generated_depth_canonical` | `prior_depth_canonical` | Legacy key remains in model outputs for compatibility. |
| `focal_depth_canonical` | `focus_depth_canonical` | Legacy key remains in model outputs for compatibility. |
| `abstention_weight` | `abstention_probability` | Both output keys are currently emitted. |

## Installation

```bash
pip install -e .
# Optional extras:
# pip install -e .[train,datasets,evaluation,development]
```

CPU smoke tests use lightweight module and mock paths; they do not download the SD3.5 checkpoint.

## Data and Protocols

Configure file lists in `configs/*.yaml` using disjoint splits:

* `data.train_sources` — labeled source training data.
* `data.val_sources` — source/validation data for threshold selection.
* `data.adaptation_sources` — unlabeled target adaptation data for M1 only.
* `data.test_sources` — held-out target-test data.

Every source must declare `focal_coordinate_type`, `focal_coordinate_unit`, `depth_coordinate_type`, `camera_calibration`, and `evaluation_mode`. The loader rejects incompatible calibrated/canonical combinations and adaptation/test file-list overlap.

## Commands

### Source training

```bash
python -m script.train --config configs/base.yaml \
  --override training.unsupervised_adaptation.enabled=false protocol.name=source_training
```

### Feed-forward zero-shot evaluation

```bash
python -m script.evaluate \
  --config configs/base.yaml \
  --checkpoint outputs/experiments/base/checkpoints/best.pt \
  --dataset target_test \
  --output_dir outputs/eval_m0_zero_shot \
  --num_inference_steps 30 \
  --num_refinement_steps 0
```

### Per-instance test-time optimization

```bash
python -m script.evaluate \
  --config configs/base.yaml \
  --checkpoint outputs/experiments/base/checkpoints/best.pt \
  --dataset target_test \
  --output_dir outputs/eval_m0_tto \
  --num_inference_steps 30 \
  --num_refinement_steps 4
```

### Optional unsupervised target adaptation

```bash
python -m script.train --config configs/base.yaml \
  --override training.unsupervised_adaptation.enabled=true \
             training.unsupervised_adaptation.round_index=1 \
             training.unsupervised_adaptation.parent_checkpoint=outputs/experiments/base/checkpoints/best.pt \
             protocol.name=unsupervised_target_adaptation
```

Evaluate an adapted model only on a disjoint `target_test` split and report it as unsupervised adaptation, not zero-shot.

## Metrics and Reporting

Implemented metric hooks include depth AbsRel/RMSE/L1, verifier-derived consistency-risk curves, internal violation AUROC/AUPRC, and GT depth-error AUROC/AUPRC when depth GT is passed to the evaluator. AIF PSNR/SSIM/LPIPS, AUSE/AURG, accepted refinement harm/improvement rates, and per-region metrics should be reported only when the corresponding labels or masks are available. Do not report unmeasured benchmark numbers.

## Unsupported or Not-Yet-Supported Claims

* Transparent-object depth is not solved; use model-mismatch and abstention diagnostics for such regions.
* Consistency traces are not ground-truth correctness labels.
* Metric depth claims require calibrated focal-plane distances and valid camera metadata.
* Target-domain adaptation results require disjoint adaptation/test splits and must not be described as zero-shot.

## Repository Structure

```text
src/models/       Focus likelihood, reliability fusion, focal processor, decoder
src/pipelines/    FocalStackGeneration pipeline and SD3 transformer wrapper
src/training/     Trainer, losses, validation, unsupervised adaptation replay
src/data/         Dataset and differentiable focal-stack renderer
script/           Training, inference, evaluation entry points
configs/          Protocol/configuration files
tests/            CPU smoke and module tests
```

## Release protocol and reproducibility

### Method diagram (text)

`focal stack + focal-plane coordinates` → `FocalSweepEncoder` → `FocusLikelihoodEstimator` → `JointReconstructionDecoder prior/AIF` → `ReliabilityFusionHead` → `FocalConsistencyEvaluator` → optional held-out-verified TTO → `FocalTraceOutput`.

### Implemented / Experimental / Planned

| Status | Items |
| --- | --- |
| Implemented | Canonical focal-depth output, focus likelihood diagnostics, reliability fusion, internal consistency diagnostics, optional selective TTO, checkpoint/config schema migration helpers. |
| Experimental | Unsupervised adaptation with accepted-refinement replay, calibrated metric rendering when camera metadata are available, LPIPS evaluation when installed. |
| Planned | Published benchmark numbers, full ablation tables, transparent/specular region protocols for datasets that provide masks. |

### Protocol definitions

* `source_validation`: source-domain validation used for model selection and smoke evaluation.
* `target_test`: disjoint target-domain test split for zero-shot/TTO reporting.
* `target_adaptation`: optional unlabeled adaptation split; evaluation requires `--allow-target-adaptation` and must not be reused as `target_test`.

Internal consistency metrics are diagnostics of agreement with the focal/image-formation model and must not be reported as ground-truth error metrics.

### Coordinate conventions

Canonical depth is normalized over the represented focal sweep. Metric depth is available only in calibrated mode with camera metadata. Canonical mode uses normalized focal coordinates and is not a claim of metric accuracy.

### Calibrated versus canonical rendering

`SyntheticFocalStackRenderer(mode="calibrated")` uses thin-lens CoC in metric units. `mode="canonical"` uses the normalized defocus surrogate for canonical-depth TTO and diagnostics.

### Checkpoint and config schema

Active schema versions are `checkpoint_schema_version=2` and `config_schema_version=2`. Use `migrate_checkpoint_schema` and `migrate_config_schema` from `src.pipelines.pipeline_utils` when loading older artifacts.

### Reproducible commands

```bash
python -m script.evaluate --config configs/base.yaml --checkpoint /path/to/checkpoint.pt --dataset source_validation --data_root /path/to/data --split source_validation --device cpu
python -m script.evaluate --config configs/base.yaml --checkpoint /path/to/checkpoint.pt --dataset target_test --data_root /path/to/data --split target_test --num_refinement_steps 0
python -m script.evaluate --config configs/base.yaml --checkpoint /path/to/checkpoint.pt --dataset target_test --data_root /path/to/data --split target_test --num_refinement_steps 2
python -m script.evaluate --config configs/base.yaml --checkpoint /path/to/checkpoint.pt --dataset target_adaptation --allow-target-adaptation --data_root /path/to/data --split target_adaptation
```

Expected outputs are `metrics_summary.json`, optional per-sample visualizations, and refinement summaries containing attempted/accepted steps, held-out measurements, runtime, and memory.

### Limitations and unsupported claims

This repository does not ship SD3.5 weights, benchmark datasets, or unmeasured performance claims. Benchmark tables and ablations must be generated from the released configs and reported with dataset, split, seed, runtime, and memory settings.
