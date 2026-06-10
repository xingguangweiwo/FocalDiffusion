# Codex cleanup task: remove legacy aliases and finish decoder rename

You are working in the repository `xingguangweiwo/FocalDiffusion`.

Goal: continue slimming the codebase by removing old compatibility aliases, redundant output fields, and the remaining `dual_decoder` naming chain. Preserve the current training/inference behavior, but remove historical API names that are not needed for the current codebase.

## 1. Finish the decoder rename

The implementation now lives in:

```text
src/models/task_output_decoder.py
```

with class:

```python
TaskOutputDecoder
```

Replace remaining internal references:

```text
DualOutputDecoder -> TaskOutputDecoder
dual_decoder -> task_output_decoder
src.models.dual_decoder -> src.models.task_output_decoder
```

Apply this consistently in:

```text
src/pipelines/focal_stack_generation_pipeline.py
src/training/trainer.py
src/pipelines/pipeline_utils.py
src/training/checkpointing.py
src/models/__init__.py
any scripts/tests that still import DualOutputDecoder or dual_decoder
```

After all internal imports are updated, delete:

```text
src/models/dual_decoder.py
```

Do not keep a compatibility shim unless a test or public CLI still requires it.

Expected canonical component name:

```python
task_output_decoder
```

Expected canonical class name:

```python
TaskOutputDecoder
```

## 2. Remove legacy output aliases from the pipeline output dataclass

In `src/pipelines/focal_stack_generation_pipeline.py`, remove these backward-compatible fields from `FocalStackGenerationOutput`:

```python
depth_prior
depth_focus
depth_final
focus_reliability
focus_posterior
focus_entropy
focus_peakiness
physical_support
gate_focus
gate_prior
gate_abstain
uncertainty_decoder
```

Keep only canonical fields:

```python
depth_map
all_in_focus_image
depth_colored
uncertainty
generated_depth_canonical
focal_depth_canonical
final_depth_canonical
focal_posterior
focal_entropy
focal_peak_confidence
physical_evidence_support
focal_evidence_weight
generative_prior_weight
abstention_weight
posterior_margin
depth_disagreement
generative_uncertainty
uncertainty_focus
uncertainty_disagreement
uncertainty_final
depth_focus_metric
```

Then remove all construction-time assignments to the old fields. For example, remove mappings such as:

```python
depth_prior=generated_depth_canonical
depth_focus=focal_depth_canonical
depth_final=final_depth_canonical
focus_posterior=focal_posterior
physical_support=physical_evidence_support
gate_focus=focal_evidence_weight
```

## 3. Remove legacy keyword translation in the loss

In `src/training/losses.py`, delete support for old names:

```python
depth_prior_norm
depth_focus_norm
depth_final_norm
focus_posterior
focus_entropy
focus_distances
gate_focus
gate_prior
gate_abstain
physical_support
focus_reliability
```

Specifically:

- remove `focus_reliability` from the `FocalStackGenerationLoss.forward(...)` signature;
- remove `**legacy_kwargs` from the `forward(...)` signature;
- remove the block that pops old names from `legacy_kwargs`;
- remove unused `del focus_reliability, ...` logic;
- remove bottom-of-file aliases:

```python
normalize_focus_coordinates = normalize_focal_coordinates
build_soft_focus_target_from_depth = build_focal_axis_soft_targets
build_coc_focus_target_from_depth = build_coc_posterior_targets
FocalDiffusionLoss = FocalStackGenerationLoss
```

Only accept canonical names in new code.

## 4. Remove unused loss arguments from trainer

In `src/training/trainer.py`, remove this unused argument when calling the loss:

```python
focus_reliability=support_outputs["physical_evidence_support"].float(),
```

The loss already receives:

```python
physical_evidence_support=support_outputs["physical_evidence_support"].float(),
```

so `focus_reliability` is redundant.

Also remove fallback config keys such as:

```python
self.config['losses'].get('aif_focus_evidence_weight', 0.1)
```

Use only:

```python
self.config['losses'].get('all_in_focus_focal_evidence_weight', 0.1)
```

For model hidden size, remove old fallback:

```python
self.config["model"].get("physical_support_hidden", 16)
```

Use only:

```python
self.config["model"].get("physical_evidence_support_hidden", 16)
```

## 5. Remove legacy trainer alias

In `src/training/trainer.py`, delete:

```python
FocalDiffusionTrainer = FocalStackGenerationTrainer
```

`src/training/__init__.py` should export only:

```python
FocalStackGenerationTrainer
FocalStackGenerationLoss
get_optimizer
get_scheduler
```

## 6. Remove legacy generation-task aliases

This is already partially done. Confirm that `src/generation_tasks.py` has no `TASK_ALIASES` and accepts only canonical names:

```python
all_in_focus
depth
uncertainty
focal_evidence
refocus
```

Do not reintroduce `aif`.

## 7. Search for leftovers

Run repository-wide searches and remove remaining internal usages of:

```text
DualOutputDecoder
dual_decoder
FocalDiffusionTrainer
FocalDiffusionLoss
focus_reliability
focus_posterior
focus_entropy
focus_peakiness
focus_distances
gate_focus
gate_prior
gate_abstain
physical_support
depth_prior_norm
depth_focus_norm
depth_final_norm
aif_focus_evidence_weight
physical_support_hidden
TASK_ALIASES
```

Some strings may appear in documentation or migration notes; remove them unless they are truly necessary.

## 8. Validate

Run at least:

```bash
python -m compileall src
python -m pytest tests -q
```

If tests are unavailable or require large models, at minimum run:

```bash
python -m compileall src
```

Then add a short summary of files changed and any tests that could not run.

## Acceptance criteria

- No internal code imports `DualOutputDecoder`.
- No internal code uses `dual_decoder` as a component name.
- `src/models/dual_decoder.py` is deleted.
- Pipeline output dataclass contains only canonical fields.
- Loss accepts only canonical keyword names.
- Training call no longer passes `focus_reliability`.
- No `FocalDiffusionTrainer` or `FocalDiffusionLoss` aliases remain.
- `python -m compileall src` passes.
