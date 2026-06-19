"""Validation utilities for :mod:`src.training.trainer`."""

from __future__ import annotations

from typing import Dict, Tuple

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .sd3_objective import predict_clean_latents_from_flow, sample_sd3_flow_matching_batch
from ..models.focal_evidence_encoder import build_physical_evidence_features
from ..models.physics_modules import _split_unit_and_signed_ranges
from ..utils.metrics import compute_metrics


_TEACHER_FORCED_KEYS = (
    "teacher_forced_abs_rel",
    "teacher_forced_rmse",
    "teacher_forced_l1",
)
_GENERATIVE_KEYS = (
    "generative_abs_rel",
    "generative_rmse",
    "generative_l1",
)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(pred, target)

    mask = mask.to(device=pred.device, dtype=torch.bool)
    if pred.dim() == mask.dim() + 1 and pred.shape[1] == 1:
        mask = mask.unsqueeze(1)
    elif mask.dim() == pred.dim() + 1 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    if not mask.any():
        return pred.new_tensor(0.0)

    return F.l1_loss(pred[mask], target[mask])


def _zero_metrics(keys: Tuple[str, ...]) -> Dict[str, float]:
    return {key: 0.0 for key in keys}


def _module_device_dtype(
    module: torch.nn.Module,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    if tensor is None:
        return fallback_device, fallback_dtype
    return tensor.device, tensor.dtype


def _prepare_depth_target(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    depth_gt = batch.get("depth")
    depth_range = batch.get("depth_range")
    mask = batch.get("valid_mask")

    if depth_gt is not None:
        depth_gt = depth_gt.to(device=device)
        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)
    if depth_range is not None:
        depth_range = depth_range.to(device=device)
    if mask is not None:
        mask = mask.to(device=device)

    return depth_gt, depth_range, mask


def _canonical_to_metric_depth(
    final_depth_canonical: torch.Tensor,
    depth_range: torch.Tensor,
) -> torch.Tensor:
    if final_depth_canonical.dim() == 3:
        final_depth_canonical = final_depth_canonical.unsqueeze(1)
    depth_min = depth_range[:, 0].view(-1, 1, 1, 1)
    depth_max = depth_range[:, 1].view(-1, 1, 1, 1)
    return final_depth_canonical * (depth_max - depth_min).clamp(min=1e-6) + depth_min


def _accumulate_depth_metrics(
    metrics_accum: Dict[str, float],
    prefix: str,
    final_depth_canonical: torch.Tensor,
    batch: dict,
) -> bool:
    """Accumulate metric-depth abs-rel/rmse/l1 when a metric target exists."""

    depth_gt, depth_range, mask = _prepare_depth_target(batch, final_depth_canonical.device)
    if depth_gt is None or depth_range is None:
        return False
    if final_depth_canonical.dim() == 3:
        final_depth_canonical = final_depth_canonical.unsqueeze(1)
    if final_depth_canonical.shape[-2:] != depth_gt.shape[-2:]:
        final_depth_canonical = F.interpolate(
            final_depth_canonical,
            size=depth_gt.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    pred_metric = _canonical_to_metric_depth(final_depth_canonical, depth_range)
    depth_metrics = compute_metrics(pred_metric.squeeze(1), depth_gt.squeeze(1), mask=mask)
    metrics_accum[f"{prefix}_abs_rel"] += depth_metrics["abs_rel"]
    metrics_accum[f"{prefix}_rmse"] += depth_metrics["rmse"]
    metrics_accum[f"{prefix}_l1"] += _masked_l1(pred_metric, depth_gt, mask).item()
    return True


def _final_depth_from_heads(
    *,
    trainer: "FocalStackGenerationTrainer",
    focal_stack: torch.Tensor,
    focal_plane_distances: torch.Tensor,
    clean_latent_pred: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Run decoder, focal-evidence head, and physical evidence fusion."""

    decoder_device, decoder_dtype = _module_device_dtype(
        trainer.task_output_decoder,
        clean_latent_pred.device,
        clean_latent_pred.dtype,
    )
    decoder_outputs = trainer.task_output_decoder(clean_latent_pred.to(device=decoder_device, dtype=decoder_dtype))
    generated_depth_canonical = decoder_outputs["generated_depth_canonical"]
    generative_uncertainty = decoder_outputs["uncertainty"]

    evidence_device, evidence_dtype = _module_device_dtype(
        trainer.focal_evidence_head,
        focal_stack.device,
        focal_stack.dtype,
    )
    focal_evidence = trainer.focal_evidence_head(
        focal_stack.to(device=evidence_device, dtype=evidence_dtype),
        focal_plane_distances.to(device=evidence_device, dtype=evidence_dtype),
    )
    focal_depth_canonical = F.interpolate(
        focal_evidence["focal_depth_canonical"],
        size=generated_depth_canonical.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).to(device=generated_depth_canonical.device, dtype=generated_depth_canonical.dtype)
    focal_entropy = F.interpolate(
        focal_evidence["focal_entropy"],
        size=generated_depth_canonical.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).to(device=generated_depth_canonical.device, dtype=generated_depth_canonical.dtype)
    focal_posterior = F.interpolate(
        focal_evidence["focal_posterior"],
        size=generated_depth_canonical.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).to(device=generated_depth_canonical.device, dtype=generated_depth_canonical.dtype)
    focal_posterior = focal_posterior / focal_posterior.sum(dim=1, keepdim=True).clamp(min=1e-6)

    support_inputs, support_maps = build_physical_evidence_features(
        focal_posterior=focal_posterior,
        focal_entropy=focal_entropy,
        focal_depth_canonical=focal_depth_canonical,
        generated_depth_canonical=generated_depth_canonical,
        generative_uncertainty=generative_uncertainty,
    )
    support_device, support_dtype = _module_device_dtype(
        trainer.physical_evidence_support_head,
        support_inputs.device,
        support_inputs.dtype,
    )
    support_outputs = trainer.physical_evidence_support_head(
        support_inputs.to(device=support_device, dtype=support_dtype)
    )
    focal_evidence_weight = support_outputs["focal_evidence_weight"]
    generative_prior_weight = support_outputs["generative_prior_weight"]
    gate_sum = (focal_evidence_weight + generative_prior_weight).clamp(min=1e-6)
    focal_evidence_weight_norm = focal_evidence_weight / gate_sum
    generative_prior_weight_norm = generative_prior_weight / gate_sum
    final_depth_canonical = (
        focal_evidence_weight_norm * focal_depth_canonical
        + generative_prior_weight_norm * generated_depth_canonical
    )

    head_outputs = {
        "generated_depth_canonical": generated_depth_canonical,
        "focal_depth_canonical": focal_depth_canonical,
        "focal_entropy": focal_entropy,
        "focal_posterior": focal_posterior,
        "generative_uncertainty": generative_uncertainty,
        "final_depth_canonical": final_depth_canonical,
        "focal_evidence_weight_norm": focal_evidence_weight_norm,
        "generative_prior_weight_norm": generative_prior_weight_norm,
    }
    return final_depth_canonical, head_outputs, support_outputs, support_maps


def run_teacher_forced_validation(trainer: "FocalStackGenerationTrainer", epoch: int) -> Dict[str, float]:
    """Validate the train-time tensor path without backpropagation.

    This path teacher-forces SD3 with clean all-in-focus VAE latents, adds
    scheduler noise, runs the transformer once, predicts clean latents, and then
    evaluates the decoder/focal-evidence/physical-evidence heads.
    """

    _ = epoch
    trainer.pipeline.eval()
    metrics = _zero_metrics(_TEACHER_FORCED_KEYS)
    metric_depth_batches = 0
    processed_batches = 0

    prompt_embeds, pooled_prompt_embeds = trainer._get_empty_prompt_embeddings()

    with torch.no_grad():
        for batch in tqdm(
            trainer.val_dataloader,
            desc="Teacher-forced validation",
            disable=not trainer.accelerator.is_local_main_process,
        ):
            if batch.get("all_in_focus") is None:
                continue

            device = trainer.accelerator.device
            focal_stack = batch["focal_stack"].to(device)
            focal_plane_distances = batch["focal_plane_distances"].to(device)
            rgb_gt = batch["all_in_focus"].to(device)

            _, focal_stack = _split_unit_and_signed_ranges(focal_stack.float())
            _, rgb_target = _split_unit_and_signed_ranges(rgb_gt.float())

            focal_features = trainer.focal_processor(focal_stack, focal_plane_distances)
            focal_features = {
                key: value.to(trainer.pipeline.transformer.dtype)
                if isinstance(value, torch.Tensor) and value is not None else value
                for key, value in focal_features.items()
            }

            vae_dtype = next(trainer.pipeline.vae.parameters()).dtype
            latents_dist = trainer.pipeline.vae.encode(rgb_target.to(dtype=vae_dtype)).latent_dist
            clean_latents = latents_dist.sample() * trainer.pipeline.vae.config.scaling_factor
            flow_batch = sample_sd3_flow_matching_batch(trainer.pipeline.scheduler, clean_latents)

            if hasattr(trainer.pipeline.scheduler, "scale_model_input"):
                model_input = trainer.pipeline.scheduler.scale_model_input(
                    flow_batch.noisy_latents,
                    flow_batch.timesteps,
                )
            else:
                model_input = flow_batch.noisy_latents
            model_input = model_input.to(trainer.pipeline.transformer.dtype)

            batch_prompt_embeds, batch_pooled_prompt_embeds = trainer._repeat_prompt_embeddings(
                prompt_embeds,
                pooled_prompt_embeds,
                batch_size=model_input.shape[0],
            )
            diffusion_pred = trainer.pipeline.transformer(
                hidden_states=model_input,
                timestep=flow_batch.timesteps,
                encoder_hidden_states=batch_prompt_embeds,
                pooled_projections=batch_pooled_prompt_embeds,
                focal_features=focal_features,
                return_dict=False,
            )[0]
            clean_latent_pred = predict_clean_latents_from_flow(
                noisy_latents=flow_batch.noisy_latents,
                model_pred=diffusion_pred,
                sigmas=flow_batch.sigmas,
            )
            final_depth_canonical, _, _, _ = _final_depth_from_heads(
                trainer=trainer,
                focal_stack=focal_stack,
                focal_plane_distances=focal_plane_distances,
                clean_latent_pred=clean_latent_pred,
            )

            if _accumulate_depth_metrics(metrics, "teacher_forced", final_depth_canonical, batch):
                metric_depth_batches += 1
            processed_batches += 1

    if metric_depth_batches > 0:
        for key in _TEACHER_FORCED_KEYS:
            metrics[key] /= metric_depth_batches

    if processed_batches == 0:
        return metrics
    return metrics


def run_generative_validation(trainer: "FocalStackGenerationTrainer", epoch: int) -> Dict[str, float]:
    """Validate the current pipeline sampling path."""

    _ = epoch
    trainer.pipeline.eval()
    metrics = {
        **_zero_metrics(_GENERATIVE_KEYS),
        'normalized_loss': 0.0,
        'focal_entropy_mean': 0.0,
        'physical_evidence_support_mean': 0.0,
        'focal_evidence_weight_mean': 0.0,
        'generative_prior_weight_mean': 0.0,
        'abstention_weight_mean': 0.0,
        'depth_disagreement_mean': 0.0,
        'posterior_margin_mean': 0.0,
        'uncertainty_final_mean': 0.0,
        'uncertainty_error_l1': 0.0,
        'generated_focal_depth_disagreement': 0.0,
        'uncertainty_mean': 0.0,
        'heldout_high_confidence_physical_violation_rate': 0.0,
        'heldout_selective_physical_risk_at_coverage': 0.0,
    }
    metric_depth_batches = 0
    normalized_depth_batches = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(
            trainer.val_dataloader,
            desc="Generative validation",
            disable=not trainer.accelerator.is_local_main_process,
        ):
            output = trainer.pipeline(
                focal_stack=batch['focal_stack'],
                focal_plane_distances=batch['focal_plane_distances'],
                num_inference_steps=trainer.config['validation']['num_inference_steps'],
                guidance_scale=trainer.config['validation']['guidance_scale'],
                output_type='pt',
                return_dict=True,
            )

            final_depth_canonical = output.depth_map
            final_depth_canonical_for_loss = (
                final_depth_canonical.unsqueeze(1)
                if final_depth_canonical.dim() == 3 else final_depth_canonical
            )
            uncertainty = getattr(output, "uncertainty_final", None)
            if uncertainty is None:
                uncertainty = getattr(output, "uncertainty", None)
            if uncertainty is not None:
                metrics["uncertainty_mean"] += uncertainty.mean().item()
            if output.focal_entropy is not None:
                metrics["focal_entropy_mean"] += output.focal_entropy.mean().item()
            if output.physical_evidence_support is not None:
                metrics["physical_evidence_support_mean"] += output.physical_evidence_support.mean().item()
            if output.generated_depth_canonical is not None and output.focal_depth_canonical is not None:
                metrics["generated_focal_depth_disagreement"] += torch.abs(
                    output.generated_depth_canonical - output.focal_depth_canonical
                ).mean().item()
            if output.focal_evidence_weight is not None:
                metrics["focal_evidence_weight_mean"] += output.focal_evidence_weight.mean().item()
            if output.generative_prior_weight is not None:
                metrics["generative_prior_weight_mean"] += output.generative_prior_weight.mean().item()
            if output.abstention_weight is not None:
                metrics["abstention_weight_mean"] += output.abstention_weight.mean().item()
            if output.depth_disagreement is not None:
                metrics["depth_disagreement_mean"] += output.depth_disagreement.mean().item()
            if output.posterior_margin is not None:
                metrics["posterior_margin_mean"] += output.posterior_margin.mean().item()
            if output.uncertainty_final is not None:
                metrics["uncertainty_final_mean"] += output.uncertainty_final.mean().item()
            trace = getattr(output, "physical_verification_trace", None)
            evaluation_verifier = getattr(trainer, "evaluation_verifier", None)
            if evaluation_verifier is not None and output.final_depth_canonical is not None:
                focal_stack_unit, _ = _split_unit_and_signed_ranges(batch["focal_stack"].to(final_depth_canonical.device).float())
                all_in_focus = output.all_in_focus_image
                if isinstance(all_in_focus, torch.Tensor):
                    trace = evaluation_verifier(
                        focal_stack=focal_stack_unit,
                        focal_plane_distances=batch["focal_plane_distances"].to(final_depth_canonical.device).float(),
                        depth_canonical=output.final_depth_canonical.unsqueeze(1) if output.final_depth_canonical.dim() == 3 else output.final_depth_canonical,
                        all_in_focus=all_in_focus.to(final_depth_canonical.device).float(),
                        generated_depth_canonical=output.generated_depth_canonical.unsqueeze(1) if output.generated_depth_canonical.dim() == 3 else output.generated_depth_canonical,
                    )
            if trace is not None:
                invalid = trace.invalid_score.detach().float().clamp(0.0, 1.0)
                conflict = trace.conflict_score.detach().float().clamp(0.0, 1.0)
                if uncertainty is None:
                    uncertainty_for_trace = torch.zeros_like(conflict)
                else:
                    uncertainty_for_trace = uncertainty.unsqueeze(1) if uncertainty.dim() == 3 else uncertainty
                    if uncertainty_for_trace.shape[-2:] != conflict.shape[-2:]:
                        uncertainty_for_trace = F.interpolate(uncertainty_for_trace.float(), size=conflict.shape[-2:], mode="bilinear", align_corners=False)
                heldout_cfg = trainer.config.get("validation", {}).get("heldout_verifier", {})
                confidence_threshold = float(heldout_cfg.get("confidence_threshold", 0.8))
                invalid_threshold = float(heldout_cfg.get("invalid_threshold", 0.5))
                conflict_threshold = float(heldout_cfg.get("conflict_threshold", 0.5))
                coverage = float(heldout_cfg.get("coverage", 0.2))
                confidence = (1.0 - uncertainty_for_trace.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
                high_confidence = confidence >= confidence_threshold
                physically_valid = invalid < invalid_threshold
                phr_den = (high_confidence & physically_valid).float().sum()
                phr_num = (high_confidence & physically_valid & (conflict >= conflict_threshold)).float().sum()
                metrics["heldout_high_confidence_physical_violation_rate"] += float("nan") if phr_den.item() == 0 else (phr_num / phr_den).item()
                sample_risks = []
                physical_risk = torch.maximum(conflict, invalid).clamp(0.0, 1.0)
                for sample_idx in range(confidence.shape[0]):
                    flat_confidence = confidence[sample_idx].flatten()
                    flat_risk = physical_risk[sample_idx].flatten()
                    coverage_count = max(1, min(flat_confidence.numel(), int(math.ceil(flat_confidence.numel() * coverage))))
                    top_indices = torch.topk(flat_confidence, k=coverage_count).indices
                    sample_risks.append(flat_risk[top_indices].mean())
                metrics["heldout_selective_physical_risk_at_coverage"] += torch.stack(sample_risks).mean().item()

            depth_gt, depth_range, mask = _prepare_depth_target(batch, final_depth_canonical.device)
            if depth_gt is not None and depth_range is not None:
                if _accumulate_depth_metrics(metrics, "generative", final_depth_canonical_for_loss, batch):
                    metric_depth_batches += 1
                depth_min = depth_range[:, 0].view(-1, 1, 1, 1)
                depth_max = depth_range[:, 1].view(-1, 1, 1, 1)
                final_depth_for_error = final_depth_canonical_for_loss
                if final_depth_for_error.shape[-2:] != depth_gt.shape[-2:]:
                    final_depth_for_error = F.interpolate(
                        final_depth_for_error,
                        size=depth_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if uncertainty is not None:
                    uncertainty_for_error = uncertainty.unsqueeze(1) if uncertainty.dim() == 3 else uncertainty
                    if uncertainty_for_error.shape[-2:] != final_depth_for_error.shape[-2:]:
                        uncertainty_for_error = F.interpolate(
                            uncertainty_for_error,
                            size=final_depth_for_error.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    error_norm = torch.abs(
                        final_depth_for_error.detach()
                        - ((depth_gt - depth_min) / (depth_max - depth_min).clamp(min=1e-6)).clamp(0.0, 1.0)
                    )
                    metrics["uncertainty_error_l1"] += _masked_l1(
                        uncertainty_for_error,
                        error_norm,
                        mask,
                    ).item()
            elif depth_gt is not None and depth_range is None:
                if final_depth_canonical_for_loss.shape[-2:] != depth_gt.shape[-2:]:
                    final_depth_canonical_for_loss = F.interpolate(
                        final_depth_canonical_for_loss,
                        size=depth_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                metrics['normalized_loss'] += F.l1_loss(final_depth_canonical_for_loss, depth_gt).item()
                normalized_depth_batches += 1

            num_batches += 1

    if num_batches == 0:
        return metrics

    for key in metrics:
        if key in _GENERATIVE_KEYS:
            metrics[key] = metrics[key] / max(metric_depth_batches, 1)
        elif key == "normalized_loss":
            metrics[key] = metrics[key] / max(normalized_depth_batches, 1)
        else:
            metrics[key] /= num_batches

    return metrics


def run_validation(trainer: "FocalStackGenerationTrainer", epoch: int) -> Dict[str, float]:
    """Run teacher-forced and generative validation and merge metrics."""

    teacher_metrics = run_teacher_forced_validation(trainer, epoch)
    generative_metrics = run_generative_validation(trainer, epoch)
    val_metrics = {**teacher_metrics, **generative_metrics}

    # Backward-compatible aliases for existing logging dashboards/checkpoint monitors.
    val_metrics["loss"] = val_metrics.get("generative_l1", 0.0)
    val_metrics["abs_rel"] = val_metrics.get("generative_abs_rel", 0.0)
    val_metrics["rmse"] = val_metrics.get("generative_rmse", 0.0)

    self_improvement_cfg = trainer.config.get("training", {}).get("self_improvement", {}) or {}
    if bool(self_improvement_cfg.get("enabled", False)):
        baseline = trainer.config.get("validation", {}).get("m0_baseline_metrics", {}) or {}
        val_metrics["m1_unlabeled_adaptation_hcpvr"] = val_metrics.get("heldout_high_confidence_physical_violation_rate", float("nan"))
        val_metrics["m1_unlabeled_adaptation_physical_aurc"] = val_metrics.get("heldout_selective_physical_risk_at_coverage", float("nan"))
        val_metrics["m1_unlabeled_adaptation_depth_l1"] = val_metrics.get("generative_l1", float("nan"))
        val_metrics["m1_source_domain_retention_l1"] = val_metrics.get("teacher_forced_l1", float("nan"))
        val_metrics["m1_independent_verifier_hcpvr"] = val_metrics.get("heldout_high_confidence_physical_violation_rate", float("nan"))
        val_metrics["delta_hcpvr"] = val_metrics["m1_unlabeled_adaptation_hcpvr"] - float(baseline.get("m0_zero_shot_hcpvr", val_metrics["m1_unlabeled_adaptation_hcpvr"]))
        val_metrics["delta_physical_aurc"] = val_metrics["m1_unlabeled_adaptation_physical_aurc"] - float(baseline.get("m0_zero_shot_physical_aurc", val_metrics["m1_unlabeled_adaptation_physical_aurc"]))
        val_metrics["delta_abs_rel"] = val_metrics.get("generative_abs_rel", float("nan")) - float(baseline.get("m0_zero_shot_abs_rel", val_metrics.get("generative_abs_rel", 0.0)))
        val_metrics["delta_psnr"] = val_metrics.get("aif_psnr", float("nan")) - float(baseline.get("m0_zero_shot_psnr", val_metrics.get("aif_psnr", 0.0)))
    else:
        val_metrics["m0_zero_shot_hcpvr"] = val_metrics.get("heldout_high_confidence_physical_violation_rate", float("nan"))
        val_metrics["m0_zero_shot_physical_aurc"] = val_metrics.get("heldout_selective_physical_risk_at_coverage", float("nan"))
        val_metrics["m0_zero_shot_abs_rel"] = val_metrics.get("generative_abs_rel", float("nan"))
        val_metrics["m0_zero_shot_psnr"] = val_metrics.get("aif_psnr", float("nan"))

    return val_metrics
