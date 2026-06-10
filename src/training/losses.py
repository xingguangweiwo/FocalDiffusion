"""Losses for FocalStackGeneration focal-evidence training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_focal_coordinates(focal_plane_distances: torch.Tensor) -> torch.Tensor:
    """Normalize per-sample focus distances to [0, 1]."""
    mn = focal_plane_distances.min(dim=1, keepdim=True).values
    mx = focal_plane_distances.max(dim=1, keepdim=True).values
    return (focal_plane_distances - mn) / (mx - mn + 1e-6)


def build_focal_axis_soft_targets(
    depth_norm: torch.Tensor,
    focal_plane_distances: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a depth-derived soft target posterior over focus planes."""
    focus_coordinates = normalize_focal_coordinates(focal_plane_distances)
    depth_hw = depth_norm.squeeze(1) if depth_norm.dim() == 4 else depth_norm
    logits = -torch.abs(focus_coordinates[:, :, None, None] - depth_hw[:, None]) / max(temperature, 1e-6)
    return torch.softmax(logits, dim=1), focus_coordinates


def _safe_denominator(value: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp denominator magnitudes while preserving sign where possible."""
    sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
    return torch.where(value.abs() < eps, sign * eps, value)


def _camera_param_to_broadcast(
    value: torch.Tensor | float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    """Convert scalar/[B]/[B,1] camera params to [B,1,1,1]."""
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.dim() == 0:
        tensor = tensor.expand(batch_size)
    else:
        tensor = tensor.reshape(tensor.shape[0], -1)[:, 0]
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size)
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"Camera parameter batch {tensor.shape[0]} does not match depth batch {batch_size}.")
    return tensor.clamp(min=eps).view(batch_size, 1, 1, 1)


def build_coc_posterior_targets(
    depth_metric: torch.Tensor,
    focal_plane_distances: torch.Tensor,
    camera_params: dict[str, torch.Tensor],
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a CoC-derived soft target posterior over metric focus planes."""
    if "focal_length" not in camera_params or "pixel_size" not in camera_params:
        raise ValueError("camera_params must contain focal_length and pixel_size.")
    if "f_number" not in camera_params and "aperture" not in camera_params:
        raise ValueError("camera_params must contain either f_number or aperture.")

    depth = depth_metric.squeeze(1) if depth_metric.dim() == 4 else depth_metric
    if depth.dim() != 3:
        raise ValueError(f"depth_metric must have shape [B,1,H,W] or [B,H,W], got {tuple(depth_metric.shape)}.")

    batch_size = depth.shape[0]
    device = depth.device
    dtype = depth.dtype
    focus = focal_plane_distances.to(device=device, dtype=dtype)
    if focus.dim() == 1:
        focus = focus.unsqueeze(0).expand(batch_size, -1)
    elif focus.dim() == 2:
        if focus.shape[0] == 1 and batch_size != 1:
            focus = focus.expand(batch_size, -1)
        elif focus.shape[0] != batch_size:
            raise ValueError(f"focal_plane_distances batch {focus.shape[0]} does not match depth batch {batch_size}.")
    else:
        raise ValueError(f"focal_plane_distances must have shape [N] or [B,N], got {tuple(focal_plane_distances.shape)}.")

    focal_length = _camera_param_to_broadcast(camera_params["focal_length"], batch_size, device, dtype, eps)
    pixel_size = _camera_param_to_broadcast(camera_params["pixel_size"], batch_size, device, dtype, eps)
    if "aperture" in camera_params:
        aperture = _camera_param_to_broadcast(camera_params["aperture"], batch_size, device, dtype, eps)
    else:
        f_number = _camera_param_to_broadcast(camera_params["f_number"], batch_size, device, dtype, eps)
        aperture = focal_length / f_number

    focus = focus[:, :, None, None]
    depth = depth[:, None].clamp(min=eps)
    v_focus = focus * focal_length / _safe_denominator(focus - focal_length, eps)
    coc = aperture * v_focus.abs() * (1.0 / focal_length - 1.0 / _safe_denominator(v_focus, eps) - 1.0 / depth).abs()
    coc_pixels = coc / pixel_size
    focus_target = torch.softmax(-coc_pixels / max(float(temperature), eps), dim=1)
    return focus_target, coc_pixels


# Structure-guided focal evidence regularization implemented as training-time
# consistency losses. It uses local image affinity from the focal stack or AIF
# target, without external foundation models or inference-time overhead.
def _resize_and_normalize_posterior(posterior: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a focus posterior and re-normalize across focal planes."""
    if posterior.shape[-2:] == size:
        return posterior
    posterior = F.interpolate(posterior, size=size, mode="bilinear", align_corners=False)
    return posterior / posterior.sum(dim=1, keepdim=True).clamp(min=1e-6)


def _build_evidence_image(
    focal_stack: torch.Tensor,
    focal_posterior: torch.Tensor,
    rgb_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build an image-like reference for local affinity computation."""
    if rgb_target is not None:
        return rgb_target.detach()

    posterior = focal_posterior.detach()
    if posterior.shape[-2:] != focal_stack.shape[-2:]:
        posterior = _resize_and_normalize_posterior(posterior, focal_stack.shape[-2:])
    evidence_image = (posterior[:, :, None] * focal_stack).sum(dim=1)
    return evidence_image.detach()


def _compute_local_affinity(evidence_image: torch.Tensor, sigma: float = 0.10) -> dict[str, torch.Tensor]:
    """Compute local x/y affinities without forming a global pixel Gram matrix."""
    sigma = max(float(sigma), 1e-6)
    image = evidence_image.detach()
    diff_x = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    diff_y = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    sim_x = torch.exp(-diff_x / sigma)
    sim_y = torch.exp(-diff_y / sigma)
    return {"x": sim_x.detach(), "y": sim_y.detach()}


def _pairwise_valid_mask(mask: torch.Tensor | None, direction: str) -> torch.Tensor | None:
    """Convert a dense valid mask into pairwise x/y valid weights."""
    if mask is None:
        return None
    mask = mask.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if direction == "x":
        return mask[:, :, :, 1:] * mask[:, :, :, :-1]
    if direction == "y":
        return mask[:, :, 1:, :] * mask[:, :, :-1, :]
    raise ValueError(f"Unsupported pairwise direction: {direction}")


def _weighted_pairwise_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average pairwise values with affinity and optional validity weights."""
    weight = weight.to(device=value.device, dtype=value.dtype)
    if valid is not None:
        valid = valid.to(device=value.device, dtype=value.dtype)
        weight = weight * valid
    return (value * weight).sum() / weight.sum().clamp(min=1.0)


def _posterior_consistency_loss(
    focal_posterior: torch.Tensor,
    affinity: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encourage similar local regions to keep consistent focus posteriors."""
    diff_x = (focal_posterior[:, :, :, 1:] - focal_posterior[:, :, :, :-1]).abs().sum(dim=1, keepdim=True)
    diff_y = (focal_posterior[:, :, 1:, :] - focal_posterior[:, :, :-1, :]).abs().sum(dim=1, keepdim=True)
    loss_x = _weighted_pairwise_mean(diff_x, affinity["x"], _pairwise_valid_mask(valid_mask, "x"))
    loss_y = _weighted_pairwise_mean(diff_y, affinity["y"], _pairwise_valid_mask(valid_mask, "y"))
    return 0.5 * (loss_x + loss_y)


def _depth_affinity_smoothness_loss(
    final_depth_canonical: torch.Tensor,
    affinity: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth normalized depth within locally similar regions while preserving edges."""
    diff_x = (final_depth_canonical[:, :, :, 1:] - final_depth_canonical[:, :, :, :-1]).abs()
    diff_y = (final_depth_canonical[:, :, 1:, :] - final_depth_canonical[:, :, :-1, :]).abs()
    loss_x = _weighted_pairwise_mean(diff_x, affinity["x"], _pairwise_valid_mask(valid_mask, "x"))
    loss_y = _weighted_pairwise_mean(diff_y, affinity["y"], _pairwise_valid_mask(valid_mask, "y"))
    return 0.5 * (loss_x + loss_y)


def _gate_consistency_loss(
    focal_evidence_weight: torch.Tensor,
    affinity: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discourage random local jitter in focus fusion gates."""
    diff_x = (focal_evidence_weight[:, :, :, 1:] - focal_evidence_weight[:, :, :, :-1]).abs()
    diff_y = (focal_evidence_weight[:, :, 1:, :] - focal_evidence_weight[:, :, :-1, :]).abs()
    loss_x = _weighted_pairwise_mean(diff_x, affinity["x"], _pairwise_valid_mask(valid_mask, "x"))
    loss_y = _weighted_pairwise_mean(diff_y, affinity["y"], _pairwise_valid_mask(valid_mask, "y"))
    return 0.5 * (loss_x + loss_y)


def _focal_axis_smoothness_loss(focal_posterior: torch.Tensor) -> torch.Tensor:
    """Apply a small curvature penalty along the focal-plane axis."""
    if focal_posterior.shape[1] < 3:
        return focal_posterior.new_tensor(0.0)
    curvature = focal_posterior[:, 2:] - 2 * focal_posterior[:, 1:-1] + focal_posterior[:, :-2]
    return curvature.abs().mean()



class FocalStackGenerationLoss(nn.Module):
    """Main FEP training objective for flow, depth, evidence, AIF and uncertainty."""

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        depth_weight: float = 0.0,
        rgb_weight: float = 0.0,
        supervision_mode: str = "supervised",
        focal_posterior_kl_weight: float = 0.2,
        focus_depth_weight: float = 0.2,
        prior_depth_weight: float = 0.05,
        all_in_focus_focal_evidence_weight: float = 0.1,
        uncertainty_focus_weight: float = 0.05,
        uncertainty_error_weight: float = 0.05,
        focus_target_temperature: float = 0.07,
        focal_target_type: str = "normalized",
        coc_posterior_temperature: float = 1.0,
        gate_calibration_weight: float = 0.05,
        posterior_consistency_weight: float = 0.02,
        depth_affinity_smoothness_weight: float = 0.01,
        gate_consistency_weight: float = 0.0,
        focal_axis_smoothness_weight: float = 0.0,
        local_affinity_sigma: float = 0.10,
    ):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.depth_weight = depth_weight
        self.rgb_weight = rgb_weight
        self.supervision_mode = supervision_mode
        self.focal_posterior_kl_weight = focal_posterior_kl_weight
        self.focus_depth_weight = focus_depth_weight
        self.prior_depth_weight = prior_depth_weight
        self.all_in_focus_focal_evidence_weight = all_in_focus_focal_evidence_weight
        self.uncertainty_focus_weight = uncertainty_focus_weight
        self.uncertainty_error_weight = uncertainty_error_weight
        if focal_target_type not in {"normalized", "coc"}:
            raise ValueError(f"Unsupported focal_target_type: {focal_target_type}")
        self.focus_target_temperature = focus_target_temperature
        self.focal_target_type = focal_target_type
        self.coc_posterior_temperature = coc_posterior_temperature
        self.gate_calibration_weight = gate_calibration_weight
        self.posterior_consistency_weight = posterior_consistency_weight
        self.depth_affinity_smoothness_weight = depth_affinity_smoothness_weight
        self.gate_consistency_weight = gate_consistency_weight
        self.focal_axis_smoothness_weight = focal_axis_smoothness_weight
        self.local_affinity_sigma = local_affinity_sigma

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return value.mean()
        if mask.dim() == value.dim() - 1:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=value.device, dtype=value.dtype)
        return (value * mask).sum() / mask.sum().clamp(min=1.0)

    def forward(
        self,
        diffusion_pred: torch.Tensor,
        diffusion_target: torch.Tensor,
        depth_target: torch.Tensor | None = None,
        rgb_pred: torch.Tensor | None = None,
        rgb_target: torch.Tensor | None = None,
        generated_depth_canonical: torch.Tensor | None = None,
        focal_depth_canonical: torch.Tensor | None = None,
        final_depth_canonical: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
        focal_posterior: torch.Tensor | None = None,
        focal_entropy: torch.Tensor | None = None,
        focal_plane_distances: torch.Tensor | None = None,
        focal_stack: torch.Tensor | None = None,
        depth_mask: torch.Tensor | None = None,
        depth_range: torch.Tensor | None = None,
        focal_evidence_weight: torch.Tensor | None = None,
        generative_prior_weight: torch.Tensor | None = None,
        abstention_weight: torch.Tensor | None = None,
        physical_evidence_support: torch.Tensor | None = None,
        camera_params: dict[str, torch.Tensor] | None = None,
    ):
        del generative_prior_weight, abstention_weight, physical_evidence_support

        losses: dict[str, torch.Tensor] = {
            "loss_flow_matching": F.mse_loss(diffusion_pred, diffusion_target),
        }

        enable_supervised = self.supervision_mode in {"supervised", "semi_supervised"}
        depth_gt_norm = None
        mask = None
        if enable_supervised and depth_target is not None and depth_range is not None and final_depth_canonical is not None:
            depth_final_resized = F.interpolate(final_depth_canonical, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
            depth_min = depth_range[:, 0].view(-1, 1, 1, 1)
            depth_max = depth_range[:, 1].view(-1, 1, 1, 1)
            depth_gt_norm = ((depth_target - depth_min) / (depth_max - depth_min).clamp(min=1e-6)).clamp(0.0, 1.0)
            if depth_mask is not None:
                mask = depth_mask.unsqueeze(1) if depth_mask.dim() == depth_final_resized.dim() - 1 else depth_mask
            losses["loss_depth_final"] = self._masked_mean(torch.abs(depth_final_resized - depth_gt_norm), mask)

            if focal_depth_canonical is not None:
                depth_focus_resized = F.interpolate(focal_depth_canonical, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                losses["loss_depth_focus"] = self._masked_mean(torch.abs(depth_focus_resized - depth_gt_norm), mask)
            if generated_depth_canonical is not None:
                depth_prior_resized = F.interpolate(generated_depth_canonical, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                losses["loss_depth_prior"] = self._masked_mean(torch.abs(depth_prior_resized - depth_gt_norm), mask)
            if focal_posterior is not None and focal_plane_distances is not None:
                posterior_resized = focal_posterior
                if posterior_resized.shape[-2:] != depth_target.shape[-2:]:
                    posterior_resized = F.interpolate(posterior_resized, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                    posterior_resized = posterior_resized / posterior_resized.sum(dim=1, keepdim=True).clamp(min=1e-6)
                focus_target = None
                if self.focal_target_type == "normalized":
                    focus_target, _ = build_focal_axis_soft_targets(
                        depth_gt_norm,
                        focal_plane_distances,
                        temperature=self.focus_target_temperature,
                    )
                elif camera_params is not None:
                    focus_target, _ = build_coc_posterior_targets(
                        depth_target,
                        focal_plane_distances,
                        camera_params,
                        temperature=self.coc_posterior_temperature,
                    )
                if focus_target is not None:
                    focus_target = focus_target.to(device=posterior_resized.device, dtype=posterior_resized.dtype)
                    kl = focus_target * (torch.log(focus_target + 1e-6) - torch.log(posterior_resized + 1e-6))
                    losses["loss_focal_posterior_kl"] = self._masked_mean(kl.sum(dim=1, keepdim=True), mask)
            if uncertainty is not None and final_depth_canonical is not None:
                uncertainty_resized = F.interpolate(uncertainty, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                error_norm = torch.abs(depth_final_resized.detach() - depth_gt_norm)
                losses["loss_uncertainty_error"] = self._masked_mean(torch.abs(uncertainty_resized - error_norm), mask)
            if focal_evidence_weight is not None and focal_depth_canonical is not None and generated_depth_canonical is not None:
                focus_for_gate = focal_depth_canonical.detach()
                prior_for_gate = generated_depth_canonical.detach()
                if focus_for_gate.shape[-2:] != depth_target.shape[-2:]:
                    focus_for_gate = F.interpolate(focus_for_gate, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                if prior_for_gate.shape[-2:] != depth_target.shape[-2:]:
                    prior_for_gate = F.interpolate(prior_for_gate, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                err_focus = torch.abs(focus_for_gate - depth_gt_norm)
                err_prior = torch.abs(prior_for_gate - depth_gt_norm)
                target_focus = (err_focus < err_prior).float()
                pred_focus = focal_evidence_weight
                if pred_focus.shape[-2:] != target_focus.shape[-2:]:
                    pred_focus = F.interpolate(pred_focus, size=target_focus.shape[-2:], mode="bilinear", align_corners=False)
                loss_focal_evidence_weight = F.binary_cross_entropy(
                    pred_focus.clamp(1e-6, 1.0 - 1e-6),
                    target_focus,
                    reduction="none",
                )
                losses["loss_focal_evidence_weight"] = self._masked_mean(loss_focal_evidence_weight, mask)

        if enable_supervised and rgb_pred is not None and rgb_target is not None:
            losses["loss_rgb_reconstruction"] = F.l1_loss(rgb_pred, rgb_target)

        if focal_posterior is not None and focal_stack is not None and rgb_pred is not None:
            posterior_for_all_in_focus = focal_posterior
            if posterior_for_all_in_focus.shape[-2:] != focal_stack.shape[-2:]:
                posterior_for_all_in_focus = F.interpolate(posterior_for_all_in_focus, size=focal_stack.shape[-2:], mode="bilinear", align_corners=False)
                posterior_for_all_in_focus = posterior_for_all_in_focus / posterior_for_all_in_focus.sum(dim=1, keepdim=True).clamp(min=1e-6)
            all_in_focus_from_focal_evidence = (posterior_for_all_in_focus[:, :, None] * focal_stack).sum(dim=1)
            rgb_pred_resized = rgb_pred
            if rgb_pred_resized.shape[-2:] != all_in_focus_from_focal_evidence.shape[-2:]:
                rgb_pred_resized = F.interpolate(rgb_pred_resized, size=all_in_focus_from_focal_evidence.shape[-2:], mode="bilinear", align_corners=False)
            hp_pred = rgb_pred_resized - F.avg_pool2d(rgb_pred_resized, 5, 1, 2)
            all_in_focus_from_focal_evidence_detached = all_in_focus_from_focal_evidence.detach()
            hp_focus = all_in_focus_from_focal_evidence_detached - F.avg_pool2d(all_in_focus_from_focal_evidence_detached, 5, 1, 2)
            losses["loss_all_in_focus_focal_consistency"] = F.l1_loss(hp_pred, hp_focus)

        if uncertainty is not None and focal_entropy is not None:
            uncertainty_resized = uncertainty
            focal_entropy_target = focal_entropy.detach()
            if uncertainty_resized.shape[-2:] != focal_entropy_target.shape[-2:]:
                uncertainty_resized = F.interpolate(uncertainty_resized, size=focal_entropy_target.shape[-2:], mode="bilinear", align_corners=False)
            losses["loss_uncertainty_focus"] = F.l1_loss(uncertainty_resized, focal_entropy_target)

        if focal_stack is not None and focal_posterior is not None and final_depth_canonical is not None:
            evidence_image = _build_evidence_image(
                focal_stack=focal_stack,
                focal_posterior=focal_posterior,
                rgb_target=rgb_target,
            )
            if evidence_image.shape[-2:] != final_depth_canonical.shape[-2:]:
                evidence_image = F.interpolate(
                    evidence_image,
                    size=final_depth_canonical.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).detach()

            if focal_posterior.shape[-2:] != final_depth_canonical.shape[-2:]:
                posterior_for_reg = _resize_and_normalize_posterior(focal_posterior, final_depth_canonical.shape[-2:])
            else:
                posterior_for_reg = focal_posterior

            valid_mask_for_reg = depth_mask
            if valid_mask_for_reg is not None and valid_mask_for_reg.shape[-2:] != final_depth_canonical.shape[-2:]:
                if valid_mask_for_reg.dim() == 3:
                    valid_mask_for_reg = valid_mask_for_reg.unsqueeze(1)
                valid_mask_for_reg = F.interpolate(
                    valid_mask_for_reg.float(),
                    size=final_depth_canonical.shape[-2:],
                    mode="nearest",
                )

            affinity = _compute_local_affinity(
                evidence_image,
                sigma=self.local_affinity_sigma,
            )
            losses["loss_posterior_consistency"] = _posterior_consistency_loss(
                posterior_for_reg,
                affinity,
                valid_mask=valid_mask_for_reg,
            )
            losses["loss_depth_affinity_smoothness"] = _depth_affinity_smoothness_loss(
                final_depth_canonical,
                affinity,
                valid_mask=valid_mask_for_reg,
            )

            if focal_evidence_weight is not None:
                if focal_evidence_weight.shape[-2:] != final_depth_canonical.shape[-2:]:
                    gate_for_reg = F.interpolate(
                        focal_evidence_weight,
                        size=final_depth_canonical.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    gate_for_reg = focal_evidence_weight
                losses["loss_gate_consistency"] = _gate_consistency_loss(
                    gate_for_reg,
                    affinity,
                    valid_mask=valid_mask_for_reg,
                )

            losses["loss_focal_axis_smoothness"] = _focal_axis_smoothness_loss(
                posterior_for_reg,
            )

        total = torch.zeros_like(losses["loss_flow_matching"])
        total = total + self.diffusion_weight * losses["loss_flow_matching"]
        total = total + self.depth_weight * losses.get("loss_depth_final", torch.zeros_like(total))
        total = total + self.focal_posterior_kl_weight * losses.get("loss_focal_posterior_kl", torch.zeros_like(total))
        total = total + self.focus_depth_weight * losses.get("loss_depth_focus", torch.zeros_like(total))
        total = total + self.prior_depth_weight * losses.get("loss_depth_prior", torch.zeros_like(total))
        total = total + self.rgb_weight * losses.get("loss_rgb_reconstruction", torch.zeros_like(total))
        total = total + self.all_in_focus_focal_evidence_weight * losses.get("loss_all_in_focus_focal_consistency", torch.zeros_like(total))
        total = total + self.uncertainty_focus_weight * losses.get("loss_uncertainty_focus", torch.zeros_like(total))
        total = total + self.uncertainty_error_weight * losses.get("loss_uncertainty_error", torch.zeros_like(total))
        total = total + self.gate_calibration_weight * losses.get("loss_focal_evidence_weight", torch.zeros_like(total))
        total = total + self.posterior_consistency_weight * losses.get(
            "loss_posterior_consistency",
            torch.zeros_like(total),
        )
        total = total + self.depth_affinity_smoothness_weight * losses.get(
            "loss_depth_affinity_smoothness",
            torch.zeros_like(total),
        )
        total = total + self.gate_consistency_weight * losses.get(
            "loss_gate_consistency",
            torch.zeros_like(total),
        )
        total = total + self.focal_axis_smoothness_weight * losses.get(
            "loss_focal_axis_smoothness",
            torch.zeros_like(total),
        )

        return {**losses, "total": total}
