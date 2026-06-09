"""Losses for FSDiffusion focal-evidence training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_focus_coordinates(focus_distances: torch.Tensor) -> torch.Tensor:
    """Normalize per-sample focus distances to [0, 1]."""
    mn = focus_distances.min(dim=1, keepdim=True).values
    mx = focus_distances.max(dim=1, keepdim=True).values
    return (focus_distances - mn) / (mx - mn + 1e-6)


def build_soft_focus_target_from_depth(
    depth_norm: torch.Tensor,
    focus_distances: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a depth-derived soft target posterior over focus planes."""
    focus_coordinates = normalize_focus_coordinates(focus_distances)
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


def build_coc_focus_target_from_depth(
    depth_metric: torch.Tensor,
    focus_distances: torch.Tensor,
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
    focus = focus_distances.to(device=device, dtype=dtype)
    if focus.dim() == 1:
        focus = focus.unsqueeze(0).expand(batch_size, -1)
    elif focus.dim() == 2:
        if focus.shape[0] == 1 and batch_size != 1:
            focus = focus.expand(batch_size, -1)
        elif focus.shape[0] != batch_size:
            raise ValueError(f"focus_distances batch {focus.shape[0]} does not match depth batch {batch_size}.")
    else:
        raise ValueError(f"focus_distances must have shape [N] or [B,N], got {tuple(focus_distances.shape)}.")

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
    focus_posterior: torch.Tensor,
    rgb_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build an image-like reference for local affinity computation."""
    if rgb_target is not None:
        return rgb_target.detach()

    posterior = focus_posterior.detach()
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
    focus_posterior: torch.Tensor,
    affinity: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encourage similar local regions to keep consistent focus posteriors."""
    diff_x = (focus_posterior[:, :, :, 1:] - focus_posterior[:, :, :, :-1]).abs().sum(dim=1, keepdim=True)
    diff_y = (focus_posterior[:, :, 1:, :] - focus_posterior[:, :, :-1, :]).abs().sum(dim=1, keepdim=True)
    loss_x = _weighted_pairwise_mean(diff_x, affinity["x"], _pairwise_valid_mask(valid_mask, "x"))
    loss_y = _weighted_pairwise_mean(diff_y, affinity["y"], _pairwise_valid_mask(valid_mask, "y"))
    return 0.5 * (loss_x + loss_y)


def _depth_affinity_smoothness_loss(
    depth_final_norm: torch.Tensor,
    affinity: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth normalized depth within locally similar regions while preserving edges."""
    diff_x = (depth_final_norm[:, :, :, 1:] - depth_final_norm[:, :, :, :-1]).abs()
    diff_y = (depth_final_norm[:, :, 1:, :] - depth_final_norm[:, :, :-1, :]).abs()
    loss_x = _weighted_pairwise_mean(diff_x, affinity["x"], _pairwise_valid_mask(valid_mask, "x"))
    loss_y = _weighted_pairwise_mean(diff_y, affinity["y"], _pairwise_valid_mask(valid_mask, "y"))
    return 0.5 * (loss_x + loss_y)


def _gate_consistency_loss(
    gate_focus: torch.Tensor,
    affinity: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discourage random local jitter in focus fusion gates."""
    diff_x = (gate_focus[:, :, :, 1:] - gate_focus[:, :, :, :-1]).abs()
    diff_y = (gate_focus[:, :, 1:, :] - gate_focus[:, :, :-1, :]).abs()
    loss_x = _weighted_pairwise_mean(diff_x, affinity["x"], _pairwise_valid_mask(valid_mask, "x"))
    loss_y = _weighted_pairwise_mean(diff_y, affinity["y"], _pairwise_valid_mask(valid_mask, "y"))
    return 0.5 * (loss_x + loss_y)


def _focal_axis_smoothness_loss(focus_posterior: torch.Tensor) -> torch.Tensor:
    """Apply a small curvature penalty along the focal-plane axis."""
    if focus_posterior.shape[1] < 3:
        return focus_posterior.new_tensor(0.0)
    curvature = focus_posterior[:, 2:] - 2 * focus_posterior[:, 1:-1] + focus_posterior[:, :-2]
    return curvature.abs().mean()



class FocalDiffusionLoss(nn.Module):
    """Main FEP training objective for flow, depth, evidence, AIF and uncertainty."""

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        depth_weight: float = 0.0,
        rgb_weight: float = 0.0,
        supervision_mode: str = "supervised",
        focus_posterior_kl_weight: float = 0.2,
        focus_depth_weight: float = 0.2,
        prior_depth_weight: float = 0.05,
        aif_focus_evidence_weight: float = 0.1,
        uncertainty_focus_weight: float = 0.05,
        uncertainty_error_weight: float = 0.05,
        focus_target_temperature: float = 0.07,
        focus_target_type: str = "normalized",
        coc_target_temperature: float = 1.0,
        gate_calibration_weight: float = 0.05,
        posterior_consistency_weight: float = 0.02,
        depth_affinity_smoothness_weight: float = 0.01,
        gate_consistency_weight: float = 0.0,
        focal_axis_smoothness_weight: float = 0.0,
        local_affinity_sigma: float = 0.10,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.diffusion_weight = diffusion_weight
        self.depth_weight = depth_weight
        self.rgb_weight = rgb_weight
        self.supervision_mode = supervision_mode
        self.focus_posterior_kl_weight = focus_posterior_kl_weight
        self.focus_depth_weight = focus_depth_weight
        self.prior_depth_weight = prior_depth_weight
        self.aif_focus_evidence_weight = aif_focus_evidence_weight
        self.uncertainty_focus_weight = uncertainty_focus_weight
        self.uncertainty_error_weight = uncertainty_error_weight
        if focus_target_type not in {"normalized", "coc"}:
            raise ValueError(f"Unsupported focus_target_type: {focus_target_type}")
        self.focus_target_temperature = focus_target_temperature
        self.focus_target_type = focus_target_type
        self.coc_target_temperature = coc_target_temperature
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
        depth_prior_norm: torch.Tensor | None = None,
        depth_focus_norm: torch.Tensor | None = None,
        depth_final_norm: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
        focus_posterior: torch.Tensor | None = None,
        focus_entropy: torch.Tensor | None = None,
        focus_reliability: torch.Tensor | None = None,
        focus_distances: torch.Tensor | None = None,
        focal_stack: torch.Tensor | None = None,
        depth_mask: torch.Tensor | None = None,
        depth_range: torch.Tensor | None = None,
        gate_focus: torch.Tensor | None = None,
        gate_prior: torch.Tensor | None = None,
        gate_abstain: torch.Tensor | None = None,
        physical_support: torch.Tensor | None = None,
        camera_params: dict[str, torch.Tensor] | None = None,
    ):
        del focus_reliability, gate_prior, gate_abstain, physical_support

        losses: dict[str, torch.Tensor] = {
            "loss_flow_matching": F.mse_loss(diffusion_pred, diffusion_target),
        }

        enable_supervised = self.supervision_mode in {"supervised", "semi_supervised"}
        depth_gt_norm = None
        mask = None
        if enable_supervised and depth_target is not None and depth_range is not None and depth_final_norm is not None:
            depth_final_resized = F.interpolate(depth_final_norm, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
            depth_min = depth_range[:, 0].view(-1, 1, 1, 1)
            depth_max = depth_range[:, 1].view(-1, 1, 1, 1)
            depth_gt_norm = ((depth_target - depth_min) / (depth_max - depth_min).clamp(min=1e-6)).clamp(0.0, 1.0)
            if depth_mask is not None:
                mask = depth_mask.unsqueeze(1) if depth_mask.dim() == depth_final_resized.dim() - 1 else depth_mask
            losses["loss_depth_final"] = self._masked_mean(torch.abs(depth_final_resized - depth_gt_norm), mask)

            if depth_focus_norm is not None:
                depth_focus_resized = F.interpolate(depth_focus_norm, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                losses["loss_depth_focus"] = self._masked_mean(torch.abs(depth_focus_resized - depth_gt_norm), mask)
            if depth_prior_norm is not None:
                depth_prior_resized = F.interpolate(depth_prior_norm, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                losses["loss_depth_prior"] = self._masked_mean(torch.abs(depth_prior_resized - depth_gt_norm), mask)
            if focus_posterior is not None and focus_distances is not None:
                posterior_resized = focus_posterior
                if posterior_resized.shape[-2:] != depth_target.shape[-2:]:
                    posterior_resized = F.interpolate(posterior_resized, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                    posterior_resized = posterior_resized / posterior_resized.sum(dim=1, keepdim=True).clamp(min=1e-6)
                focus_target = None
                if self.focus_target_type == "normalized":
                    focus_target, _ = build_soft_focus_target_from_depth(
                        depth_gt_norm,
                        focus_distances,
                        temperature=self.focus_target_temperature,
                    )
                elif camera_params is not None:
                    focus_target, _ = build_coc_focus_target_from_depth(
                        depth_target,
                        focus_distances,
                        camera_params,
                        temperature=self.coc_target_temperature,
                    )
                if focus_target is not None:
                    focus_target = focus_target.to(device=posterior_resized.device, dtype=posterior_resized.dtype)
                    kl = focus_target * (torch.log(focus_target + 1e-6) - torch.log(posterior_resized + 1e-6))
                    losses["loss_focus_posterior_kl"] = self._masked_mean(kl.sum(dim=1, keepdim=True), mask)
            if uncertainty is not None and depth_final_norm is not None:
                uncertainty_resized = F.interpolate(uncertainty, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                error_norm = torch.abs(depth_final_resized.detach() - depth_gt_norm)
                losses["loss_uncertainty_error"] = self._masked_mean(torch.abs(uncertainty_resized - error_norm), mask)
            if gate_focus is not None and depth_focus_norm is not None and depth_prior_norm is not None:
                focus_for_gate = depth_focus_norm.detach()
                prior_for_gate = depth_prior_norm.detach()
                if focus_for_gate.shape[-2:] != depth_target.shape[-2:]:
                    focus_for_gate = F.interpolate(focus_for_gate, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                if prior_for_gate.shape[-2:] != depth_target.shape[-2:]:
                    prior_for_gate = F.interpolate(prior_for_gate, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                err_focus = torch.abs(focus_for_gate - depth_gt_norm)
                err_prior = torch.abs(prior_for_gate - depth_gt_norm)
                target_focus = (err_focus < err_prior).float()
                pred_focus = gate_focus
                if pred_focus.shape[-2:] != target_focus.shape[-2:]:
                    pred_focus = F.interpolate(pred_focus, size=target_focus.shape[-2:], mode="bilinear", align_corners=False)
                loss_gate_focus = F.binary_cross_entropy(
                    pred_focus.clamp(1e-6, 1.0 - 1e-6),
                    target_focus,
                    reduction="none",
                )
                losses["loss_gate_focus"] = self._masked_mean(loss_gate_focus, mask)

        if enable_supervised and rgb_pred is not None and rgb_target is not None:
            losses["loss_rgb_reconstruction"] = F.l1_loss(rgb_pred, rgb_target)

        if focus_posterior is not None and focal_stack is not None and rgb_pred is not None:
            posterior_for_aif = focus_posterior
            if posterior_for_aif.shape[-2:] != focal_stack.shape[-2:]:
                posterior_for_aif = F.interpolate(posterior_for_aif, size=focal_stack.shape[-2:], mode="bilinear", align_corners=False)
                posterior_for_aif = posterior_for_aif / posterior_for_aif.sum(dim=1, keepdim=True).clamp(min=1e-6)
            aif_focus = (posterior_for_aif[:, :, None] * focal_stack).sum(dim=1)
            rgb_pred_resized = rgb_pred
            if rgb_pred_resized.shape[-2:] != aif_focus.shape[-2:]:
                rgb_pred_resized = F.interpolate(rgb_pred_resized, size=aif_focus.shape[-2:], mode="bilinear", align_corners=False)
            hp_pred = rgb_pred_resized - F.avg_pool2d(rgb_pred_resized, 5, 1, 2)
            aif_focus_detached = aif_focus.detach()
            hp_focus = aif_focus_detached - F.avg_pool2d(aif_focus_detached, 5, 1, 2)
            losses["loss_aif_focus_consistency"] = F.l1_loss(hp_pred, hp_focus)

        if uncertainty is not None and focus_entropy is not None:
            uncertainty_resized = uncertainty
            focus_entropy_target = focus_entropy.detach()
            if uncertainty_resized.shape[-2:] != focus_entropy_target.shape[-2:]:
                uncertainty_resized = F.interpolate(uncertainty_resized, size=focus_entropy_target.shape[-2:], mode="bilinear", align_corners=False)
            losses["loss_uncertainty_focus"] = F.l1_loss(uncertainty_resized, focus_entropy_target)

        if focal_stack is not None and focus_posterior is not None and depth_final_norm is not None:
            evidence_image = _build_evidence_image(
                focal_stack=focal_stack,
                focus_posterior=focus_posterior,
                rgb_target=rgb_target,
            )
            if evidence_image.shape[-2:] != depth_final_norm.shape[-2:]:
                evidence_image = F.interpolate(
                    evidence_image,
                    size=depth_final_norm.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).detach()

            if focus_posterior.shape[-2:] != depth_final_norm.shape[-2:]:
                posterior_for_reg = _resize_and_normalize_posterior(focus_posterior, depth_final_norm.shape[-2:])
            else:
                posterior_for_reg = focus_posterior

            valid_mask_for_reg = depth_mask
            if valid_mask_for_reg is not None and valid_mask_for_reg.shape[-2:] != depth_final_norm.shape[-2:]:
                if valid_mask_for_reg.dim() == 3:
                    valid_mask_for_reg = valid_mask_for_reg.unsqueeze(1)
                valid_mask_for_reg = F.interpolate(
                    valid_mask_for_reg.float(),
                    size=depth_final_norm.shape[-2:],
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
                depth_final_norm,
                affinity,
                valid_mask=valid_mask_for_reg,
            )

            if gate_focus is not None:
                if gate_focus.shape[-2:] != depth_final_norm.shape[-2:]:
                    gate_for_reg = F.interpolate(
                        gate_focus,
                        size=depth_final_norm.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    gate_for_reg = gate_focus
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
        total = total + self.focus_posterior_kl_weight * losses.get("loss_focus_posterior_kl", torch.zeros_like(total))
        total = total + self.focus_depth_weight * losses.get("loss_depth_focus", torch.zeros_like(total))
        total = total + self.prior_depth_weight * losses.get("loss_depth_prior", torch.zeros_like(total))
        total = total + self.rgb_weight * losses.get("loss_rgb_reconstruction", torch.zeros_like(total))
        total = total + self.aif_focus_evidence_weight * losses.get("loss_aif_focus_consistency", torch.zeros_like(total))
        total = total + self.uncertainty_focus_weight * losses.get("loss_uncertainty_focus", torch.zeros_like(total))
        total = total + self.uncertainty_error_weight * losses.get("loss_uncertainty_error", torch.zeros_like(total))
        total = total + self.gate_calibration_weight * losses.get("loss_gate_focus", torch.zeros_like(total))
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
