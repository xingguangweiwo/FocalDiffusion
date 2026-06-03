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
        gate_calibration_weight: float = 0.05,
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
        self.focus_target_temperature = focus_target_temperature
        self.gate_calibration_weight = gate_calibration_weight

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
                focus_target, _ = build_soft_focus_target_from_depth(
                    depth_gt_norm,
                    focus_distances,
                    temperature=self.focus_target_temperature,
                )
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

        return {**losses, "total": total}
