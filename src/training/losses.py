"""Loss functions for FocalDiffusion with Focal Evidence Posterior supervision."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_focus_coordinates(focus_distances: torch.Tensor) -> torch.Tensor:
    mn = focus_distances.min(dim=1, keepdim=True).values
    mx = focus_distances.max(dim=1, keepdim=True).values
    return (focus_distances - mn) / (mx - mn + 1e-6)


def build_focus_target_from_depth(
    depth_norm: torch.Tensor,
    focus_distances: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a soft focus-plane target from normalized depth.

    The target is a focus-measure-like curve over focal planes: pixels whose
    normalized depth is close to a focus distance receive high probability at
    that focus plane and lower probability elsewhere.
    """
    tau = normalize_focus_coordinates(focus_distances)
    depth_hw = depth_norm.squeeze(1) if depth_norm.dim() == 4 else depth_norm
    logits = -torch.abs(tau[:, :, None, None] - depth_hw[:, None]) / max(temperature, 1e-6)
    return torch.softmax(logits, dim=1), tau


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(pred, target)
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.shape != pred.shape:
        mask = mask.expand_as(pred)
    return (torch.abs(pred - target) * mask).sum() / mask.sum().clamp(min=1.0)


class FocalDiffusionLoss(nn.Module):
    """Training objective for FSDiffusion.

    Main supervision:
    - SD3 flow-matching loss for AIF latent generation.
    - final depth supervision for the physics-gated output.
    - Focal Evidence Posterior supervision via a depth-derived soft focus target.
    - focus-depth and AIF-focus consistency losses.
    - uncertainty calibration from focus entropy / prediction error.
    """

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        depth_weight: float = 1.0,
        rgb_weight: float = 0.5,
        focus_posterior_kl_weight: float = 0.2,
        focus_depth_weight: float = 0.2,
        prior_depth_weight: float = 0.05,
        aif_focus_evidence_weight: float = 0.1,
        uncertainty_focus_weight: float = 0.05,
        uncertainty_error_weight: float = 0.05,
        focus_target_temperature: float = 0.07,
        supervision_mode: str = "supervised",
    ):
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.depth_weight = depth_weight
        self.rgb_weight = rgb_weight
        self.focus_posterior_kl_weight = focus_posterior_kl_weight
        self.focus_depth_weight = focus_depth_weight
        self.prior_depth_weight = prior_depth_weight
        self.aif_focus_evidence_weight = aif_focus_evidence_weight
        self.uncertainty_focus_weight = uncertainty_focus_weight
        self.uncertainty_error_weight = uncertainty_error_weight
        self.focus_target_temperature = focus_target_temperature
        self.supervision_mode = supervision_mode

    def forward(
        self,
        diffusion_pred: torch.Tensor,
        diffusion_target: torch.Tensor,
        depth_target: torch.Tensor | None = None,
        rgb_pred: torch.Tensor | None = None,
        rgb_target: torch.Tensor | None = None,
        shape_norm: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
        focal_stack: torch.Tensor | None = None,
        depth_prior: torch.Tensor | None = None,
        depth_focus: torch.Tensor | None = None,
        depth_final: torch.Tensor | None = None,
        focus_prob: torch.Tensor | None = None,
        focus_entropy: torch.Tensor | None = None,
        focus_distances: torch.Tensor | None = None,
        depth_mask: torch.Tensor | None = None,
        depth_range: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        supervised_losses: dict[str, torch.Tensor] = {}
        generator_losses: dict[str, torch.Tensor] = {
            "loss_fm": F.mse_loss(diffusion_pred, diffusion_target)
        }

        enable_supervised = self.supervision_mode in {"supervised", "semi_supervised"}
        depth_prediction = depth_final if depth_final is not None else shape_norm

        depth_gt_norm = None
        mask = None
        if enable_supervised and depth_target is not None and depth_prediction is not None and depth_range is not None:
            depth_pred_resized = F.interpolate(
                depth_prediction,
                size=depth_target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            depth_min = depth_range[:, 0].view(-1, 1, 1, 1)
            depth_max = depth_range[:, 1].view(-1, 1, 1, 1)
            depth_gt_norm = ((depth_target - depth_min) / (depth_max - depth_min).clamp(min=1e-6)).clamp(0.0, 1.0)
            mask = depth_mask.unsqueeze(1) if depth_mask is not None and depth_mask.dim() == depth_pred_resized.dim() - 1 else depth_mask

            supervised_losses["loss_shape_supervised"] = masked_l1(depth_pred_resized, depth_gt_norm, mask)

            if depth_focus is not None:
                focus_resized = F.interpolate(depth_focus, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                supervised_losses["loss_focus_depth"] = masked_l1(focus_resized, depth_gt_norm, mask)

            if depth_prior is not None:
                prior_resized = F.interpolate(depth_prior, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                supervised_losses["loss_prior_depth"] = masked_l1(prior_resized, depth_gt_norm, mask)

            if focus_prob is not None and focus_distances is not None:
                focus_prob_resized = focus_prob
                if focus_prob_resized.shape[-2:] != depth_target.shape[-2:]:
                    focus_prob_resized = F.interpolate(
                        focus_prob_resized,
                        size=depth_target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    focus_prob_resized = focus_prob_resized / focus_prob_resized.sum(dim=1, keepdim=True).clamp(min=1e-6)

                focus_target, _ = build_focus_target_from_depth(
                    depth_gt_norm,
                    focus_distances,
                    temperature=self.focus_target_temperature,
                )
                kl = focus_target * (torch.log(focus_target + 1e-6) - torch.log(focus_prob_resized + 1e-6))
                kl = kl.sum(dim=1, keepdim=True)
                if mask is not None:
                    supervised_losses["loss_focus_posterior_kl"] = (kl * mask).sum() / mask.sum().clamp(min=1)
                else:
                    supervised_losses["loss_focus_posterior_kl"] = kl.mean()

            if uncertainty is not None and depth_final is not None:
                u = F.interpolate(uncertainty, size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                final_resized = F.interpolate(depth_final.detach(), size=depth_target.shape[-2:], mode="bilinear", align_corners=False)
                error_norm = torch.abs(final_resized - depth_gt_norm)
                generator_losses["loss_uncertainty_error"] = masked_l1(u, error_norm, mask)

        if enable_supervised and rgb_pred is not None and rgb_target is not None:
            supervised_losses["rgb"] = F.l1_loss(rgb_pred, rgb_target)

        if focus_prob is not None and focal_stack is not None and rgb_pred is not None:
            focus_for_aif = focus_prob
            if focus_for_aif.shape[-2:] != focal_stack.shape[-2:]:
                focus_for_aif = F.interpolate(focus_for_aif, size=focal_stack.shape[-2:], mode="bilinear", align_corners=False)
                focus_for_aif = focus_for_aif / focus_for_aif.sum(dim=1, keepdim=True).clamp(min=1e-6)
            aif_focus = (focus_for_aif[:, :, None] * focal_stack).sum(dim=1).detach()
            rp = rgb_pred
            if rp.shape[-2:] != aif_focus.shape[-2:]:
                rp = F.interpolate(rp, size=aif_focus.shape[-2:], mode="bilinear", align_corners=False)
            hp_pred = rp - F.avg_pool2d(rp, 5, 1, 2)
            hp_focus = aif_focus - F.avg_pool2d(aif_focus, 5, 1, 2)
            generator_losses["loss_aif_focus_evidence"] = F.l1_loss(hp_pred, hp_focus)

        if uncertainty is not None and focus_entropy is not None:
            u = uncertainty
            fe = focus_entropy.detach()
            if u.shape[-2:] != fe.shape[-2:]:
                u = F.interpolate(u, size=fe.shape[-2:], mode="bilinear", align_corners=False)
            generator_losses["loss_uncertainty_focus"] = F.l1_loss(u, fe)

        total = torch.zeros_like(generator_losses["loss_fm"])
        total = total + self.diffusion_weight * generator_losses["loss_fm"]
        total = total + self.depth_weight * supervised_losses.get("loss_shape_supervised", torch.zeros_like(total))
        total = total + self.rgb_weight * supervised_losses.get("rgb", torch.zeros_like(total))
        total = total + self.focus_posterior_kl_weight * supervised_losses.get("loss_focus_posterior_kl", torch.zeros_like(total))
        total = total + self.focus_depth_weight * supervised_losses.get("loss_focus_depth", torch.zeros_like(total))
        total = total + self.prior_depth_weight * supervised_losses.get("loss_prior_depth", torch.zeros_like(total))
        total = total + self.aif_focus_evidence_weight * generator_losses.get("loss_aif_focus_evidence", torch.zeros_like(total))
        total = total + self.uncertainty_focus_weight * generator_losses.get("loss_uncertainty_focus", torch.zeros_like(total))
        total = total + self.uncertainty_error_weight * generator_losses.get("loss_uncertainty_error", torch.zeros_like(total))

        return {
            **generator_losses,
            **supervised_losses,
            "total": total,
            "generator_losses": generator_losses,
            "supervised_losses": supervised_losses,
        }
