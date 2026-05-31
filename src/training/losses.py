"""Loss functions and focus-consistency critic for FocalDiffusion."""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_focus_coordinates(focus_distances: torch.Tensor) -> torch.Tensor:
    mn = focus_distances.min(dim=1, keepdim=True).values
    mx = focus_distances.max(dim=1, keepdim=True).values
    return (focus_distances - mn) / (mx - mn + 1e-6)


def build_focus_target_from_depth(depth_norm, focus_distances, temperature=0.07):
    tau = normalize_focus_coordinates(focus_distances)
    if depth_norm.dim() == 4:
        depth_hw = depth_norm.squeeze(1)
    else:
        depth_hw = depth_norm
    logits = -torch.abs(tau[:, :, None, None] - depth_hw[:, None]) / max(temperature, 1e-6)
    return torch.softmax(logits, dim=1), tau


def build_aif_physical(focal_stack: torch.Tensor, shape_norm: torch.Tensor, tau: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = -torch.abs(tau.unsqueeze(-1).unsqueeze(-1) - shape_norm.unsqueeze(1)) / max(temperature, 1e-6)
    w = torch.softmax(logits, dim=1).unsqueeze(2)
    return (w * focal_stack).sum(dim=1)


class FocusConsistencyCritic(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 32, temperature: float = 0.07, margin: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.sharpness_estimator = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def estimate_sharpness(self, focal_stack: torch.Tensor) -> torch.Tensor:
        b,n,c,h,w = focal_stack.shape
        return self.sharpness_estimator(focal_stack.reshape(b*n,c,h,w)).reshape(b,n,h,w)

    def focus_energy(self, sharpness: torch.Tensor, tau: torch.Tensor, shape_hw: torch.Tensor):
        target = torch.softmax(-torch.abs(tau.unsqueeze(-1).unsqueeze(-1) - shape_hw.unsqueeze(1)) / self.temperature, dim=1)
        observed = torch.softmax(sharpness / self.temperature, dim=1)
        kl = (target * (torch.log(target + 1e-8) - torch.log(observed + 1e-8))).sum(dim=1)
        entropy = -(observed * torch.log(observed + 1e-8)).sum(dim=1)
        n_focus = max(observed.shape[1], 2)
        reliability = 1.0 - (entropy / torch.log(torch.tensor(float(n_focus), device=entropy.device, dtype=entropy.dtype)))
        reliability = reliability.clamp(0.0, 1.0)
        return (kl * reliability).mean(), reliability

    def forward(self, focal_stack: torch.Tensor, focus_distances: torch.Tensor, shape_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        tau = normalize_focus_coordinates(focus_distances)
        shape_hw = shape_norm.squeeze(1)
        if shape_hw.shape[-2:] != focal_stack.shape[-2:]:
            shape_hw = F.interpolate(shape_hw.unsqueeze(1), size=focal_stack.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        sharpness = self.estimate_sharpness(focal_stack)
        pos_energy, reliability = self.focus_energy(sharpness, tau, shape_hw)

        b, n = tau.shape
        permb = torch.stack([torch.randperm(n, device=tau.device) for _ in range(b)], dim=0)
        tau_shuf = tau.gather(1, permb)
        neg_tau, _ = self.focus_energy(sharpness, tau_shuf, shape_hw)

        stack_shuf = focal_stack.gather(1, permb[:,:,None,None,None].expand_as(focal_stack))
        sharp_shuf = self.estimate_sharpness(stack_shuf)
        neg_stack, _ = self.focus_energy(sharp_shuf, tau, shape_hw)

        mismatch = self.focus_energy(sharp_shuf, tau_shuf, shape_hw)[0]

        shifted = (shape_hw + 0.15).clamp(0,1)
        flipped = 1.0 - shape_hw
        smooth = F.avg_pool2d(shape_hw.unsqueeze(1),5,1,2).squeeze(1)
        rand = torch.rand_like(shape_hw)
        wrong = torch.stack([
            self.focus_energy(sharpness, tau, shifted)[0],
            self.focus_energy(sharpness, tau, flipped)[0],
            self.focus_energy(sharpness, tau, smooth)[0],
            self.focus_energy(sharpness, tau, rand)[0],
        ]).mean()

        return {
            'tau': tau,
            'sharpness_score': sharpness,
            'focus_energy': pos_energy,
            'reliability_map': reliability,
            'tau_contrast': F.relu(self.margin + pos_energy - neg_tau),
            'stack_contrast': F.relu(self.margin + pos_energy - neg_stack),
            'mismatch_contrast': F.relu(self.margin + pos_energy - mismatch),
            'shape_candidate_contrast': F.relu(self.margin + pos_energy - wrong),
        }


class FocalDiffusionLoss(nn.Module):
    def __init__(self, diffusion_weight: float = 1.0, depth_weight: float = 0.0, rgb_weight: float = 0.0, focus_energy_weight: float = 0.0, tau_contrast_weight: float = 0.0, stack_contrast_weight: float = 0.0, mismatch_contrast_weight: float = 0.0, shape_candidate_contrast_weight: float = 0.0, uncertainty_weight: float = 0.0, aif_highpass_weight: float = 0.0, supervision_mode: str = "supervised", focus_posterior_kl_weight: float = 0.2, focus_depth_weight: float = 0.2, prior_depth_weight: float = 0.05, aif_focus_evidence_weight: float = 0.1, uncertainty_focus_weight: float = 0.05, focus_target_temperature: float = 0.07, **kwargs):
        super().__init__()
        self.diffusion_weight=diffusion_weight; self.depth_weight=depth_weight; self.rgb_weight=rgb_weight
        self.focus_energy_weight=focus_energy_weight; self.tau_contrast_weight=tau_contrast_weight
        self.stack_contrast_weight=stack_contrast_weight; self.mismatch_contrast_weight=mismatch_contrast_weight; self.shape_candidate_contrast_weight=shape_candidate_contrast_weight
        self.uncertainty_weight=uncertainty_weight; self.aif_highpass_weight=aif_highpass_weight
        self.focus_posterior_kl_weight=focus_posterior_kl_weight; self.focus_depth_weight=focus_depth_weight
        self.prior_depth_weight=prior_depth_weight; self.aif_focus_evidence_weight=aif_focus_evidence_weight
        self.uncertainty_focus_weight=uncertainty_focus_weight; self.focus_target_temperature=focus_target_temperature
        self.supervision_mode = supervision_mode

    def forward(
        self,
        diffusion_pred,
        diffusion_target,
        depth_target=None,
        rgb_pred=None,
        rgb_target=None,
        shape_norm=None,
        uncertainty=None,
        focal_stack=None,
        depth_prior=None,
        depth_focus=None,
        depth_final=None,
        focus_prob=None,
        focus_entropy=None,
        focus_reliability=None,
        focus_distances=None,
        critic_outputs=None,
        critic_generator_outputs=None,
        depth_mask=None,
        depth_range=None,
    ):
        supervised_losses = {}
        critic_losses = {}
        generator_losses = {'loss_fm': F.mse_loss(diffusion_pred, diffusion_target)}

        enable_supervised = self.supervision_mode in {"supervised", "semi_supervised"}

        depth_gt_norm = None
        mask = None
        depth_prediction = depth_final if depth_final is not None else shape_norm
        if enable_supervised and depth_target is not None and depth_prediction is not None and depth_range is not None:
            depth_pred_resized = F.interpolate(depth_prediction, size=depth_target.shape[-2:], mode='bilinear', align_corners=False)
            depth_min = depth_range[:, 0].view(-1, 1, 1, 1)
            depth_max = depth_range[:, 1].view(-1, 1, 1, 1)
            depth_gt_norm = ((depth_target - depth_min) / (depth_max - depth_min).clamp(min=1e-6)).clamp(0.0, 1.0)
            if depth_mask is not None:
                mask = depth_mask.unsqueeze(1) if depth_mask.dim() == depth_pred_resized.dim() - 1 else depth_mask
                supervised_losses['loss_shape_supervised'] = (torch.abs(depth_pred_resized - depth_gt_norm) * mask).sum() / mask.sum().clamp(min=1)
            else:
                supervised_losses['loss_shape_supervised'] = F.l1_loss(depth_pred_resized, depth_gt_norm)

            if depth_focus is not None:
                focus_resized = F.interpolate(depth_focus, size=depth_target.shape[-2:], mode='bilinear', align_corners=False)
                if mask is not None:
                    supervised_losses['loss_focus_depth'] = (torch.abs(focus_resized - depth_gt_norm) * mask).sum() / mask.sum().clamp(min=1)
                else:
                    supervised_losses['loss_focus_depth'] = F.l1_loss(focus_resized, depth_gt_norm)
            if depth_prior is not None:
                prior_resized = F.interpolate(depth_prior, size=depth_target.shape[-2:], mode='bilinear', align_corners=False)
                if mask is not None:
                    supervised_losses['loss_prior_depth'] = (torch.abs(prior_resized - depth_gt_norm) * mask).sum() / mask.sum().clamp(min=1)
                else:
                    supervised_losses['loss_prior_depth'] = F.l1_loss(prior_resized, depth_gt_norm)
            if focus_prob is not None and focus_distances is not None:
                focus_prob_resized = focus_prob
                if focus_prob_resized.shape[-2:] != depth_target.shape[-2:]:
                    focus_prob_resized = F.interpolate(focus_prob_resized, size=depth_target.shape[-2:], mode='bilinear', align_corners=False)
                    focus_prob_resized = focus_prob_resized / focus_prob_resized.sum(dim=1, keepdim=True).clamp(min=1e-6)
                focus_target, _ = build_focus_target_from_depth(depth_gt_norm, focus_distances, temperature=self.focus_target_temperature)
                kl = focus_target * (torch.log(focus_target + 1e-6) - torch.log(focus_prob_resized + 1e-6))
                kl = kl.sum(dim=1, keepdim=True)
                supervised_losses['loss_focus_posterior_kl'] = (kl * mask).sum() / mask.sum().clamp(min=1) if mask is not None else kl.mean()
            if uncertainty is not None and depth_final is not None:
                u = F.interpolate(uncertainty, size=depth_target.shape[-2:], mode='bilinear', align_corners=False)
                final_resized = F.interpolate(depth_final.detach(), size=depth_target.shape[-2:], mode='bilinear', align_corners=False)
                error_norm = torch.abs(final_resized - depth_gt_norm)
                generator_losses['loss_uncertainty_error'] = F.l1_loss(u, error_norm) if mask is None else (torch.abs(u - error_norm) * mask).sum() / mask.sum().clamp(min=1)
        if enable_supervised and rgb_pred is not None and rgb_target is not None:
            supervised_losses['rgb'] = F.l1_loss(rgb_pred, rgb_target)
        if focus_prob is not None and focal_stack is not None and rgb_pred is not None:
            focus_for_aif = focus_prob
            if focus_for_aif.shape[-2:] != focal_stack.shape[-2:]:
                focus_for_aif = F.interpolate(focus_for_aif, size=focal_stack.shape[-2:], mode='bilinear', align_corners=False)
                focus_for_aif = focus_for_aif / focus_for_aif.sum(dim=1, keepdim=True).clamp(min=1e-6)
            aif_focus = (focus_for_aif[:, :, None] * focal_stack).sum(dim=1)
            rp = rgb_pred
            if rp.shape[-2:] != aif_focus.shape[-2:]:
                rp = F.interpolate(rp, size=aif_focus.shape[-2:], mode='bilinear', align_corners=False)
            hp_pred = rp - F.avg_pool2d(rp, 5, 1, 2)
            aif_focus_detached = aif_focus.detach()
            hp_focus = aif_focus_detached - F.avg_pool2d(aif_focus_detached, 5, 1, 2)
            generator_losses['loss_aif_focus_evidence'] = F.l1_loss(hp_pred, hp_focus)
        if uncertainty is not None and focus_entropy is not None:
            u = uncertainty
            fe = focus_entropy.detach()
            if u.shape[-2:] != fe.shape[-2:]:
                u = F.interpolate(u, size=fe.shape[-2:], mode='bilinear', align_corners=False)
            generator_losses['loss_uncertainty_focus'] = F.l1_loss(u, fe)

        if critic_outputs is not None:
            critic_losses['loss_critic_tau'] = critic_outputs['tau_contrast']
            critic_losses['loss_critic_stack'] = critic_outputs['stack_contrast']
            critic_losses['loss_critic_mismatch'] = critic_outputs['mismatch_contrast']
            critic_losses['loss_critic_shape_candidate'] = critic_outputs['shape_candidate_contrast']
        if critic_generator_outputs is not None:
            generator_losses['loss_focus_generator'] = critic_generator_outputs['focus_energy']
            if uncertainty is not None:
                u=uncertainty.squeeze(1)
                rel=critic_generator_outputs['reliability_map']
                if u.shape[-2:] != rel.shape[-2:]:
                    u=F.interpolate(u.unsqueeze(1), size=rel.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
                generator_losses['loss_uncertainty']=F.l1_loss(u, 1.0-rel.detach())
            if rgb_pred is not None and shape_norm is not None and focal_stack is not None:
                s=shape_norm.squeeze(1)
                if s.shape[-2:] != focal_stack.shape[-2:]:
                    s=F.interpolate(s.unsqueeze(1), size=focal_stack.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
                aif_phys=build_aif_physical(focal_stack,s,critic_generator_outputs['tau'])
                rp=rgb_pred
                if rp.shape[-2:] != aif_phys.shape[-2:]:
                    rp=F.interpolate(rp,size=aif_phys.shape[-2:],mode='bilinear',align_corners=False)
                hp_pred=rp-F.avg_pool2d(rp,5,1,2); hp_phys=aif_phys-F.avg_pool2d(aif_phys,5,1,2)
                generator_losses['loss_aif_highpass']=F.l1_loss(hp_pred,hp_phys)


        total=torch.zeros_like(generator_losses['loss_fm'])
        total = total + self.diffusion_weight * generator_losses['loss_fm']
        total = total + self.focus_energy_weight * generator_losses.get('loss_focus_generator', torch.zeros_like(total))
        total = total + self.uncertainty_weight * generator_losses.get('loss_uncertainty', torch.zeros_like(total))
        total = total + self.uncertainty_weight * generator_losses.get('loss_uncertainty_error', torch.zeros_like(total))
        total = total + self.uncertainty_focus_weight * generator_losses.get('loss_uncertainty_focus', torch.zeros_like(total))
        total = total + self.aif_highpass_weight * generator_losses.get('loss_aif_highpass', torch.zeros_like(total))
        total = total + self.aif_focus_evidence_weight * generator_losses.get('loss_aif_focus_evidence', torch.zeros_like(total))
        total = total + self.tau_contrast_weight * critic_losses.get('loss_critic_tau', torch.zeros_like(total))
        total = total + self.stack_contrast_weight * critic_losses.get('loss_critic_stack', torch.zeros_like(total))
        total = total + self.mismatch_contrast_weight * critic_losses.get('loss_critic_mismatch', torch.zeros_like(total))
        total = total + self.shape_candidate_contrast_weight * critic_losses.get('loss_critic_shape_candidate', torch.zeros_like(total))
        total = total + self.depth_weight * supervised_losses.get('loss_shape_supervised', torch.zeros_like(total))
        total = total + self.focus_posterior_kl_weight * supervised_losses.get('loss_focus_posterior_kl', torch.zeros_like(total))
        total = total + self.focus_depth_weight * supervised_losses.get('loss_focus_depth', torch.zeros_like(total))
        total = total + self.prior_depth_weight * supervised_losses.get('loss_prior_depth', torch.zeros_like(total))
        total = total + self.rgb_weight * supervised_losses.get('rgb', torch.zeros_like(total))

        return {
            **generator_losses,
            **critic_losses,
            **supervised_losses,
            'total': total,
            'generator_losses': generator_losses,
            'critic_losses': critic_losses,
            'supervised_losses': supervised_losses,
        }
