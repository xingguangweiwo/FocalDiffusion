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
    def __init__(self, diffusion_weight: float = 1.0, depth_weight: float = 0.0, rgb_weight: float = 0.0, focus_energy_weight: float = 0.5, tau_contrast_weight: float = 0.2, stack_contrast_weight: float = 0.2, mismatch_contrast_weight: float = 0.2, shape_candidate_contrast_weight: float = 0.2, uncertainty_weight: float = 0.05, aif_highpass_weight: float = 0.1, supervision_mode: str = "supervised", **kwargs):
        super().__init__()
        self.diffusion_weight=diffusion_weight; self.depth_weight=depth_weight; self.rgb_weight=rgb_weight
        self.focus_energy_weight=focus_energy_weight; self.tau_contrast_weight=tau_contrast_weight
        self.stack_contrast_weight=stack_contrast_weight; self.mismatch_contrast_weight=mismatch_contrast_weight; self.shape_candidate_contrast_weight=shape_candidate_contrast_weight
        self.uncertainty_weight=uncertainty_weight; self.aif_highpass_weight=aif_highpass_weight
        self.supervision_mode = supervision_mode

    def forward(self, diffusion_pred, diffusion_target, depth_pred=None, depth_target=None, rgb_pred=None, rgb_target=None, shape_norm=None, uncertainty=None, focal_stack=None, critic_outputs=None, critic_generator_outputs=None, depth_mask=None, focal_features=None, confidence_map=None, depth_range=None, **kwargs):
        supervised_losses = {}
        critic_losses = {}
        generator_losses = {'loss_fm': F.mse_loss(diffusion_pred, diffusion_target)}

        if self.supervision_mode == "self_supervised":
            enable_supervised = False
        elif self.supervision_mode == "semi_supervised":
            enable_supervised = True
        else:
            enable_supervised = True

        if enable_supervised and depth_target is not None and shape_norm is not None and depth_range is not None:
            shape_norm_resized = F.interpolate(shape_norm, size=depth_target.shape[-2:], mode='bilinear', align_corners=False)
            depth_min = depth_range[:, 0].view(-1, 1, 1, 1)
            depth_max = depth_range[:, 1].view(-1, 1, 1, 1)
            depth_gt_norm = ((depth_target - depth_min) / (depth_max - depth_min).clamp(min=1e-6)).clamp(0.0, 1.0)
            if depth_mask is not None:
                mask = depth_mask.unsqueeze(1) if depth_mask.dim() == shape_norm_resized.dim() - 1 else depth_mask
                supervised_losses['loss_shape_supervised'] = (torch.abs(shape_norm_resized - depth_gt_norm) * mask).sum() / mask.sum().clamp(min=1)
            else:
                supervised_losses['loss_shape_supervised'] = F.l1_loss(shape_norm_resized, depth_gt_norm)
        if enable_supervised and rgb_pred is not None and rgb_target is not None:
            supervised_losses['rgb'] = F.l1_loss(rgb_pred, rgb_target)
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

        if self.supervision_mode == "self_supervised":
            supervised_losses = {}

        total=torch.zeros_like(generator_losses['loss_fm'])
        total = total + self.diffusion_weight * generator_losses['loss_fm']
        total = total + self.focus_energy_weight * generator_losses.get('loss_focus_generator', torch.zeros_like(total))
        total = total + self.uncertainty_weight * generator_losses.get('loss_uncertainty', torch.zeros_like(total))
        total = total + self.aif_highpass_weight * generator_losses.get('loss_aif_highpass', torch.zeros_like(total))
        total = total + self.tau_contrast_weight * critic_losses.get('loss_critic_tau', torch.zeros_like(total))
        total = total + self.stack_contrast_weight * critic_losses.get('loss_critic_stack', torch.zeros_like(total))
        total = total + self.mismatch_contrast_weight * critic_losses.get('loss_critic_mismatch', torch.zeros_like(total))
        total = total + self.shape_candidate_contrast_weight * critic_losses.get('loss_critic_shape_candidate', torch.zeros_like(total))
        total = total + self.depth_weight * supervised_losses.get('loss_shape_supervised', torch.zeros_like(total))
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


class DepthLoss(nn.Module):
    def forward(self, pred, target, mask=None):
        return F.l1_loss(pred, target)
