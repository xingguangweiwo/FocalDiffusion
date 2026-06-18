"""Conservative physical verification modules for focal-stack evidence."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .verification_trace import PhysicalVerificationTrace


class FocusMeasureBank(nn.Module):
    """Compute interpretable focus measures with fixed derivative operators."""

    def __init__(self, eps: float = 1e-6) -> None:
        """Initialize the fixed Sobel/Laplacian focus-measure bank."""
        super().__init__()
        self.eps = eps
        self.register_buffer("sobel_x", torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0)
        self.register_buffer("sobel_y", torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0)
        self.register_buffer("laplacian", torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3))

    @staticmethod
    def _validate_stack(focal_stack: torch.Tensor) -> None:
        """Validate that ``focal_stack`` follows ``[B, K, C, H, W]``."""
        if focal_stack.dim() != 5:
            raise ValueError(f"focal_stack must have shape [B, K, C, H, W], got {tuple(focal_stack.shape)}")
        if focal_stack.shape[1] < 1:
            raise ValueError("focal_stack must contain at least one focal plane.")
        if not torch.isfinite(focal_stack).all():
            raise ValueError("focal_stack must contain only finite values.")

    def _filter_gray(self, gray: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply a fixed 2-D kernel to flattened grayscale focal planes."""
        batch, planes, _, height, width = gray.shape
        flat = gray.reshape(batch * planes, 1, height, width)
        return F.conv2d(flat, kernel.to(device=gray.device, dtype=gray.dtype), padding=1).reshape(batch, planes, 1, height, width)

    def forward(self, focal_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return focus posterior and operator scores for a focal stack."""
        self._validate_stack(focal_stack)
        gray = focal_stack.mean(dim=2, keepdim=True)
        gx = self._filter_gray(gray, self.sobel_x)
        gy = self._filter_gray(gray, self.sobel_y)
        lap = self._filter_gray(gray, self.laplacian)
        tenengrad = gx.square() + gy.square()
        lap_energy = lap.abs()
        local_var = (gray - F.avg_pool3d(gray, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))).square()
        measures = torch.cat([tenengrad, lap_energy, local_var], dim=2)
        plane_scores = measures.mean(dim=2)
        normalized_scores = plane_scores / plane_scores.amax(dim=1, keepdim=True).clamp(min=self.eps)
        posterior = torch.softmax(normalized_scores / 0.1, dim=1)
        topk = torch.topk(posterior, k=min(2, posterior.shape[1]), dim=1).values
        peak = topk[:, 0:1]
        margin = peak if posterior.shape[1] == 1 else topk[:, 0:1] - topk[:, 1:2]
        entropy = -(posterior * torch.log(posterior + self.eps)).sum(dim=1, keepdim=True) / math.log(max(posterior.shape[1], 2))
        winners = measures.argmax(dim=1)
        agreement = (winners == winners[:, 0:1]).float().mean(dim=1, keepdim=True)
        texture = plane_scores.amax(dim=1, keepdim=True)
        texture = (texture / (texture.mean(dim=(-2, -1), keepdim=True) + texture.std(dim=(-2, -1), keepdim=True) + self.eps)).clamp(0.0, 1.0)
        return {
            "focus_scores": normalized_scores,
            "focus_posterior": posterior,
            "focus_peak": peak,
            "focus_margin": margin.clamp(0.0, 1.0),
            "focus_entropy": entropy.clamp(0.0, 1.0),
            "operator_agreement": agreement.clamp(0.0, 1.0),
            "texture_confidence": texture,
        }


class DefocusConsistencyVerifier(nn.Module):
    """Check whether depth, AIF image, and focal stack agree under simple blur."""

    def __init__(self, max_blur_radius: int = 5, eps: float = 1e-6) -> None:
        """Initialize a non-learned defocus consistency verifier."""
        super().__init__()
        self.max_blur_radius = max(1, int(max_blur_radius))
        self.eps = eps

    @staticmethod
    def _normalize_focal_distances(focal_plane_distances: torch.Tensor, batch: int, planes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Normalize focal distances to ``[B, K]`` canonical coordinates."""
        distances = focal_plane_distances.to(device=device, dtype=dtype)
        if distances.dim() == 1:
            distances = distances.unsqueeze(0)
        if distances.shape[0] == 1 and batch != 1:
            distances = distances.expand(batch, -1)
        if distances.shape != (batch, planes):
            raise ValueError(f"focal_plane_distances must have shape [B, K] or [K], got {tuple(focal_plane_distances.shape)}")
        mn = distances.min(dim=1, keepdim=True).values
        mx = distances.max(dim=1, keepdim=True).values
        return (distances - mn) / (mx - mn).clamp(min=1e-6)

    def _multi_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Approximate defocus by averaging a small bank of box blurs."""
        blurred = [image]
        for radius in (1, 3, self.max_blur_radius):
            kernel = 2 * radius + 1
            blurred.append(F.avg_pool2d(image, kernel_size=kernel, stride=1, padding=radius))
        return torch.stack(blurred, dim=1).mean(dim=1)

    def forward(self, focal_stack: torch.Tensor, focal_plane_distances: torch.Tensor, depth_canonical: torch.Tensor, all_in_focus: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return residual maps from defocus and refocus consistency checks."""
        FocusMeasureBank._validate_stack(focal_stack)
        batch, planes, _, height, width = focal_stack.shape
        if depth_canonical.dim() == 3:
            depth_canonical = depth_canonical.unsqueeze(1)
        depth = F.interpolate(depth_canonical, size=(height, width), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        aif = F.interpolate(all_in_focus, size=(height, width), mode="bilinear", align_corners=False)
        coords = self._normalize_focal_distances(focal_plane_distances, batch, planes, focal_stack.device, focal_stack.dtype)
        blur_amount = (depth[:, None] - coords[:, :, None, None, None]).abs().clamp(0.0, 1.0)
        blurred = self._multi_blur(aif)[:, None]
        rendered = (1.0 - blur_amount) * aif[:, None] + blur_amount * blurred
        defocus_residual = (rendered - focal_stack).abs().mean(dim=2).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        refocused = focal_stack.mean(dim=1)
        refocus_residual = (refocused - aif).abs().mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        return {"defocus_residual": defocus_residual, "refocus_residual": refocus_residual, "rendered_stack": rendered}


class FocalPhysicalVerifier(nn.Module):
    """Fuse fixed focus and defocus checks into a PhysicalVerificationTrace."""

    def __init__(self, eps: float = 1e-6) -> None:
        """Initialize the conservative FocalTrace physical verifier."""
        super().__init__()
        self.eps = eps
        self.focus_bank = FocusMeasureBank(eps=eps)
        self.defocus_verifier = DefocusConsistencyVerifier(eps=eps)

    def forward(self, focal_stack: torch.Tensor, focal_plane_distances: torch.Tensor, depth_canonical: torch.Tensor, all_in_focus: torch.Tensor, generated_depth_canonical: torch.Tensor | None = None) -> PhysicalVerificationTrace:
        """Compute a batch-first physical verification trace for FocalTrace."""
        focus = self.focus_bank(focal_stack)
        batch, planes = focal_stack.shape[:2]
        coords = DefocusConsistencyVerifier._normalize_focal_distances(focal_plane_distances, batch, planes, focal_stack.device, focal_stack.dtype)
        focus_depth = (focus["focus_posterior"] * coords[:, :, None, None]).sum(dim=1, keepdim=True)
        depth = F.interpolate(depth_canonical if depth_canonical.dim() == 4 else depth_canonical.unsqueeze(1), size=focus_depth.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        prior = depth if generated_depth_canonical is None else F.interpolate(generated_depth_canonical if generated_depth_canonical.dim() == 4 else generated_depth_canonical.unsqueeze(1), size=focus_depth.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        residuals = self.defocus_verifier(focal_stack, focal_plane_distances, depth, all_in_focus)
        discrepancy = (depth - focus_depth).abs().clamp(0.0, 1.0)
        generation_discrepancy = (prior - focus_depth).abs().clamp(0.0, 1.0)
        focus_support = (focus["focus_peak"] * focus["focus_margin"] * (1.0 - focus["focus_entropy"]) * focus["operator_agreement"] * focus["texture_confidence"]).clamp(0.0, 1.0)
        physical_penalty = torch.maximum(discrepancy, residuals["defocus_residual"])
        generation_support = ((1.0 - generation_discrepancy) * (1.0 - residuals["refocus_residual"])).clamp(0.0, 1.0)
        conflict_score = torch.maximum(discrepancy, generation_discrepancy).clamp(0.0, 1.0)
        invalid_score = torch.maximum(focus["focus_entropy"], torch.maximum(residuals["defocus_residual"], residuals["refocus_residual"])).clamp(0.0, 1.0)
        support_logit = focus_support + generation_support - physical_penalty
        verdict_logits = torch.cat([support_logit, conflict_score, invalid_score], dim=1)
        return PhysicalVerificationTrace(
            focus_peak=focus["focus_peak"],
            focus_margin=focus["focus_margin"],
            focus_entropy=focus["focus_entropy"],
            operator_agreement=focus["operator_agreement"],
            texture_confidence=focus["texture_confidence"],
            depth_focus_discrepancy=discrepancy,
            defocus_residual=residuals["defocus_residual"],
            refocus_residual=residuals["refocus_residual"],
            focus_support=focus_support,
            generation_support=generation_support,
            conflict_score=conflict_score,
            invalid_score=invalid_score,
            verdict_logits=verdict_logits,
        )
