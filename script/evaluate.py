"""
Evaluation script for FocalStackGeneration with normalized-depth aware metrics.
"""

import argparse
import json
import math
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import yaml
import torch.nn.functional as F


def _as_bchw(value: torch.Tensor | None) -> torch.Tensor | None:
    """Convert optional tensor maps to BCHW format for metric computation."""
    if value is None:
        return None
    if value.dim() == 3:
        return value.unsqueeze(1)
    if value.dim() == 4:
        return value
    raise ValueError(f"Expected [B,H,W] or [B,C,H,W] tensor, got {tuple(value.shape)}")


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide tensors and return NaN when a metric is undefined."""
    if float(denominator.item()) <= 0.0:
        return numerator.new_tensor(float("nan"))
    return numerator / denominator


def _per_sample_vpr_at_coverage(
    confidence: torch.Tensor,
    conflict: torch.Tensor,
    invalid: torch.Tensor,
    coverage: float,
    conflict_threshold: float,
    invalid_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute verification pass rate after per-sample confidence top-k retention."""
    batch = confidence.shape[0]
    confidence_flat = confidence.reshape(batch, -1)
    pass_flat = ((conflict < conflict_threshold) & (invalid < invalid_threshold)).reshape(batch, -1).float()
    coverage = min(max(float(coverage), 0.0), 1.0)
    rates = []
    actual_coverages = []
    for sample_idx in range(batch):
        total = confidence_flat.shape[1]
        k = int(torch.ceil(confidence_flat.new_tensor(coverage * total)).item())
        if k <= 0:
            rates.append(confidence_flat.new_tensor(float("nan")))
            actual_coverages.append(confidence_flat.new_tensor(0.0))
            continue
        topk_idx = torch.topk(confidence_flat[sample_idx], k=k, largest=True).indices
        rates.append(pass_flat[sample_idx, topk_idx].mean())
        actual_coverages.append(confidence_flat.new_tensor(k / total))
    return torch.stack(rates), torch.stack(actual_coverages)


def compute_trace_metrics(
    output,
    confidence_threshold: float = 0.5,
    conflict_threshold: float = 0.5,
    invalid_threshold: float = 0.5,
    vpr_coverage: float = 0.8,
    violation_threshold: float | None = None,
) -> dict[str, float]:
    """Compute correctness-first FocalTrace physical verification metrics."""
    if violation_threshold is not None:
        conflict_threshold = violation_threshold
        invalid_threshold = violation_threshold
    trace = getattr(output, "physical_verification_trace", None)
    if trace is None:
        nan = float("nan")
        return {
            "PHR": nan,
            "IOR": nan,
            "VPR_at_coverage_mean": nan,
            "VPR_at_coverage_std": nan,
            "accepted_coverage": 0.0,
            "coverage": 0.0,
            "mean_conflict_score": 0.0,
            "mean_invalid_score": 0.0,
            "uncertainty_violation_alignment": nan,
            "FCPV": nan,
            "FCPV_soft": nan,
            "PVPR_at_coverage": nan,
        }

    conflict = trace.conflict_score.detach().float().clamp(0.0, 1.0)
    invalid = trace.invalid_score.detach().float().clamp(0.0, 1.0)
    uncertainty = _as_bchw(getattr(output, "uncertainty_final", None))
    if uncertainty is None:
        uncertainty = _as_bchw(getattr(output, "uncertainty", None))
    if uncertainty is None:
        uncertainty = torch.zeros_like(conflict)
    else:
        uncertainty = uncertainty.detach().float().clamp(0.0, 1.0)
        if uncertainty.shape[-2:] != conflict.shape[-2:]:
            uncertainty = F.interpolate(uncertainty, size=conflict.shape[-2:], mode="bilinear", align_corners=False)

    confidence = (1.0 - uncertainty).clamp(0.0, 1.0)
    high_confidence = confidence >= confidence_threshold
    physically_valid_evidence = invalid < invalid_threshold
    conflict_event = conflict >= conflict_threshold
    invalid_event = invalid >= invalid_threshold
    accepted = high_confidence & physically_valid_evidence

    phr = _safe_ratio((accepted & conflict_event).float().sum(), accepted.float().sum())
    ior = _safe_ratio((high_confidence & invalid_event).float().sum(), invalid_event.float().sum())
    accepted_coverage = accepted.float().mean()

    vpr_rates, vpr_coverages = _per_sample_vpr_at_coverage(
        confidence=confidence,
        conflict=conflict,
        invalid=invalid,
        coverage=vpr_coverage,
        conflict_threshold=conflict_threshold,
        invalid_threshold=invalid_threshold,
    )
    finite_vpr = vpr_rates[torch.isfinite(vpr_rates)]
    if finite_vpr.numel() == 0:
        vpr_mean = confidence.new_tensor(float("nan"))
        vpr_std = confidence.new_tensor(float("nan"))
    else:
        vpr_mean = finite_vpr.mean()
        vpr_std = finite_vpr.std(unbiased=False)

    physical_risk = (0.5 * (conflict + invalid)).clamp(0.0, 1.0)
    uncertainty_flat = uncertainty.reshape(-1)
    risk_flat = physical_risk.reshape(-1)
    if uncertainty_flat.numel() < 2 or float(uncertainty_flat.std(unbiased=False).item()) <= 1e-12 or float(risk_flat.std(unbiased=False).item()) <= 1e-12:
        alignment = uncertainty.new_tensor(float("nan"))
    else:
        uncertainty_centered = uncertainty_flat - uncertainty_flat.mean()
        risk_centered = risk_flat - risk_flat.mean()
        alignment = (uncertainty_centered * risk_centered).mean() / (
            uncertainty_flat.std(unbiased=False) * risk_flat.std(unbiased=False) + 1e-8
        )

    return {
        "PHR": float(phr.item()),
        "IOR": float(ior.item()),
        "VPR_at_coverage_mean": float(vpr_mean.item()),
        "VPR_at_coverage_std": float(vpr_std.item()),
        "accepted_coverage": float(accepted_coverage.item()),
        "coverage": float(vpr_coverages.mean().item()),
        "mean_conflict_score": float(conflict.mean().item()),
        "mean_invalid_score": float(invalid.mean().item()),
        "uncertainty_violation_alignment": float(alignment.item()),
        # Backward-compatible names now follow correctness-first definitions.
        "FCPV": float(phr.item()),
        "FCPV_soft": float((confidence * conflict * (1.0 - invalid)).sum().div((confidence * (1.0 - invalid)).sum().clamp(min=1e-6)).item()),
        "PVPR_at_coverage": float(vpr_mean.item()),
    }


def _to_numpy_map(value: torch.Tensor) -> np.ndarray:
    """Convert a tensor map to a normalized 2-D numpy array for visualization."""
    value = value.detach().float().cpu()
    if value.dim() == 4:
        value = value[0, 0]
    elif value.dim() == 3:
        value = value[0]
    array = value.numpy()
    mn = float(array.min())
    mx = float(array.max())
    return (array - mn) / (mx - mn + 1e-8)


def _save_map_png(value: torch.Tensor, path: Path, cmap: str = "magma") -> None:
    """Save a single tensor map as a PNG visualization."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, _to_numpy_map(value), cmap=cmap)


def save_trace_visualizations(output, output_dir: Path, prefix: str) -> None:
    """Save depth, uncertainty, and physical trace visualization maps."""
    trace = getattr(output, "physical_verification_trace", None)
    _save_map_png(_as_bchw(output.depth_map), output_dir / f"{prefix}_depth.png", cmap="magma")
    uncertainty = _as_bchw(getattr(output, "uncertainty_final", None))
    if uncertainty is None:
        uncertainty = _as_bchw(getattr(output, "uncertainty", None))
    if uncertainty is not None:
        _save_map_png(uncertainty, output_dir / f"{prefix}_uncertainty.png", cmap="viridis")
    if trace is None:
        return
    _save_map_png(trace.focus_support, output_dir / f"{prefix}_focus_support.png", cmap="viridis")
    _save_map_png(trace.conflict_score, output_dir / f"{prefix}_conflict_score.png", cmap="inferno")
    _save_map_png(trace.invalid_score, output_dir / f"{prefix}_invalid_score.png", cmap="inferno")
    physical_verdict = trace.verdict_logits.detach().argmax(dim=1, keepdim=True).float()
    _save_map_png(physical_verdict, output_dir / f"{prefix}_physical_verdict.png", cmap="tab10")




def _tensor_curve_summary(value: torch.Tensor) -> dict[str, float | list[int]]:
    """Return a compact JSON-serializable summary for a refinement tensor."""
    value = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()) if value.numel() > 1 else 0.0,
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def _serialize_refinement_history(history: list[dict]) -> list[dict]:
    """Convert in-memory refinement history into a compact curve for JSON."""
    curve = []
    for index, item in enumerate(history):
        curve.append(
            {
                "step": int(item.get("step", index)),
                "final_depth_canonical": _tensor_curve_summary(item["final_depth_canonical"]),
                "uncertainty_final": _tensor_curve_summary(item["uncertainty_final"]),
                "mean_conflict_score": float(item["mean_conflict_score"]),
                "mean_invalid_score": float(item["mean_invalid_score"]),
                "mean_focus_support": float(item["mean_focus_support"]),
                "mean_generation_support": float(item["mean_generation_support"]),
            }
        )
    return curve


def evaluate(args):
    from src.pipelines import load_pipeline
    from src.data import create_dataloader
    from src.utils.metrics import compute_metrics

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading model from {args.checkpoint}...")
    pipeline = load_pipeline(
        checkpoint_path=args.checkpoint,
        base_model_id=config['model']['base_model_id'],
        device=args.device,
        dtype=torch.float16 if args.fp16 else torch.float32,
    )
    pipeline.eval()

    dataloader = create_dataloader(
        dataset_type=args.dataset,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=tuple(config['data']['image_size']),
        focal_stack_size=config['data']['focal_stack_size'],
        focal_range=tuple(config['data']['focal_range']),
        augmentation=False,
    )

    all_metrics = []
    all_trace_metrics = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    refinement_curves = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if args.max_samples and batch_idx * args.batch_size >= args.max_samples:
            break

        focal_stack = batch['focal_stack'].to(args.device)
        focal_plane_distances = batch['focal_plane_distances'].to(args.device)
        depth_gt = batch.get('depth')
        depth_range = batch.get('depth_range')
        with torch.no_grad():
            output = pipeline(
                focal_stack=focal_stack,
                focal_plane_distances=focal_plane_distances,
                num_inference_steps=args.num_inference_steps,
                output_type='pt',
                num_refinement_steps=args.num_refinement_steps,
                return_refinement_history=args.num_refinement_steps > 0,
            )

        final_depth_canonical = output.depth_map
        sample_metrics = {}

        if depth_gt is not None and depth_range is not None:
            depth_gt = depth_gt.to(args.device).squeeze(1)
            depth_range = depth_range.to(args.device)
            depth_min = depth_range[:, 0].view(-1, 1, 1)
            depth_max = depth_range[:, 1].view(-1, 1, 1)
            pred_metric = final_depth_canonical * (depth_max - depth_min).clamp(min=1e-6) + depth_min
            sample_metrics.update(compute_metrics(pred_metric, depth_gt))
            sample_metrics["loss"] = F.l1_loss(pred_metric, depth_gt).item()
        elif depth_gt is not None:
            depth_gt = depth_gt.to(args.device)
            if final_depth_canonical.dim() == 3:
                final_depth_canonical = final_depth_canonical.unsqueeze(1)
            if depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(1)
            sample_metrics["normalized_loss"] = F.l1_loss(final_depth_canonical, depth_gt).item()

        trace_metrics = compute_trace_metrics(
            output,
            confidence_threshold=args.confidence_threshold,
            violation_threshold=args.violation_threshold,
            coverage=args.coverage,
        )
        sample_metrics.update(trace_metrics)
        all_trace_metrics.append(trace_metrics)

        if batch_idx == 0 or getattr(args, "save_all_visualizations", False):
            save_trace_visualizations(output, visualization_dir, prefix=f"sample_{batch_idx:05d}")

        refinement_history = getattr(output, "refinement_history", None)
        if refinement_history:
            refinement_curves.append(
                {
                    "sample_index": batch_idx,
                    "history": _serialize_refinement_history(refinement_history),
                }
            )

        uncertainty = getattr(output, "uncertainty_final", None)
        if uncertainty is None:
            uncertainty = output.uncertainty
        sample_metrics["uncertainty_mean"] = float(uncertainty.mean().item()) if uncertainty is not None else 0.0
        sample_metrics["focal_entropy_mean"] = float(output.focal_entropy.mean().item()) if output.focal_entropy is not None else 0.0
        sample_metrics["physical_evidence_support_mean"] = float(output.physical_evidence_support.mean().item()) if output.physical_evidence_support is not None else 0.0
        if output.generated_depth_canonical is not None and output.focal_depth_canonical is not None:
            sample_metrics["generated_focal_depth_disagreement"] = float(torch.abs(output.generated_depth_canonical - output.focal_depth_canonical).mean().item())
        all_metrics.append(sample_metrics)

    keys = sorted({k for m in all_metrics for k in m.keys()})
    final_metrics = {}
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        final_metrics[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'median': float(np.median(values))}

    trace_keys = sorted({k for m in all_trace_metrics for k in m.keys()})
    final_trace_metrics = {}
    for key in trace_keys:
        values = [m[key] for m in all_trace_metrics if key in m]
        final_trace_metrics[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'median': float(np.median(values))}

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    with open(output_dir / 'trace_metrics.json', 'w') as f:
        json.dump(final_trace_metrics, f, indent=2)
    if refinement_curves:
        with open(output_dir / 'refinement_curve.json', 'w') as f:
            json.dump(refinement_curves, f, indent=2)

    print("\nEvaluation Results:")
    print("-" * 40)
    for key, values in final_metrics.items():
        print(f"{key}: {values['mean']:.4f} ± {values['std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FocalStackGeneration model")
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['hypersim', 'virtual_kitti'], required=True)
    parser.add_argument('--data_root', type=str, required=True, help='Dataset root')
    parser.add_argument('--split', type=str, default='test', help='Dataset split')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--num_refinement_steps', type=int, default=0)
    parser.add_argument('--save_all_visualizations', action='store_true')
    parser.add_argument('--max_samples', type=int, help='Maximum samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.8,
        help='Confidence threshold for false confident physical violations',
    )
    parser.add_argument(
        '--violation-threshold',
        type=float,
        default=0.5,
        help='Violation threshold for physical trace metrics',
    )
    parser.add_argument(
        '--coverage',
        type=float,
        default=0.2,
        help='Top-confidence coverage fraction for VPR_at_coverage',
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
