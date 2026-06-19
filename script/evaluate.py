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


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    """Return numerator/denominator or NaN when the denominator is empty."""
    if float(denominator.item()) == 0.0:
        return float("nan")
    return float((numerator / denominator).item())


def _sample_vpr_at_coverage(
    confidence: torch.Tensor,
    conflict: torch.Tensor,
    invalid: torch.Tensor,
    coverage: float,
    conflict_threshold: float,
    invalid_threshold: float,
) -> tuple[float, float]:
    """Compute per-sample VPR@coverage mean/std and accepted coverage."""
    batch = confidence.shape[0]
    coverage = min(max(float(coverage), 0.0), 1.0)
    sample_rates: list[float] = []
    accepted_coverages: list[float] = []
    for index in range(batch):
        flat_confidence = confidence[index].flatten()
        flat_conflict = conflict[index].flatten()
        flat_invalid = invalid[index].flatten()
        if flat_confidence.numel() == 0:
            sample_rates.append(float("nan"))
            accepted_coverages.append(float("nan"))
            continue
        coverage_count = max(1, min(flat_confidence.numel(), int(math.ceil(flat_confidence.numel() * coverage))))
        top_indices = torch.topk(flat_confidence, k=coverage_count).indices
        pass_mask = (flat_conflict[top_indices] < conflict_threshold) & (flat_invalid[top_indices] < invalid_threshold)
        sample_rates.append(float(pass_mask.float().mean().item()))
        accepted_coverages.append(float(coverage_count / flat_confidence.numel()))
    rates = torch.tensor(sample_rates, dtype=torch.float32)
    valid_rates = rates[torch.isfinite(rates)]
    if valid_rates.numel() == 0:
        return float("nan"), float("nan")
    std = float(valid_rates.std(unbiased=False).item()) if valid_rates.numel() > 1 else 0.0
    return float(valid_rates.mean().item()), std


def compute_trace_metrics(
    output,
    confidence_threshold: float = 0.8,
    violation_threshold: float = 0.5,
    coverage: float = 0.2,
    invalid_threshold: float | None = None,
    conflict_threshold: float | None = None,
) -> dict[str, float]:
    """Compute FocalTrace metrics from a pipeline output object."""
    invalid_threshold = violation_threshold if invalid_threshold is None else invalid_threshold
    conflict_threshold = violation_threshold if conflict_threshold is None else conflict_threshold
    trace = getattr(output, "physical_verification_trace", None)
    if trace is None:
        nan = float("nan")
        return {
            "physical_hallucination_rate": float("nan"),
            "invalid_overconfidence_rate": float("nan"),
            "VPR_at_coverage": float("nan"),
            "VPR_at_coverage_std": float("nan"),
            "coverage": float(coverage),
            "accepted_coverage": float("nan"),
            "mean_conflict_score": float("nan"),
            "mean_invalid_score": float("nan"),
            "invalid_calibration_error": float("nan"),
            "diagnostics": {},
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

    uncertainty = uncertainty.float().clamp(0.0, 1.0)
    confidence = (1.0 - uncertainty).clamp(0.0, 1.0)
    high_confidence = confidence >= confidence_threshold
    physically_valid_evidence = invalid < invalid_threshold
    conflict_event = conflict >= conflict_threshold
    phr_denominator = (high_confidence & physically_valid_evidence).float().sum()
    phr_numerator = (high_confidence & physically_valid_evidence & conflict_event).float().sum()
    ior_denominator = (invalid >= invalid_threshold).float().sum()
    ior_numerator = (high_confidence & (invalid >= invalid_threshold)).float().sum()
    phr = _safe_ratio(phr_numerator, phr_denominator)
    ior = _safe_ratio(ior_numerator, ior_denominator)

    vpr_mean, vpr_std = _sample_vpr_at_coverage(
        confidence=confidence,
        conflict=conflict,
        invalid=invalid,
        coverage=coverage,
        conflict_threshold=conflict_threshold,
        invalid_threshold=invalid_threshold,
    )

    physically_valid = (invalid < invalid_threshold).to(dtype=focus_support.dtype)
    focus_supported_valid_mass = (focus_support * physically_valid).mean()
    invalid_calibration_error = torch.abs(uncertainty - invalid).mean()
    accepted_coverage = float(phr_denominator.item() / max(high_confidence.numel(), 1))

    return {
        "physical_hallucination_rate": phr,
        "invalid_overconfidence_rate": ior,
        "VPR_at_coverage": vpr_mean,
        "VPR_at_coverage_std": vpr_std,
        "coverage": float(coverage),
        "accepted_coverage": accepted_coverage,
        "mean_conflict_score": float(conflict.mean().item()),
        "mean_invalid_score": float(invalid.mean().item()),
        "invalid_calibration_error": float(invalid_calibration_error.item()),
        "diagnostics": {
            "focus_supported_valid_mass": float(focus_supported_valid_mass.item()),
            "phr_denominator": float(phr_denominator.item()),
            "ior_denominator": float(ior_denominator.item()),
        },
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
