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

from src.utils.metrics import (
    binary_auprc as _binary_auprc,
    binary_auroc as _binary_auroc,
    risk_coverage_auc as _physical_aurc_for_sample,
    selective_risk_at_coverage as _selective_risk_for_sample,
    spearman_correlation as _spearman_correlation,
)


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


def _finite_mean(values: list[float]) -> float:
    finite = torch.tensor([value for value in values if math.isfinite(value)], dtype=torch.float32)
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().item())


def _agreement_metrics(predicted_verdict: torch.Tensor, reference_verdict: torch.Tensor) -> dict[str, float]:
    pred = predicted_verdict.detach().flatten().long()
    ref = reference_verdict.detach().flatten().long().to(device=pred.device)
    valid = torch.isfinite(ref.float())
    pred = pred[valid]
    ref = ref[valid]
    if pred.numel() == 0:
        return {"verifier_agreement_accuracy": float("nan"), "verifier_agreement_macro_f1": float("nan")}
    accuracy = float((pred == ref).float().mean().item())
    classes = torch.unique(torch.cat([pred, ref])).tolist()
    f1s: list[float] = []
    for cls in classes:
        pred_pos = pred == int(cls)
        ref_pos = ref == int(cls)
        tp = (pred_pos & ref_pos).float().sum()
        fp = (pred_pos & ~ref_pos).float().sum()
        fn = (~pred_pos & ref_pos).float().sum()
        denom = 2 * tp + fp + fn
        f1s.append(float("nan") if float(denom.item()) == 0.0 else float((2 * tp / denom).item()))
    return {"verifier_agreement_accuracy": accuracy, "verifier_agreement_macro_f1": _finite_mean(f1s)}


def compute_trace_metrics(
    output,
    confidence_threshold: float = 0.8,
    violation_threshold: float = 0.5,
    coverage: float = 1.0,
    invalid_threshold: float | None = None,
    conflict_threshold: float | None = None,
    reference_type: str = "internal_verifier",
) -> dict[str, object]:
    """Compute verifier-derived FocalTrace metrics with explicit reference semantics."""
    allowed_reference_types = {"internal_verifier", "heldout_verifier", "synthetic_ground_truth", "human_annotation"}
    if reference_type not in allowed_reference_types:
        raise ValueError(f"reference_type must be one of {sorted(allowed_reference_types)}, got {reference_type!r}")
    invalid_threshold = violation_threshold if invalid_threshold is None else invalid_threshold
    conflict_threshold = violation_threshold if conflict_threshold is None else conflict_threshold
    trace = getattr(output, "physical_verification_trace", None)
    if trace is None:
        nan = float("nan")
        return {
            "reference_type": reference_type,
            "high_confidence_physical_violation_rate": nan,
            "HCPVR": nan,
            "selective_physical_risk_at_coverage": nan,
            "physical_risk_coverage_auc": nan,
            "Physical-AURC": nan,
            "invalid_overconfidence_rate": nan,
            "accepted_coverage": nan,
            "coverage": float(coverage),
            "internal_violation_detection_auroc": nan,
            "internal_violation_detection_auprc": nan,
            "error_violation_detection_auroc": nan,
            "error_violation_detection_auprc": nan,
            "gt_depth_error_detection_auroc": nan,
            "gt_depth_error_detection_auprc": nan,
            "gt_depth_risk_coverage_auc": nan,
            "uncertainty_error_spearman": nan,
            "diagnostics": {},
        }

    conflict = trace.conflict_score.detach().float().clamp(0.0, 1.0)
    invalid = trace.invalid_score.detach().float().clamp(0.0, 1.0)
    physical_risk = torch.maximum(conflict, invalid).clamp(0.0, 1.0)
    uncertainty = _as_bchw(getattr(output, "uncertainty_final", None))
    if uncertainty is None:
        uncertainty = _as_bchw(getattr(output, "uncertainty", None))
    if uncertainty is None:
        uncertainty = torch.zeros_like(conflict)
    else:
        uncertainty = uncertainty.detach().float().clamp(0.0, 1.0)
        if uncertainty.shape[-2:] != conflict.shape[-2:]:
            uncertainty = F.interpolate(uncertainty, size=conflict.shape[-2:], mode="bilinear", align_corners=False)
    confidence = (1.0 - uncertainty.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
    high_confidence = confidence >= confidence_threshold
    violation_event = (conflict >= conflict_threshold) | (invalid >= invalid_threshold)
    physically_accepted = invalid < invalid_threshold

    sample_hcpvr: list[float] = []
    sample_ior: list[float] = []
    sample_accepted_coverage: list[float] = []
    sample_selective_risk: list[float] = []
    sample_aurc: list[float] = []
    for index in range(conflict.shape[0]):
        high = high_confidence[index]
        violation = violation_event[index]
        invalid_event = invalid[index] >= invalid_threshold
        accepted = physically_accepted[index]
        high_count = high.float().sum()
        invalid_count = invalid_event.float().sum()
        sample_hcpvr.append(_safe_ratio((high & violation).float().sum(), high_count))
        sample_ior.append(_safe_ratio((high & invalid_event).float().sum(), invalid_count))
        sample_accepted_coverage.append(float((high & accepted).float().sum().item() / max(high.numel(), 1)))
        sample_selective_risk.append(_selective_risk_for_sample(confidence[index], physical_risk[index], coverage))
        sample_aurc.append(_physical_aurc_for_sample(confidence[index], physical_risk[index]))

    auroc = _binary_auroc(uncertainty, violation_event)
    auprc = _binary_auprc(uncertainty, violation_event)
    spearman = _spearman_correlation(uncertainty, physical_risk)
    hcpvr = _finite_mean(sample_hcpvr)
    physical_aurc = _finite_mean(sample_aurc)
    metrics: dict[str, object] = {
        "reference_type": reference_type,
        "high_confidence_physical_violation_rate": hcpvr,
        "HCPVR": hcpvr,
        "selective_physical_risk_at_coverage": _finite_mean(sample_selective_risk),
        "physical_risk_coverage_auc": physical_aurc,
        "Physical-AURC": physical_aurc,
        "invalid_overconfidence_rate": _finite_mean(sample_ior),
        "accepted_coverage": _finite_mean(sample_accepted_coverage),
        "coverage": float(coverage),
        "mean_conflict_score": float(conflict.mean().item()),
        "mean_invalid_score": float(invalid.mean().item()),
        "internal_violation_detection_auroc": auroc,
        "internal_violation_detection_auprc": auprc,
        "error_violation_detection_auroc": auroc,
        "error_violation_detection_auprc": auprc,
        "uncertainty_error_spearman": spearman,
        "diagnostics": {
            "high_confidence_denominator": float(high_confidence.float().sum().item()),
            "invalid_denominator": float((invalid >= invalid_threshold).float().sum().item()),
            "sample_count": int(conflict.shape[0]),
        },
    }
    depth_gt = _as_bchw(getattr(output, "depth_gt", None))
    depth_pred = _as_bchw(getattr(output, "depth_pred", None))
    if depth_gt is not None and depth_pred is not None:
        if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
            depth_pred = F.interpolate(depth_pred.float(), size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)
        gt_error = torch.abs(depth_pred.float() - depth_gt.float())
        if gt_error.shape[-2:] != uncertainty.shape[-2:]:
            gt_error = F.interpolate(gt_error, size=uncertainty.shape[-2:], mode="bilinear", align_corners=False)
        threshold = float(getattr(output, "depth_error_threshold", gt_error.flatten().median().item()))
        gt_error_event = gt_error >= threshold
        metrics["gt_depth_error_detection_auroc"] = _binary_auroc(uncertainty, gt_error_event)
        metrics["gt_depth_error_detection_auprc"] = _binary_auprc(uncertainty, gt_error_event)
        metrics["gt_depth_risk_coverage_auc"] = _finite_mean([
            _physical_aurc_for_sample(confidence[index], gt_error[index].clamp(min=0.0))
            for index in range(gt_error.shape[0])
        ])

    reference_verdict = getattr(output, "reference_verdict", None)
    if reference_verdict is None:
        reference_verdict = getattr(trace, "reference_verdict", None)
    if reference_verdict is not None:
        predicted_verdict = torch.stack([
            1.0 - physical_risk.squeeze(1),
            conflict.squeeze(1),
            invalid.squeeze(1),
        ], dim=1).argmax(dim=1)
        metrics.update(_agreement_metrics(predicted_verdict, _as_bchw(reference_verdict).squeeze(1)))
    return metrics


def _summarize_numeric_metrics(metric_rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize sample-level numeric metrics without flattening pixels across samples."""
    summary: dict[str, object] = {}
    keys = sorted({key for row in metric_rows for key in row.keys()})
    for key in keys:
        values = [row[key] for row in metric_rows if key in row and isinstance(row[key], (int, float))]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(np.nanmean(array)),
            "std": float(np.nanstd(array)),
            "median": float(np.nanmedian(array)),
        }
    reference_types = sorted({str(row.get("reference_type")) for row in metric_rows if row.get("reference_type") is not None})
    if reference_types:
        summary["reference_type"] = reference_types[0] if len(reference_types) == 1 else reference_types
    summary["aggregation"] = "sample_mean"
    return summary


def aggregate_dataset_metric_groups(dataset_metrics: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """Aggregate per-dataset sample metrics as dataset macro means, never pixel-flattened."""
    per_dataset = {name: _summarize_numeric_metrics(rows) for name, rows in dataset_metrics.items()}
    macro: dict[str, object] = {}
    numeric_keys = sorted({key for metrics in per_dataset.values() for key, value in metrics.items() if isinstance(value, dict) and "mean" in value})
    for key in numeric_keys:
        values = [metrics[key]["mean"] for metrics in per_dataset.values() if key in metrics and isinstance(metrics[key], dict)]
        if values:
            macro[key] = float(np.nanmean(np.asarray(values, dtype=np.float64)))
    return {"per_dataset": per_dataset, "dataset_macro_mean": macro, "aggregation": "sample_mean_then_dataset_macro_mean"}

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
    physical_verdict = trace.verdict_scores.detach().argmax(dim=1, keepdim=True).float()
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

    final_metrics = _summarize_numeric_metrics(all_metrics)
    final_trace_metrics = _summarize_numeric_metrics(all_trace_metrics)
    final_trace_metrics["dataset"] = args.dataset
    final_trace_metrics["thresholds"] = {
        "confidence_threshold": float(args.confidence_threshold),
        "violation_threshold": float(args.violation_threshold),
        "coverage": float(args.coverage),
    }

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
        help='Top-confidence coverage fraction for selective_physical_risk_at_coverage',
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
