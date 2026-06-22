#!/usr/bin/env python3
"""Evaluate uncertainty/reliability maps for high-error depth detection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.utils.metrics import ause, rejection_auc, sparsification_curve

logger = logging.getLogger(__name__)

SCORES = ("uncertainty_final", "focal_entropy", "depth_disagreement", "abstention_weight", "random")


def _require_sklearn():
    if importlib.util.find_spec("sklearn") is None:  # pragma: no cover - environment dependent
        raise ImportError(
            "script.evaluate_reliability requires scikit-learn. Install it with `pip install scikit-learn`."
        )

    from sklearn.metrics import average_precision_score, roc_auc_score

    return average_precision_score, roc_auc_score


def _flatten_valid(depth_pred: np.ndarray, depth_gt: np.ndarray, valid_mask: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(depth_pred, dtype=np.float64)
    gt = np.asarray(depth_gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"depth_pred and depth_gt must have the same shape, got {pred.shape} and {gt.shape}")

    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask).astype(bool)
    if not np.any(valid):
        raise ValueError("No valid pixels remain after applying valid_mask and finite/positive-depth checks")
    return pred[valid], gt[valid], valid


def _align_score(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if score.shape != valid.shape:
        score = np.squeeze(score)
    if score.shape != valid.shape:
        raise ValueError(f"uncertainty score shape {score.shape} does not match valid shape {valid.shape}")
    values = score[valid]
    finite = np.isfinite(values)
    if not np.all(finite):
        fill = np.nanmedian(values[finite]) if np.any(finite) else 0.0
        values = np.where(finite, values, fill)
    return values


def high_error_labels(abs_rel: np.ndarray, mode: str = "top_percent", high_error_percent: float = 10.0, rel_error_threshold: float = 0.1) -> np.ndarray:
    """Return binary labels identifying high-error pixels."""
    if mode == "top_percent":
        if not (0 < high_error_percent < 100):
            raise ValueError("high_error_percent must be between 0 and 100")
        threshold = np.percentile(abs_rel, 100.0 - high_error_percent)
        return abs_rel >= threshold
    if mode == "threshold":
        return abs_rel >= rel_error_threshold
    raise ValueError(f"Unsupported high-error mode: {mode}")


def evaluate_scores(arrays: Dict[str, np.ndarray], high_error_mode: str = "top_percent", high_error_percent: float = 10.0, rel_error_threshold: float = 0.1, seed: int = 0) -> Dict[str, Dict[str, float | list]]:
    """Evaluate all available uncertainty scores in a prediction dictionary."""
    average_precision_score, roc_auc_score = _require_sklearn()
    pred, gt, valid = _flatten_valid(arrays["depth_pred"], arrays["depth_gt"], arrays.get("valid_mask"))
    abs_rel = np.abs(pred - gt) / np.maximum(np.abs(gt), 1e-6)
    high_error = high_error_labels(abs_rel, high_error_mode, high_error_percent, rel_error_threshold).astype(np.int32)
    if high_error.min() == high_error.max():
        raise ValueError("High-error labels contain only one class; adjust threshold or top-percent setting")

    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[str, float | list]] = {}
    for name in SCORES:
        if name == "random":
            score = rng.random(abs_rel.shape)
        elif name in arrays:
            try:
                score = _align_score(arrays[name], valid)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", name, exc)
                continue
        else:
            logger.warning("Skipping missing uncertainty field: %s", name)
            continue

        fractions, curve = sparsification_curve(abs_rel, score)
        corr = float(np.corrcoef(score, abs_rel)[0, 1]) if np.std(score) > 0 and np.std(abs_rel) > 0 else 0.0
        results[name] = {
            "auroc": float(roc_auc_score(high_error, score)),
            "auprc": float(average_precision_score(high_error, score)),
            "ause_absrel": ause(abs_rel, score),
            "error_correlation": corr,
            "rejection_auc_absrel": rejection_auc(abs_rel, score),
            "sparsification_fractions": fractions.tolist(),
            "sparsification_absrel": curve.tolist(),
        }
    return results


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FocalStackGeneration uncertainty maps for high-error detection")
    parser.add_argument("--pred", required=True, help="Prediction .npz containing depth_pred, depth_gt, valid_mask, and uncertainty maps")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--high-error-mode", choices=["top_percent", "threshold"], default="top_percent")
    parser.add_argument("--high-error-percent", type=float, default=10.0)
    parser.add_argument("--rel-error-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    arrays = _load_npz(Path(args.pred))
    results = evaluate_scores(
        arrays,
        high_error_mode=args.high_error_mode,
        high_error_percent=args.high_error_percent,
        rel_error_threshold=args.rel_error_threshold,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote reliability evaluation to %s", output)


if __name__ == "__main__":
    main()
