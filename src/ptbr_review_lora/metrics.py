"""Evaluation metrics: macro-F1 (main metric), secondary metrics and bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ptbr_review_lora.data import LABELS

Metric = Callable[[Sequence[str], Sequence[str]], float]


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Unweighted mean of the per-class F1 over all 3 labels (decision D10).

    `labels=LABELS` makes a class that is never predicted count as F1 = 0 instead of being skipped.
    """
    return float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0))


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    per_class = f1_score(y_true, y_pred, labels=list(LABELS), average=None, zero_division=0)
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_per_class": {label: float(f) for label, f in zip(LABELS, per_class)},
        # Rows = true label, columns = predicted label, both in LABELS order.
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
    }


def neutral_recall_by_group(
    groups: Sequence[str], y_pred: Sequence[str]
) -> dict[str, float]:
    """Fraction of pure-neutral and of mixed reviews predicted as `neutro` (decision D6).

    Recall, not F1: a false `neutro` prediction cannot be attributed to either subgroup,
    so precision (and therefore F1) is not defined per subgroup.
    """
    groups = np.asarray(groups)
    y_pred = np.asarray(y_pred)
    return {
        group: float(np.mean(y_pred[groups == group] == "neutro"))
        for group in ("neutro", "misto")
        if np.any(groups == group)
    }


# --- Bootstrap confidence intervals (tests in tests/test_metrics.py). ---


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Metric = macro_f1,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for `metric`.

    Returns (point, low, high): `point` is the metric on the full data; `low` and `high` are the
    percentiles of the metric over `n_resamples` resamples drawn with replacement.
    Raises ValueError if `y_true` and `y_pred` have different lengths.
    """
    # 1. NumPy arrays can be indexed by an array of positions; Python lists cannot.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # 2. A random generator created from the seed: same seed, same draws, same result.
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # 3. The metric on the full data is the number we report.
    point = metric(y_true, y_pred)

    # 4. Resample n positions with replacement, many times, and recompute the metric each time.
    #    Some reviews appear twice in a resample, others not at all: that simulates "another sample".
    resampled = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled.append(metric(y_true[idx], y_pred[idx]))

    # 5. The interval cuts alpha from each tail: 95% -> percentiles 2.5 and 97.5.
    alpha = (1 - confidence) / 2
    low = float(np.percentile(resampled, 100 * alpha))
    high = float(np.percentile(resampled, 100 * (1 - alpha)))
    return point, low, high


def paired_bootstrap(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: Metric = macro_f1,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap for the difference `metric(B) - metric(A)` (decision D11).

    Each resample uses the same indices for A and B. Returns:
        diff:  metric(B) - metric(A) on the full data
        low, high: percentile interval of the resampled differences
        p_b_not_better: fraction of resamples where the difference is <= 0
    Raises ValueError if the three sequences do not have the same length.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not len(y_true) == len(pred_a) == len(pred_b):
        raise ValueError("all inputs must have the same length")

    rng = np.random.default_rng(seed)
    n = len(y_true)

    diff = metric(y_true, pred_b) - metric(y_true, pred_a)

    diffs = []
    for _ in range(n_resamples):
        # Drawn once and used for both models: they are compared on exactly the same reviews.
        idx = rng.integers(0, n, size=n)
        diffs.append(metric(y_true[idx], pred_b[idx]) - metric(y_true[idx], pred_a[idx]))

    diffs = np.asarray(diffs)
    alpha = (1 - confidence) / 2
    low = float(np.percentile(diffs, 100 * alpha))
    high = float(np.percentile(diffs, 100 * (1 - alpha)))
    # Ties count as "not better": an identical model must not look like an improvement.
    p_b_not_better = float(np.mean(diffs <= 0))
    return {"diff": diff, "low": low, "high": high, "p_b_not_better": p_b_not_better}
