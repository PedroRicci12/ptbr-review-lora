"""Pre-registered protocol for the single test-set evaluation (Stage 6, decision D17).

Everything here was fixed and committed BEFORE any test metric was computed: the models, the
three primary comparisons, the multiple-comparison correction and the rule that turns an
interval into a conclusion. Changing it after seeing test results would defeat its purpose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ptbr_review_lora.metrics import paired_bootstrap

# Frozen models -> file name in predictions/test/ (all chosen on dev, D12).
MODELS = {
    "majority": "majority",
    "tfidf": "tfidf_lr__title_text__balanced",
    "bertimbau": "bertimbau",
    "zero_shot": "smollm3_base",
    "qlora_stars": "smollm3_qlora_stars",
    "qlora_human": "smollm3_qlora_human",
}

# (id, model B, model A, question): the difference is metric(B) - metric(A).
PRIMARY = (
    ("H1", "qlora_human", "zero_shot", "Does fine-tuning on 1,500 human labels improve the LLM?"),
    ("H2", "qlora_stars", "zero_shot", "Does fine-tuning on 15,000 star labels improve the LLM?"),
    ("H3", "qlora_human", "qlora_stars", "Do 1,500 human labels beat 15,000 star labels?"),
)

# Reported with 95% intervals, without the status of a conclusion.
DESCRIPTIVE = (
    ("qlora_human", "bertimbau"),
    ("qlora_human", "tfidf"),
    ("zero_shot", "bertimbau"),
    ("bertimbau", "tfidf"),
)

FAMILY_ALPHA = 0.05
N_RESAMPLES = 10_000
SEED = 42


def bonferroni_confidence(n_comparisons: int, alpha: float = FAMILY_ALPHA) -> float:
    """Confidence level per comparison so the family-wise error stays at `alpha` (Bonferroni)."""
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be at least 1")
    return 1 - alpha / n_comparisons


def verdict(low: float, high: float) -> str:
    """Pre-registered reading of an interval for metric(B) - metric(A)."""
    if low > 0:
        return "B better"
    if high < 0:
        return "B worse"
    return "inconclusive"


def run_primary(
    y_true: Sequence[str],
    preds: Mapping[str, Sequence[str]],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict[str, dict]:
    """The three primary comparisons with Bonferroni-corrected intervals and their verdicts."""
    confidence = bonferroni_confidence(len(PRIMARY))
    out = {}
    for comparison_id, b, a, question in PRIMARY:
        result = paired_bootstrap(
            y_true, preds[a], preds[b], n_resamples=n_resamples, confidence=confidence, seed=seed
        )
        out[comparison_id] = {
            "question": question,
            "b": b,
            "a": a,
            "confidence": confidence,
            **result,
            "verdict": verdict(result["low"], result["high"]),
        }
    return out


def run_descriptive(
    y_true: Sequence[str],
    preds: Mapping[str, Sequence[str]],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> dict[str, dict]:
    return {
        f"{b} vs {a}": paired_bootstrap(
            y_true, preds[a], preds[b], n_resamples=n_resamples, confidence=0.95, seed=seed
        )
        for b, a in DESCRIPTIVE
    }
