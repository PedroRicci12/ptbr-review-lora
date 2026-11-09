"""Train the baselines on the B2W pool and evaluate them on RePro dev (never on test, decision D12).

Usage (after scripts/prepare_data.py):
    uv run python scripts/run_baselines.py

Baselines:
    majority  always predicts the most frequent label of the training pool
    tfidf_lr  TF-IDF (word 1-2 grams) + logistic regression, in 4 variants:
              input = text or title + text; class weights = none or balanced

Output: reports/baselines_dev.json (metrics only, no text).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from ptbr_review_lora.data import model_input
from ptbr_review_lora.metrics import (
    bootstrap_ci,
    compute_metrics,
    neutral_recall_by_group,
    paired_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "baselines_dev.json"

SEED = 42
N_RESAMPLES = 1000

# Fixed, not tuned: the point is a reasonable classical baseline, not the best possible one.
TFIDF_PARAMS = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
LR_PARAMS = {"max_iter": 1000, "C": 1.0}

VARIANTS = {
    "tfidf_lr__text": {"use_title": False, "class_weight": None},
    "tfidf_lr__text__balanced": {"use_title": False, "class_weight": "balanced"},
    "tfidf_lr__title_text": {"use_title": True, "class_weight": None},
    "tfidf_lr__title_text__balanced": {"use_title": True, "class_weight": "balanced"},
}
REFERENCE = "tfidf_lr__text"


def evaluate(dev: pd.DataFrame, y_pred: list[str]) -> dict:
    y_true = dev["label"].tolist()
    point, low, high = bootstrap_ci(y_true, y_pred, n_resamples=N_RESAMPLES, seed=SEED)
    return {
        **compute_metrics(y_true, y_pred),
        "macro_f1_ci95": [low, high],
        "neutral_recall_by_group": neutral_recall_by_group(dev["polarity_group"], y_pred),
    }


def main() -> None:
    pool = pd.read_parquet(PROCESSED / "b2w_pool.parquet")
    dev = pd.read_parquet(PROCESSED / "repro_dev.parquet")
    y_true = dev["label"].tolist()

    predictions: dict[str, list[str]] = {}
    results: dict[str, dict] = {}

    majority = pool["label"].mode()[0]
    predictions["majority"] = [majority] * len(dev)
    results["majority"] = {"predicts": majority, **evaluate(dev, predictions["majority"])}

    for name, cfg in VARIANTS.items():
        start = time.perf_counter()
        model = make_pipeline(
            TfidfVectorizer(**TFIDF_PARAMS),
            LogisticRegression(class_weight=cfg["class_weight"], **LR_PARAMS),
        )
        model.fit(model_input(pool, cfg["use_title"]), pool["label"])
        predictions[name] = model.predict(model_input(dev, cfg["use_title"])).tolist()
        results[name] = {
            **cfg,
            "train_seconds": round(time.perf_counter() - start, 1),
            **evaluate(dev, predictions[name]),
        }
        print(f"{name}: macro-F1 {results[name]['macro_f1']:.4f}")

    comparisons = {
        f"{name} vs {REFERENCE}": paired_bootstrap(
            y_true, predictions[REFERENCE], predictions[name], n_resamples=N_RESAMPLES, seed=SEED
        )
        for name in predictions
        if name != REFERENCE
    }

    report = {
        "split": "repro_dev",
        "n": len(dev),
        "seed": SEED,
        "n_resamples": N_RESAMPLES,
        "tfidf_params": {**TFIDF_PARAMS, "ngram_range": list(TFIDF_PARAMS["ngram_range"])},
        "lr_params": LR_PARAMS,
        "results": results,
        "paired_comparisons": comparisons,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
