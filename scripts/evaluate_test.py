"""Single evaluation of all frozen models on the RePro test set (Stage 6, decisions D12 and D17).

Usage (after scripts/prepare_data.py; runs on CPU):
    uv run python scripts/evaluate_test.py

The comparisons are pre-registered in `ptbr_review_lora.protocol`. The script refuses to run if
`reports/test_results.json` already exists: the test set is used once.

Outputs:
    predictions/test/tfidf_lr__title_text__balanced.csv   TF-IDF test predictions (the only model
                                                          whose test predictions were not saved yet)
    predictions/test/majority.csv
    reports/test_results.json                             metrics only, no text
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ptbr_review_lora import protocol
from ptbr_review_lora.baselines import make_tfidf_lr
from ptbr_review_lora.data import model_input
from ptbr_review_lora.metrics import bootstrap_ci, compute_metrics, neutral_recall_by_group
from ptbr_review_lora.predictions import load_predictions, save_predictions

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PREDICTIONS = ROOT / "predictions" / "test"
REPORT = ROOT / "reports" / "test_results.json"


def main() -> None:
    if REPORT.exists():
        sys.exit(f"{REPORT.relative_to(ROOT)} already exists: the test set is evaluated only once (D12).")

    start = time.perf_counter()
    pool = pd.read_parquet(PROCESSED / "b2w_pool.parquet")
    test = pd.read_parquet(PROCESSED / "repro_test.parquet")
    y_test = test["label"].tolist()

    # Models trained on the laptop: same configuration as chosen on dev (D13).
    tfidf = make_tfidf_lr(class_weight="balanced")
    tfidf.fit(model_input(pool, use_title=True), pool["label"])
    save_predictions(PREDICTIONS / f"{protocol.MODELS['tfidf']}.csv", tfidf.predict(model_input(test, use_title=True)).tolist())
    save_predictions(PREDICTIONS / f"{protocol.MODELS['majority']}.csv", [pool["label"].mode()[0]] * len(test))

    preds = {
        name: load_predictions(PREDICTIONS / f"{stem}.csv", len(test))
        for name, stem in protocol.MODELS.items()
    }

    results = {}
    for name, y_pred in preds.items():
        _, low, high = bootstrap_ci(y_test, y_pred, n_resamples=protocol.N_RESAMPLES, seed=protocol.SEED)
        results[name] = {
            **compute_metrics(y_test, y_pred),
            "macro_f1_ci95": [low, high],
            "neutral_recall_by_group": neutral_recall_by_group(test["polarity_group"], y_pred),
        }
        print(f"{name}: macro-F1 {results[name]['macro_f1']:.4f} (95% CI {low:.4f}-{high:.4f})")

    primary = protocol.run_primary(y_test, preds)
    for comparison_id, c in primary.items():
        print(f"{comparison_id} {c['b']} vs {c['a']}: diff {c['diff']:+.4f} "
              f"({c['confidence']:.1%} CI {c['low']:+.4f} to {c['high']:+.4f}) -> {c['verdict']}")
    descriptive = protocol.run_descriptive(y_test, preds)

    report = {
        "split": "repro_test",
        "n": len(test),
        "protocol_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "n_resamples": protocol.N_RESAMPLES,
        "seed": protocol.SEED,
        "family_alpha": protocol.FAMILY_ALPHA,
        "models": protocol.MODELS,
        "results": results,
        "primary_comparisons": primary,
        "descriptive_comparisons": descriptive,
        "seconds": round(time.perf_counter() - start),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
