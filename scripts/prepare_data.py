"""Download B2W-Reviews01 and RePro, clean them, remove the overlap and create the fixed splits.

Usage:
    uv run python scripts/prepare_data.py

Outputs (data/ is git-ignored):
    data/raw/                      original CSVs, pinned and checksum-verified
    data/processed/b2w_pool.parquet   B2W without RePro reviews or their texts (training pool)
    data/processed/repro_dev.parquet  2,000 RePro reviews for analysis and tuning
    data/processed/repro_test.parquet 8,003 RePro reviews, used only for final evaluation
    reports/data_summary.json      counts only (no text), committed to the repository
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ptbr_review_lora.data import (
    B2W_GIT_BLOB_SHA,
    B2W_URL,
    REPRO_URL,
    download,
    prepare_b2w,
    prepare_repro,
    remove_overlap,
    remove_text_duplicates,
    split_repro,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SUMMARY = ROOT / "reports" / "data_summary.json"

SEED = 42
DEV_SIZE = 2000

# Read identifiers as text so the B2W and RePro keys compare equal.
KEY_DTYPES = {"reviewer_id": "string", "product_id": "string", "submission_date": "string"}


def counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def main() -> None:
    b2w_csv = download(B2W_URL, RAW / "B2W-Reviews01.csv", expected_git_sha=B2W_GIT_BLOB_SHA)
    repro_csv = download(REPRO_URL, RAW / "RePro.csv")

    b2w_raw = pd.read_csv(b2w_csv, dtype=KEY_DTYPES)
    repro_raw = pd.read_csv(repro_csv, dtype=KEY_DTYPES)

    b2w = prepare_b2w(b2w_raw)
    repro = prepare_repro(repro_raw)
    b2w_no_overlap, removed = remove_overlap(b2w, repro)
    b2w_pool, removed_text_dups = remove_text_duplicates(b2w_no_overlap, repro)
    dev, test = split_repro(repro, dev_size=DEV_SIZE, seed=SEED)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    b2w_pool.to_parquet(PROCESSED / "b2w_pool.parquet", index=False)
    dev.to_parquet(PROCESSED / "repro_dev.parquet", index=False)
    test.to_parquet(PROCESSED / "repro_test.parquet", index=False)

    summary = {
        "seed": SEED,
        "b2w": {
            "raw_rows": len(b2w_raw),
            "after_cleaning": len(b2w),
            "removed_overlap_with_repro": removed,
            "removed_same_text_as_repro": removed_text_dups,
            "pool_rows": len(b2w_pool),
            "pool_label_counts": counts(b2w_pool["label"]),
            "pool_rating_counts": counts(b2w_pool["overall_rating"]),
        },
        "repro": {
            "raw_rows": len(repro_raw),
            "dev_rows": len(dev),
            "test_rows": len(test),
            "dev_group_counts": counts(dev["polarity_group"]),
            "test_group_counts": counts(test["polarity_group"]),
        },
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
