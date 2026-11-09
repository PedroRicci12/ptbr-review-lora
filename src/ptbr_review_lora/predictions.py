"""Save and load per-review predictions, so models can be compared later without rerunning them.

A prediction file is a CSV with two columns: `row` (position in the split's parquet file, which
is deterministic for a fixed seed) and `pred` (predicted label). It contains no review text.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def save_predictions(path: Path, preds: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": range(len(preds)), "pred": list(preds)}).to_csv(
        path, index=False, lineterminator="\n"
    )


def load_predictions(path: Path, n_rows: int) -> list[str]:
    """Load predictions and check they cover rows 0..n_rows-1 in order."""
    df = pd.read_csv(path)
    if list(df["row"]) != list(range(n_rows)):
        raise ValueError(f"{path} does not cover rows 0..{n_rows - 1} in order")
    return df["pred"].tolist()
