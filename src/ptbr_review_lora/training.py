"""Training data for the QLoRA fine-tuning (Stage 5, decision D16).

Two training sets, fine-tuned separately with the same recipe:

- "stars": a class-balanced sample of the B2W pool, labeled by star rating (noisy);
- "human": 1,500 RePro dev reviews with human labels (see `data.split_dev`).

Each example is the zero-shot prompt of Stage 4 followed by the label as the assistant's answer,
so the fine-tuned model is evaluated exactly like the base model. The loss is computed only on
the answer tokens: the model learns to classify, not to repeat the instruction.

No torch here: the token-level functions work on plain lists of ids and are tested locally.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ptbr_review_lora.data import LABELS

# Label value that PyTorch's cross-entropy (and Hugging Face models) ignore in the loss.
IGNORE_INDEX = -100


def sample_balanced(pool: pd.DataFrame, per_class: int, seed: int = 42) -> pd.DataFrame:
    """`per_class` random rows of each label, shuffled; the pool index is kept for the report."""
    counts = pool["label"].value_counts()
    short = [label for label in LABELS if counts.get(label, 0) < per_class]
    if short:
        raise ValueError(f"fewer than {per_class} rows for: {short}")
    picked = [pool[pool["label"] == label].sample(per_class, random_state=seed) for label in LABELS]
    return pd.concat(picked).sample(frac=1, random_state=seed)


def nested_subsets(
    df: pd.DataFrame, sizes: Sequence[int], seed: int = 42, stratify: str = "polarity_group"
) -> dict[int, pd.DataFrame]:
    """Nested, stratified subsets for a learning curve (Stage 7, decision D18).

    Each smaller subset is drawn from the next larger one (250 inside 500 inside 1,000 ...), so
    the curve measures the effect of size, not of seeing different reviews. The index is kept.
    """
    from sklearn.model_selection import train_test_split

    ordered = sorted(sizes, reverse=True)
    if ordered[0] > len(df):
        raise ValueError(f"largest size {ordered[0]} exceeds the {len(df)} rows available")
    subsets, current = {}, df
    for size in ordered:
        if size < len(current):
            current, _ = train_test_split(
                current, train_size=size, stratify=current[stratify], random_state=seed
            )
        subsets[size] = current.sort_index()
    return subsets


def build_example(prompt_ids: Sequence[int], answer_ids: Sequence[int]) -> dict[str, list[int]]:
    """Token ids for one training example and the targets of the loss.

    `labels` copies `input_ids` but hides the prompt with IGNORE_INDEX, so only the answer
    (the label word and the end-of-turn token) contributes to the loss. The model shifts
    `labels` by one position internally: each token is predicted from the ones before it.
    """
    if not answer_ids:
        raise ValueError("the answer must have at least one token")
    input_ids = list(prompt_ids) + list(answer_ids)
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)
    return {"input_ids": input_ids, "labels": labels}


def pad_batch(examples: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch: padding gets attention 0 and is ignored by the loss."""
    width = max(len(e["input_ids"]) for e in examples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for e in examples:
        n_pad = width - len(e["input_ids"])
        batch["input_ids"].append(e["input_ids"] + [pad_id] * n_pad)
        batch["attention_mask"].append([1] * len(e["input_ids"]) + [0] * n_pad)
        batch["labels"].append(e["labels"] + [IGNORE_INDEX] * n_pad)
    return batch
