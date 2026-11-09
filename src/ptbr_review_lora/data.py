"""Download, clean and split the B2W-Reviews01 (training) and RePro (evaluation) datasets."""

from __future__ import annotations

import ast
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ("negativo", "neutro", "positivo")

# Sources are pinned to a commit so every run downloads exactly the same bytes.
B2W_URL = (
    "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/"
    "4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv"
)
B2W_GIT_BLOB_SHA = "b7c9e9f780ad810693520a9ebfff01188576e363"
REPRO_URL = (
    "https://huggingface.co/datasets/lucasnil/repro/resolve/"
    "cb93d9cc1183d4a594317a5188d325bcaa04cda4/RePro.csv"
)

# Reviewer profile data: dropped on load, never written to disk after processing.
PERSONAL_COLUMNS = ("reviewer_birth_year", "reviewer_gender", "reviewer_state")

# Together these identify a review in both datasets (RePro is a subset of B2W).
KEY_COLUMNS = ("reviewer_id", "product_id", "submission_date", "review_text")

# RePro polarity is per aspect; "misto" = both NEGATIVO and POSITIVO (decision D6).
GROUP_TO_LABEL = {
    "negativo": "negativo",
    "positivo": "positivo",
    "neutro": "neutro",
    "misto": "neutro",
}


def git_blob_sha(data: bytes) -> str:
    """Hash bytes the way Git does, so a download can be checked against GitHub's file SHA."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download(url: str, dest: Path, expected_git_sha: str | None = None) -> Path:
    """Download `url` to `dest` unless it already exists, optionally verifying its Git blob SHA."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            data = response.read()
        if expected_git_sha is not None and git_blob_sha(data) != expected_git_sha:
            raise ValueError(f"checksum mismatch for {url}")
        dest.write_bytes(data)
    return dest


def stars_to_label(rating: int) -> str:
    """Map a 1-5 star rating to a sentiment label (decision D1)."""
    if rating not in range(1, 6):
        raise ValueError(f"rating must be between 1 and 5, got {rating!r}")
    if rating <= 2:
        return "negativo"
    if rating == 3:
        return "neutro"
    return "positivo"


def polarity_group(polarity: str) -> str:
    """Map a RePro polarity list, stored as a string like "['NEGATIVO', 'POSITIVO']", to a group."""
    values = set(ast.literal_eval(polarity))
    if values == {"NEGATIVO"}:
        return "negativo"
    if values == {"POSITIVO"}:
        return "positivo"
    if values == {"NEUTRO"}:
        return "neutro"
    if values == {"NEGATIVO", "POSITIVO"}:
        return "misto"
    raise ValueError(f"unexpected polarity: {polarity!r}")


def drop_personal_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in PERSONAL_COLUMNS if c in df.columns])


def prepare_b2w(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop personal data, reviews without text and exact duplicates; add the star-based label."""
    df = drop_personal_columns(raw)
    df = df[df["review_text"].notna() & (df["review_text"].str.strip() != "")]
    df = df.drop_duplicates(subset=list(KEY_COLUMNS))
    df = df.assign(label=df["overall_rating"].map(stars_to_label))
    return df.reset_index(drop=True)


def prepare_repro(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop personal data; add the polarity group and the 3-class human label."""
    df = drop_personal_columns(raw)
    groups = df["polarity"].map(polarity_group)
    return df.assign(polarity_group=groups, label=groups.map(GROUP_TO_LABEL)).reset_index(drop=True)


def remove_overlap(b2w: pd.DataFrame, repro: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove from B2W every review that is also in RePro. Returns (clean B2W, rows removed)."""
    keys = repro[list(KEY_COLUMNS)].drop_duplicates()
    merged = b2w.merge(keys, on=list(KEY_COLUMNS), how="left", indicator=True)
    in_repro = merged["_merge"] == "both"
    clean = merged.loc[~in_repro].drop(columns="_merge").reset_index(drop=True)
    return clean, int(in_repro.sum())


def normalize_text(texts: pd.Series) -> pd.Series:
    return texts.str.strip().str.lower()


def remove_text_duplicates(b2w: pd.DataFrame, repro: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove from B2W every review whose text equals a RePro text (ignoring case and edge spaces).

    Catches the same review posted for different products, which `remove_overlap` misses
    because the reviewer/product keys differ (decision D9). Returns (clean B2W, rows removed).
    """
    repro_texts = set(normalize_text(repro["review_text"]))
    duplicated = normalize_text(b2w["review_text"]).isin(repro_texts)
    return b2w.loc[~duplicated].reset_index(drop=True), int(duplicated.sum())


def model_input(df: pd.DataFrame, use_title: bool) -> pd.Series:
    """Text given to a model: the review text, optionally preceded by its title ("title. text")."""
    if not use_title:
        return df["review_text"]
    title = df["review_title"].fillna("").str.strip()
    return (title + ". " + df["review_text"]).where(title != "", df["review_text"])


def split_repro(
    repro: pd.DataFrame, dev_size: int = 2000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split RePro into dev and test, stratified by the 4 polarity groups (decision D7)."""
    dev, test = train_test_split(
        repro, train_size=dev_size, stratify=repro["polarity_group"], random_state=seed
    )
    return dev.reset_index(drop=True), test.reset_index(drop=True)
