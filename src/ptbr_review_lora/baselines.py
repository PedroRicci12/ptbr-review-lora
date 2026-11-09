"""Classical baseline shared by scripts/run_baselines.py and the Colab notebooks."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

# Fixed, not tuned: the point is a reasonable classical baseline, not the best possible one.
TFIDF_PARAMS = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
LR_PARAMS = {"max_iter": 1000, "C": 1.0}


def make_tfidf_lr(class_weight: str | None = None) -> Pipeline:
    """TF-IDF (word 1-2 grams) + logistic regression."""
    return make_pipeline(
        TfidfVectorizer(**TFIDF_PARAMS),
        LogisticRegression(class_weight=class_weight, **LR_PARAMS),
    )
