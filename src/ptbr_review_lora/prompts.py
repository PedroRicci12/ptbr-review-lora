"""Prompts and label extraction for classifying reviews with an instruction-tuned LLM (Stage 4).

Two ways to turn the model into a classifier (decision D15):

- label scoring (main): the model scores each of the 3 labels as the answer and the most
  likely one wins, so every prediction is a valid label;
- generation (secondary, dev only): the model writes an answer and `parse_label` maps it to a
  label, or to None when the answer does not follow the format.

This module only builds prompts and interprets answers; it does not depend on torch, so it is
tested locally. The model calls live in the GPU notebooks.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

import pandas as pd

from ptbr_review_lora.data import LABELS, model_input

# The definition of `neutro` includes mixed reviews: that is the task (decision D6), not a hint.
INSTRUCTION = (
    "Você classifica o sentimento de avaliações de produtos de uma loja online brasileira.\n"
    "Responda com uma única palavra, em minúsculas e sem pontuação:\n"
    "- negativo: o cliente está insatisfeito com o produto ou com a compra;\n"
    "- neutro: a avaliação não é positiva nem negativa, ou tem pontos positivos e negativos "
    "ao mesmo tempo;\n"
    "- positivo: o cliente está satisfeito com o produto ou com a compra."
)

# Few-shot examples longer than this are skipped, so 6 examples do not blow up the prompt.
MAX_EXAMPLE_CHARS = 280

# Word stems accepted by `parse_label`: "negativa", "neutra", "positiva" also count.
_STEMS = {"negativ": "negativo", "neutr": "neutro", "positiv": "positivo"}


def user_message(review: str) -> str:
    return f"Avaliação: {review.strip()}"


def build_messages(review: str, examples: Sequence[tuple[str, str]] = ()) -> list[dict[str, str]]:
    """Chat messages for one review: instruction, optional solved examples, then the review.

    Each example becomes a user turn (the review) followed by an assistant turn (its label),
    so the model sees the exact answer format before the real question.
    """
    messages = [{"role": "system", "content": INSTRUCTION}]
    for text, label in examples:
        if label not in LABELS:
            raise ValueError(f"unknown label in few-shot example: {label!r}")
        messages.append({"role": "user", "content": user_message(text)})
        messages.append({"role": "assistant", "content": label})
    messages.append({"role": "user", "content": user_message(review)})
    return messages


def sample_few_shot(
    pool: pd.DataFrame,
    per_class: int = 2,
    seed: int = 42,
    use_title: bool = True,
    max_chars: int = MAX_EXAMPLE_CHARS,
) -> pd.DataFrame:
    """Draw `per_class` short examples of each label from the training pool, in shuffled order.

    The examples come from the B2W pool (star labels), never from dev or test. The order is
    shuffled so the prompt does not always end with the same class. Returns columns `text` and
    `label`, keeping the pool index so reports can record which rows were used (not their text).
    """
    texts = model_input(pool, use_title)
    short = pd.DataFrame({"text": texts, "label": pool["label"]})[texts.str.len() <= max_chars]
    picked = [
        short[short["label"] == label].sample(per_class, random_state=seed) for label in LABELS
    ]
    return pd.concat(picked).sample(frac=1, random_state=seed)


def _normalize(text: str) -> str:
    """Lowercase and strip accents, so "Negativo." and "negativo" compare equal."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def parse_label(answer: str) -> str | None:
    """Map a generated answer to a label, or None if it names no label or more than one.

    "Negativo." -> "negativo"; "A avaliação é neutra" -> "neutro";
    "positivo e negativo" -> None (ambiguous); "não sei" -> None.
    """
    words = re.findall(r"[a-z]+", _normalize(answer))
    found = {label for word in words for stem, label in _STEMS.items() if word.startswith(stem)}
    return found.pop() if len(found) == 1 else None


def choose_label(scores: Mapping[str, float]) -> str:
    """Pick the label with the highest score (e.g. the log-probability of the label as answer)."""
    missing = set(LABELS) - set(scores)
    if missing:
        raise ValueError(f"missing scores for: {sorted(missing)}")
    return max(LABELS, key=lambda label: scores[label])
