import pandas as pd
import pytest

from ptbr_review_lora.data import LABELS
from ptbr_review_lora.prompts import (
    INSTRUCTION,
    build_messages,
    choose_label,
    parse_label,
    sample_few_shot,
)


def test_zero_shot_messages():
    messages = build_messages("  Chegou quebrado. ")
    assert messages == [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": "Avaliação: Chegou quebrado."},
    ]


def test_few_shot_messages_alternate_user_and_assistant():
    examples = [("Ótimo produto", "positivo"), ("Não funciona", "negativo")]
    messages = build_messages("Razoável", examples)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in messages if m["role"] == "assistant"] == ["positivo", "negativo"]
    assert messages[-1]["content"] == "Avaliação: Razoável"


def test_few_shot_rejects_unknown_label():
    with pytest.raises(ValueError):
        build_messages("x", [("y", "bom")])


def test_instruction_defines_mixed_reviews_as_neutral():
    assert "positivos e negativos" in INSTRUCTION
    assert all(label in INSTRUCTION for label in LABELS)


def _pool():
    rows = []
    for label in LABELS:
        for i in range(5):
            rows.append({"review_title": f"t{i}", "review_text": f"{label} curto {i}", "label": label})
        rows.append({"review_title": "", "review_text": label + " " + "longo " * 100, "label": label})
    return pd.DataFrame(rows)


def test_sample_few_shot_is_balanced_short_and_deterministic():
    pool = _pool()
    examples = sample_few_shot(pool, per_class=2, seed=7)
    assert sorted(examples["label"]) == sorted(LABELS * 2)
    assert (examples["text"].str.len() <= 280).all()
    assert examples.index.is_unique and len(examples) == 6
    assert (pool.loc[examples.index, "label"] == examples["label"]).all()  # index points to the pool row
    pd.testing.assert_frame_equal(examples, sample_few_shot(pool, per_class=2, seed=7))


def test_sample_few_shot_uses_title():
    examples = sample_few_shot(_pool(), per_class=1, seed=0)
    assert all(text.startswith("t") and ". " in text for text in examples["text"])


def test_sample_few_shot_feeds_build_messages():
    examples = sample_few_shot(_pool(), per_class=1, seed=0)
    messages = build_messages("x", list(zip(examples["text"], examples["label"])))
    assert len(messages) == 1 + 2 * 3 + 1


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("negativo", "negativo"),
        ("Neutro.", "neutro"),
        ("  POSITIVO\n", "positivo"),
        ("A avaliação é negativa", "negativo"),
        ("neutra", "neutro"),
        ("positivo e negativo", None),
        ("não sei", None),
        ("", None),
    ],
)
def test_parse_label(answer, expected):
    assert parse_label(answer) == expected


def test_choose_label_picks_highest_score():
    assert choose_label({"negativo": -3.2, "neutro": -0.4, "positivo": -1.1}) == "neutro"


def test_choose_label_requires_all_labels():
    with pytest.raises(ValueError):
        choose_label({"negativo": -1.0, "positivo": -2.0})
