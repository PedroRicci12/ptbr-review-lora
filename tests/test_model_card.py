import ast
import importlib.util
from pathlib import Path

import pytest

from ptbr_review_lora.data import LABELS
from ptbr_review_lora.model_card import TEST_MODELS, fmt, fmt_diff, fmt_int, render
from ptbr_review_lora.prompts import INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]


def test_number_formats_per_language():
    assert fmt(0.91064, "en") == "0.911"
    assert fmt(0.91064, "pt-BR") == "0,911"
    assert fmt_diff(0.0975, "en") == "+0.098"
    assert fmt_diff(-0.0249, "pt-BR") == "−0,025"
    assert fmt_int(8003, "en") == "8,003"
    assert fmt_int(8003, "pt-BR") == "8.003"


def test_render_fails_on_a_missing_value():
    assert render("F1 $f1", {"f1": "0.9"}) == "F1 0.9"
    with pytest.raises(KeyError):
        render("F1 $f1, n $n", {"f1": "0.9"})


def _constants(path: Path) -> dict:
    """Top-level NAME = literal assignments of a script, read without running it (it needs a GPU)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_usage_example_uses_the_training_prompt_and_labels():
    consts = _constants(ROOT / "model_card" / "usage_example.py")
    assert consts["INSTRUCTION"] == INSTRUCTION
    assert consts["LABELS"] == LABELS
    assert consts["BASE_REVISION"] == "a07cc9a04f16550a088caea529712d1d335b0ac1"


def _build_cards():
    spec = importlib.util.spec_from_file_location("build_model_card", ROOT / "scripts" / "build_model_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build()


def test_cards_render_from_the_committed_reports():
    cards = _build_cards()
    assert set(cards) == {"en", "pt-BR"}
    assert "0.911" in cards["en"] and "0,911" in cards["pt-BR"]
    for lang, text in cards.items():
        assert "$" not in text, lang  # every placeholder was filled
        for names in TEST_MODELS.values():
            assert names[lang] in text
    assert cards["en"].startswith("---\n")  # Hugging Face metadata block


def test_committed_cards_are_up_to_date():
    cards = _build_cards()
    assert (ROOT / "model_card" / "README.md").read_text(encoding="utf-8") == cards["en"]
    assert (ROOT / "model_card" / "README.pt-BR.md").read_text(encoding="utf-8") == cards["pt-BR"]
