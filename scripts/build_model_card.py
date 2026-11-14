"""Build the Hugging Face model card (English + pt-BR) from the committed reports.

Writes `model_card/README.md` and `model_card/README.pt-BR.md`. Run it again whenever a report or a
template changes; `scripts/publish_hf.py` uploads these two files next to the adapter.

    uv run python scripts/build_model_card.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ptbr_review_lora.model_card import card_values, render

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CARD_DIR = ROOT / "model_card"
OUTPUTS = {"en": "README.md", "pt-BR": "README.pt-BR.md"}


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def build() -> dict[str, str]:
    test = load("test_results.json")
    human_select = load("smollm3_qlora_human_select.json")
    curve = load("ablation_learning_curve_select.json")
    seeds = load("ablation_seeds_select.json")
    usage = (CARD_DIR / "usage_example.py").read_text(encoding="utf-8").rstrip("\n")
    cards = {}
    for lang in OUTPUTS:
        template = (CARD_DIR / f"template.{lang}.md").read_text(encoding="utf-8")
        values = {**card_values(test, human_select, curve, seeds, lang), "usage": usage}
        cards[lang] = render(template, values)
    return cards


def main() -> None:
    for lang, text in build().items():
        path = CARD_DIR / OUTPUTS[lang]
        path.write_text(text, encoding="utf-8", newline="\n")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
