"""Draw the README / article figures from the committed reports (English and pt-BR versions).

    uv run --group figures python scripts/make_figures.py

Writes `figures/<lang>/test_macro_f1.png` and `figures/<lang>/learning_curve.png`. Every value
is read from `reports/`; nothing is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
HIGHLIGHT = "#2a78d6"   # this project's adapter
OTHER = "#9a9893"       # every other model

TEXT = {
    "en": {
        "models": {
            "majority": "Majority class",
            "tfidf": "TF-IDF + logistic regression",
            "bertimbau": "BERTimbau, fine-tuned (star labels)",
            "qlora_stars": "SmolLM3-3B + QLoRA, 15,000 star labels",
            "zero_shot": "SmolLM3-3B zero-shot",
            "qlora_human": "SmolLM3-3B + QLoRA, 1,500 human labels",
        },
        "test_title": "Test set: macro-F1 with 95% CI",
        "test_note": "{n} human-labeled reviews, scored once",
        "test_ci": "Macro-F1 (95% CIs of about ±0.01, hidden behind the dots)",
        "curve_title": "How many human labels? (dev selection set, 500 reviews)",
        "curve_x": "Human-labeled training reviews (0 = zero-shot)",
        "curve_y": "Macro-F1",
        "band": "95% CI",
        "seeds": "3 training seeds at 1,500",
        "stars": "QLoRA, 15,000 star labels: {v}",
        "thousands": ",",
        "decimal": ".",
    },
    "pt-BR": {
        "models": {
            "majority": "Classe majoritária",
            "tfidf": "TF-IDF + regressão logística",
            "bertimbau": "BERTimbau ajustado (rótulos de estrelas)",
            "qlora_stars": "SmolLM3-3B + QLoRA, 15.000 rótulos de estrelas",
            "zero_shot": "SmolLM3-3B zero-shot",
            "qlora_human": "SmolLM3-3B + QLoRA, 1.500 rótulos humanos",
        },
        "test_title": "Teste: macro-F1 com IC de 95%",
        "test_note": "{n} avaliações rotuladas por humanos, avaliadas uma única vez",
        "test_ci": "Macro-F1 (ICs de 95% de cerca de ±0,01, escondidos atrás dos pontos)",
        "curve_title": "Quantos rótulos humanos? (seleção do dev, 500 avaliações)",
        "curve_x": "Avaliações de treino rotuladas por humanos (0 = zero-shot)",
        "curve_y": "Macro-F1",
        "band": "IC de 95%",
        "seeds": "3 seeds de treino com 1.500",
        "stars": "QLoRA, 15.000 rótulos de estrelas: {v}",
        "thousands": ".",
        "decimal": ",",
    },
}


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def num(x: float, t: dict, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", t["decimal"])


def integer(n: int, t: dict) -> str:
    return f"{n:,}".replace(",", t["thousands"])


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def test_figure(test: dict, t: dict, path: Path) -> None:
    order = ["majority", "tfidf", "bertimbau", "qlora_stars", "zero_shot", "qlora_human"]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style(ax)
    for y, key in enumerate(order):
        r = test["results"][key]
        f1, (low, high) = r["macro_f1"], r["macro_f1_ci95"]
        color = HIGHLIGHT if key == "qlora_human" else OTHER
        ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(f1, y, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        weight = "bold" if key == "qlora_human" else "normal"
        ax.text(high + 0.012, y, num(f1, t), va="center", fontsize=9, color=INK, fontweight=weight)
    ax.set_yticks(range(len(order)), [t["models"][k] for k in order], color=INK, fontsize=9)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.xaxis.set_major_formatter(lambda v, _: num(v, t, 1))
    ax.set_xlabel(t["test_ci"], fontsize=9, color=INK_2)
    # Title and subtitle belong to the figure (left edge), so long subtitles do not squeeze the plot.
    fig.text(0.02, 0.95, t["test_title"], fontsize=11, color=INK, va="top")
    fig.text(0.02, 0.88, t["test_note"].format(n=integer(test["n"], t)), fontsize=9, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def curve_figure(human: dict, curve: dict, seeds: dict, stars: dict, t: dict, path: Path) -> None:
    points = [(0, curve["results"]["zero_shot"])]
    points += [(n, curve["results"][f"qlora_human_{n}_seed42"]) for n in (250, 500, 1000)]
    points.append((1500, human["results"]["smollm3_qlora_human"]))
    xs = [n for n, _ in points]
    f1 = [r["macro_f1"] for _, r in points]
    low = [r["macro_f1_ci95"][0] for _, r in points]
    high = [r["macro_f1_ci95"][1] for _, r in points]
    stars_f1 = stars["results"]["smollm3_qlora_stars"]["macro_f1"]
    seed_f1 = seeds["summary"]["macro_f1"]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.fill_between(xs, low, high, color=HIGHLIGHT, alpha=0.15, linewidth=0, label=t["band"])
    ax.plot(xs, f1, color=HIGHLIGHT, linewidth=2)
    ax.plot(xs, f1, "o", color=HIGHLIGHT, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for x, y in zip(xs, f1):
        ax.text(x, y + 0.012, num(y, t), ha="center", fontsize=9, color=INK)
    # Seed results sit just right of the 1,500 point so they do not hide its label.
    ax.plot([1570] * len(seed_f1), seed_f1, "_", color=INK_2, markersize=10, markeredgewidth=1.5,
            label=t["seeds"])
    ax.axhline(stars_f1, color=OTHER, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(1600, stars_f1 + 0.005, t["stars"].format(v=num(stars_f1, t)), ha="right",
            fontsize=9, color=INK_2)
    ax.set_xticks(xs, [integer(x, t) for x in xs])
    ax.set_xlim(-80, 1620)
    ax.set_ylim(0.75, 0.96)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    ax.yaxis.set_major_formatter(lambda v, _: num(v, t, 2))
    ax.set_xlabel(t["curve_x"], fontsize=9, color=INK_2)
    ax.set_ylabel(t["curve_y"], fontsize=9, color=INK_2)
    ax.set_title(t["curve_title"], loc="left", fontsize=11, color=INK)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    test = load("test_results.json")
    human = load("smollm3_qlora_human_select.json")
    stars = load("smollm3_qlora_stars_select.json")
    curve = load("ablation_learning_curve_select.json")
    seeds = load("ablation_seeds_select.json")
    for lang, t in TEXT.items():
        folder = OUT / lang
        folder.mkdir(parents=True, exist_ok=True)
        test_figure(test, t, folder / "test_macro_f1.png")
        curve_figure(human, curve, seeds, stars, t, folder / "learning_curve.png")
        print("wrote", folder.relative_to(ROOT))


if __name__ == "__main__":
    main()
