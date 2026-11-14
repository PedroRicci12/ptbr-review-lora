"""Model card for the published adapter (Stage 8), built from the committed reports.

Every number in the card comes from a JSON report in `reports/`, so the card cannot drift from
what was measured. The templates live in `model_card/` (one per language) and use `$name`
placeholders (`string.Template`), so the braces of the code example need no escaping.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from string import Template

LANGS = ("en", "pt-BR")

# Test-set models, in the order of the table (weakest first), with their names per language.
TEST_MODELS = {
    "majority": {"en": "Majority class (`positivo`)", "pt-BR": "Classe majoritária (`positivo`)"},
    "tfidf": {"en": "TF-IDF + logistic regression", "pt-BR": "TF-IDF + regressão logística"},
    "bertimbau": {"en": "BERTimbau base, fine-tuned (star labels)",
                  "pt-BR": "BERTimbau base ajustado (rótulos de estrelas)"},
    "qlora_stars": {"en": "SmolLM3-3B + QLoRA, 15,000 star labels",
                    "pt-BR": "SmolLM3-3B + QLoRA, 15.000 rótulos de estrelas"},
    "zero_shot": {"en": "SmolLM3-3B zero-shot (base model)",
                  "pt-BR": "SmolLM3-3B zero-shot (modelo base)"},
    "qlora_human": {"en": "**This adapter** (QLoRA, 1,500 human labels)",
                    "pt-BR": "**Este adapter** (QLoRA, 1.500 rótulos humanos)"},
}

VERDICTS = {
    "B better": {"en": "better", "pt-BR": "melhor"},
    "B worse": {"en": "worse", "pt-BR": "pior"},
    "inconclusive": {"en": "inconclusive", "pt-BR": "inconclusivo"},
}


def fmt(x: float, lang: str, digits: int = 3) -> str:
    """0.9106 -> "0.911" (en) or "0,911" (pt-BR)."""
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "pt-BR" else text


def fmt_diff(x: float, lang: str, digits: int = 3) -> str:
    """Signed difference with a real minus sign: +0.098 / −0.025."""
    sign = "+" if x >= 0 else "−"
    return sign + fmt(abs(x), lang, digits)


def fmt_int(n: int, lang: str) -> str:
    """1498 -> "1,498" (en) or "1.498" (pt-BR)."""
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "pt-BR" else text


def fmt_ci(low: float, high: float, lang: str) -> str:
    return f"{fmt(low, lang)}–{fmt(high, lang)}"


def test_table(results: Mapping, lang: str) -> str:
    """Markdown rows of the test-set table: model | macro-F1 (95% CI) | accuracy | F1 neutro."""
    rows = []
    for key, names in TEST_MODELS.items():
        r = results[key]
        f1 = fmt(r["macro_f1"], lang)
        if key == "qlora_human":
            f1 = f"**{f1}**"
        rows.append(
            f"| {names[lang]} | {f1} ({fmt_ci(*r['macro_f1_ci95'], lang)}) | "
            f"{fmt(r['accuracy'], lang)} | {fmt(r['f1_per_class']['neutro'], lang)} |"
        )
    return "\n".join(rows)


def hypotheses_table(primary: Mapping, lang: str) -> str:
    """Markdown rows of the pre-registered comparisons H1-H3."""
    questions = {
        "H1": {"en": "this adapter vs zero-shot", "pt-BR": "este adapter vs zero-shot"},
        "H2": {"en": "QLoRA star labels vs zero-shot", "pt-BR": "QLoRA estrelas vs zero-shot"},
        "H3": {"en": "this adapter vs QLoRA star labels", "pt-BR": "este adapter vs QLoRA estrelas"},
    }
    word = {"en": "to", "pt-BR": "a"}[lang]
    rows = []
    for h in ("H1", "H2", "H3"):
        c = primary[h]
        rows.append(
            f"| {h} | {questions[h][lang]} | {fmt_diff(c['diff'], lang)} "
            f"({fmt_diff(c['low'], lang)} {word} {fmt_diff(c['high'], lang)}) | "
            f"{VERDICTS[c['verdict']][lang]} |"
        )
    return "\n".join(rows)


def learning_curve_table(human_select: Mapping, curve: Mapping, lang: str) -> str:
    """Markdown rows: human labels | macro-F1 on the dev selection set (95% CI)."""
    points = [(0, curve["results"]["zero_shot"])]
    for size in (250, 500, 1000):
        points.append((size, curve["results"][f"qlora_human_{size}_seed42"]))
    points.append((1500, human_select["results"]["smollm3_qlora_human"]))
    return "\n".join(
        f"| {fmt_int(n, lang)} | {fmt(r['macro_f1'], lang)} ({fmt_ci(*r['macro_f1_ci95'], lang)}) |"
        for n, r in points
    )


def card_values(
    test: Mapping, human_select: Mapping, curve: Mapping, seeds: Mapping, lang: str
) -> dict[str, str]:
    """All placeholder values of one card, formatted for `lang`."""
    if lang not in LANGS:
        raise ValueError(f"unknown language: {lang!r}")
    res = test["results"]
    primary = test["primary_comparisons"]
    sel = human_select["results"]["smollm3_qlora_human"]
    seed_f1 = seeds["summary"]["macro_f1"]
    cfg = human_select["config"]
    env = human_select["environment"]
    return {
        "test_n": fmt_int(test["n"], lang),
        "test_f1": fmt(res["qlora_human"]["macro_f1"], lang),
        "test_f1_ci": fmt_ci(*res["qlora_human"]["macro_f1_ci95"], lang),
        "test_accuracy": fmt(res["qlora_human"]["accuracy"], lang),
        "test_f1_raw": f"{res['qlora_human']['macro_f1']:.4f}",
        "test_accuracy_raw": f"{res['qlora_human']['accuracy']:.4f}",
        "zero_shot_f1": fmt(res["zero_shot"]["macro_f1"], lang),
        "h1_diff": fmt_diff(primary["H1"]["diff"], lang),
        "test_table": test_table(res, lang),
        "hypotheses_table": hypotheses_table(primary, lang),
        "family_confidence": fmt(100 * primary["H1"]["confidence"], lang, 1),
        "protocol_commit": test["protocol_commit"][:7],
        "select_f1": fmt(sel["macro_f1"], lang),
        "select_f1_ci": fmt_ci(*sel["macro_f1_ci95"], lang),
        "learning_curve_table": learning_curve_table(human_select, curve, lang),
        "seed_f1s": " / ".join(fmt(x, lang) for x in seed_f1),
        "seed_mean": fmt(statistics.mean(seed_f1), lang),
        "seed_std": fmt(statistics.stdev(seed_f1), lang),
        "training_examples": fmt_int(human_select["training_examples"], lang),
        "dropped_long": fmt_int(human_select["dropped_long_examples"], lang),
        "epochs": str(cfg["epochs"]),
        "total_steps": str(human_select["total_steps"]),
        "best_step": str(human_select["best_step"]),
        "train_minutes": str(round(human_select["train_seconds"] / 60)),
        "trainable_params": fmt(human_select["trainable_parameters"] / 1e6, lang, 1),
        "lora_r": str(cfg["lora_r"]),
        "lora_alpha": str(cfg["lora_alpha"]),
        "lora_dropout": fmt(cfg["lora_dropout"], lang, 2),
        "learning_rate": f"{cfg['learning_rate']:g}",
        "effective_batch": str(cfg["micro_batch"] * cfg["grad_accum"]),
        "max_train_tokens": str(cfg["max_train_tokens"]),
        "gpu": env["gpu"],
        "versions": (
            f"torch {env['torch']}, transformers {env['transformers']}, "
            f"peft {env['peft']}, bitsandbytes {env['bitsandbytes']}"
        ),
        "model_revision": human_select["model_revision"],
        "training_commit": human_select["repo_commit"][:7],
    }


def render(template: str, values: Mapping[str, str]) -> str:
    """Fill the `$name` placeholders; a missing value raises KeyError instead of passing silently."""
    return Template(template).substitute(values)
