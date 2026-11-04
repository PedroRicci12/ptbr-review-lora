# ptbr-review-lora

**English** | [Português](README.pt-BR.md)

QLoRA fine-tuning of a small open LLM (1B–3B parameters) to classify Brazilian Portuguese
reviews as **negative**, **neutral** or **positive**, compared against the base model and
simpler baselines.

> **Status:** work in progress — Stage 1 (project setup). No model has been trained yet
> and there are no results.

## Goal

Measure whether parameter-efficient fine-tuning (LoRA/QLoRA) of a small LLM improves
sentiment classification in pt-BR over:

1. the same model without fine-tuning (zero-shot / few-shot prompting);
2. classical and encoder-based baselines (defined in Stage 3).

## Repository structure

```
├── src/ptbr_review_lora/   # reusable code (data, prompts, evaluation)
├── notebooks/              # Colab/Kaggle notebooks (GPU training and inference)
├── tests/                  # pytest tests
└── docs/decisions.md       # design decisions and alternatives considered (pt-BR)
```

## Installation

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd ptbr-review-lora
uv sync
cp .env.example .env   # then add your Hugging Face token
```

## Usage

Run the tests:

```bash
uv run pytest
```

GPU training runs on Google Colab or Kaggle; notebooks will be added in Stage 5.

## Results

Not available yet. Every number in this section will come from an actual run, with the
notebook that produced it.

## Limitations

To be documented as the project progresses. Known from the start:

- Labels derived from star ratings are noisy (the text and the rating may disagree).

## Learnings

To be filled in at the end of each stage.

- **Stage 1 — non-ASCII paths break editable installs on Python 3.12 / Windows.**
  With the project under `...\Área de Trabalho\...`, `uv run` failed with
  `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. The editable install writes
  the path to `src/` into a `.pth` file as UTF-8 (`Á` = `C3 81`), but Python 3.12 on Windows
  reads `.pth` files with the locale encoding (cp1252), where `0x81` is undefined.
  Fix: move the project to a path without accents (and outside OneDrive). See D4 in
  [docs/decisions.md](docs/decisions.md).

## License

To be defined (it must be compatible with the dataset and base model licenses).
