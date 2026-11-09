# ptbr-review-lora

**English** | [Português](README.pt-BR.md)

QLoRA fine-tuning of a small open LLM (1B–3B parameters) to classify Brazilian Portuguese
reviews as **negative**, **neutral** or **positive**, compared against the base model and
simpler baselines.

> **Status:** work in progress — Stage 3 (evaluation and classical baselines) done.
> No LLM has been trained yet. All numbers below are on the dev set; the test set is still unused.

## Goal

Measure whether parameter-efficient fine-tuning (LoRA/QLoRA) of a small LLM improves
sentiment classification in pt-BR over:

1. the same model without fine-tuning (zero-shot / few-shot prompting);
2. classical and encoder-based baselines (defined in Stage 3).

## Data

| Role | Dataset | Labels | Size |
|---|---|---|---|
| Training pool | [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) (Americanas.com, 2018) | 1–5 stars → 3 classes | 118,093 |
| Dev (analysis and tuning) | [RePro](https://huggingface.co/datasets/lucasnil/repro), stratified split | human-annotated polarity | 2,000 |
| Test (final evaluation only) | RePro, stratified split | human-annotated polarity | 8,003 |

- Stars: 1–2 → `negativo`, 3 → `neutro`, 4–5 → `positivo`. RePro: reviews annotated as both
  negative and positive ("mixed") count as `neutro`.
- RePro is a subset of B2W: all 10,003 RePro reviews were removed from the training pool, plus
  130 pool reviews whose text is identical to a RePro text.
- Reviewer profile fields (birth year, gender, state) are dropped on load.
- Both datasets are CC BY-NC-SA 4.0, so models trained here are non-commercial.
- On dev, star-derived labels agree with human labels in **74.4%** of reviews (measured).

Counts per split: [reports/data_summary.json](reports/data_summary.json).
Rationale: decisions D5–D9 in [docs/decisions.md](docs/decisions.md).

## Evaluation

- **Main metric:** macro-F1 over the 3 classes. Secondary: accuracy, per-class F1, confusion
  matrix, and recall of `neutro` split into pure-neutral and mixed reviews.
- **Uncertainty:** 95% percentile bootstrap (1,000 resamples); models are compared with a
  paired bootstrap on the same resamples.
- **Protocol:** every choice is made on dev. The test set is used once, at the end, for all
  frozen models together.

## Repository structure

```
├── src/ptbr_review_lora/   # reusable code (data, prompts, evaluation)
├── scripts/                # command-line entry points (e.g. data preparation)
├── notebooks/              # Colab/Kaggle notebooks (GPU training and inference)
├── reports/                # small result files (counts and metrics only, no text)
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

Download (~55 MB, pinned and checksum-verified) and prepare the data:

```bash
uv run python scripts/prepare_data.py
```

Train the classical baselines and evaluate them on dev (~2 minutes on a laptop CPU):

```bash
uv run python scripts/run_baselines.py
```

GPU training runs on Google Colab or Kaggle; notebooks will be added in Stage 5.

## Results

Final (test set) results are not available yet. Every number in this section comes from an
actual run, with the script or notebook that produced it.

**Classical baselines on dev** (2,000 reviews; from
[reports/baselines_dev.json](reports/baselines_dev.json)):

| Model | Macro-F1 (95% CI) | Accuracy |
|---|---|---|
| Majority class (`positivo`) | 0.195 (0.187–0.202) | 0.413 |
| TF-IDF + logistic regression, text | 0.672 (0.651–0.693) | 0.765 |
| TF-IDF + LR, title + text, balanced class weights | **0.762** (0.741–0.779) | 0.793 |

Adding the title and balancing class weights both helped (paired bootstrap, 95% CI of the
difference above zero). The chosen configuration was selected on dev, so its dev score is
optimistic. Full table: decision D13 in [docs/decisions.md](docs/decisions.md).

## Limitations

To be documented as the project progresses. Known from the start:

- Labels derived from star ratings are noisy: on dev they agree with human labels in 74.4% of reviews.
- The test set's class distribution is artificial (RePro sampled ~2,000 reviews per star rating),
  so accuracy does not reflect real-world traffic; macro-F1 is the main metric.
- Domain: e-commerce product reviews from 2018, not restaurant or food-delivery reviews.
- Only exact text duplicates between training and evaluation were removed; near-duplicates were not.
- TF-IDF hyperparameters were fixed, not tuned, so the classical baseline is reasonable but not optimal.

## Learnings

To be filled in at the end of each stage.

- **Stage 1 — non-ASCII paths break editable installs on Python 3.12 / Windows.**
  With the project under `...\Área de Trabalho\...`, `uv run` failed with
  `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. The editable install writes
  the path to `src/` into a `.pth` file as UTF-8 (`Á` = `C3 81`), but Python 3.12 on Windows
  reads `.pth` files with the locale encoding (cp1252), where `0x81` is undefined.
  Fix: move the project to a path without accents (and outside OneDrive). See D4 in
  [docs/decisions.md](docs/decisions.md).
- **Stage 2 — check data facts in the data itself.** The B2W "dataset" on the Hugging Face Hub
  is only a loading script (no data), so the CSV is downloaded from the original GitHub source.
  The source description suggested a `;` separator, but the file uses `,`: reading the header
  settled it. Removing RePro from the training pool by ID was not enough either: 92 RePro texts
  still appeared verbatim in the pool (same review posted for different products, probably).
- **Stage 3 — a cheap baseline sets the bar.** TF-IDF + logistic regression trained in ~25 s
  reaches 0.762 macro-F1 on dev, and its 79% accuracy is already above the 74% agreement between
  star labels and human labels. Balancing class weights mattered most: it raised recall of
  pure-neutral reviews from 7% to 38%, because neutral is 12% of training but 24% of evaluation.

## License

To be defined (it must be compatible with the dataset and base model licenses).
