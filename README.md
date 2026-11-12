# ptbr-review-lora

**English** | [Português](README.pt-BR.md)

QLoRA fine-tuning of a small open LLM (1B–3B parameters) to classify Brazilian Portuguese
reviews as **negative**, **neutral** or **positive**, compared against the base model and
simpler baselines.

> **Status:** work in progress — Stage 6 (single test-set evaluation) done.
> Final results are on the test set; the model is not published yet.

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

Fine-tune the BERTimbau baseline (GPU) with
[notebooks/03b_bertimbau.ipynb](notebooks/03b_bertimbau.ipynb), on either platform:

- **Google Colab:** [open in Colab](https://colab.research.google.com/github/PedroRicci12/ptbr-review-lora/blob/main/notebooks/03b_bertimbau.ipynb),
  select a T4 GPU and run all cells; the browser downloads `bertimbau_results.zip`.
- **Kaggle:** import the notebook from GitHub (*File → Import Notebook*), set *Accelerator →
  GPU T4 x2* and *Internet → on*, then *Save Version → Save & Run All*; the zip is in the
  version's *Output* tab. A phone-verified Kaggle account is required for GPUs.

Unzip the results at the repository root: the files land in `reports/` and `predictions/`.

Evaluate the base LLM (SmolLM3-3B, zero-shot and 6-shot, GPU) with
[notebooks/04_base_llm.ipynb](notebooks/04_base_llm.ipynb), on Colab
([open in Colab](https://colab.research.google.com/github/PedroRicci12/ptbr-review-lora/blob/main/notebooks/04_base_llm.ipynb))
or Kaggle, the same way; it produces `smollm3_base_results.zip`.

Fine-tune SmolLM3-3B with QLoRA (GPU) with [notebooks/05_qlora.ipynb](notebooks/05_qlora.ipynb),
once per training set: set `TRAINING_SET = "stars"` or `"human"` in the configuration cell.
Each run produces `qlora_<set>_results.zip` (unzip at the repository root) and
`qlora_<set>_adapter.zip` (LoRA weights; keep it outside the repository).

Run the ablations (GPU) with [notebooks/07_ablations.ipynb](notebooks/07_ablations.ipynb), once per
experiment: `EXPERIMENT = "learning_curve"`, `"seeds"` or `"few_shot"`; each run produces
`ablation_<experiment>_results.zip`. They use only the dev selection set, never the test set.

## Results

Every number in this section comes from an actual run, with the script or notebook that produced it.

### Final results on the test set

8,003 human-labeled RePro reviews, never used for any choice and evaluated once, with comparisons
pre-registered in [src/ptbr_review_lora/protocol.py](src/ptbr_review_lora/protocol.py) before the
run (decision D17); from [reports/test_results.json](reports/test_results.json):

| Model | Macro-F1 (95% CI) | Accuracy | F1 `neutro` |
|---|---|---|---|
| Majority class (`positivo`) | 0.195 (0.191–0.198) | 0.413 | 0.000 |
| TF-IDF + LR, title + text, balanced | 0.757 (0.747–0.766) | 0.785 | 0.558 |
| BERTimbau base, fine-tuned (star labels) | 0.784 (0.775–0.793) | 0.803 | 0.618 |
| SmolLM3-3B + QLoRA, 15,000 star labels | 0.789 (0.780–0.798) | 0.810 | 0.618 |
| SmolLM3-3B zero-shot (no fine-tuning) | 0.814 (0.804–0.823) | 0.850 | 0.634 |
| **SmolLM3-3B + QLoRA, 1,500 human labels** | **0.911** (0.904–0.917) | **0.920** | **0.846** |

Pre-registered comparisons (paired bootstrap, 10,000 resamples, Bonferroni-corrected 98.3% CIs):

| | Comparison | Difference (98.3% CI) | Verdict |
|---|---|---|---|
| H1 | QLoRA human labels vs zero-shot | +0.098 (+0.086 to +0.109) | better |
| H2 | QLoRA star labels vs zero-shot | −0.025 (−0.037 to −0.013) | **worse** |
| H3 | QLoRA human labels vs QLoRA star labels | +0.122 (+0.111 to +0.134) | better |

- Fine-tuning helped only with good labels: 1,500 human labels added ~10 points over zero-shot,
  while 15,000 star labels made the model slightly worse than using no fine-tuning at all.
- The test confirms dev for every model (differences of at most 0.012 macro-F1); on dev, H2 was
  inconclusive, and the larger test set settles it as a small loss.
- The human-label model was trained on RePro reviews (a different split from test), so it learned the
  same annotation scheme the test uses; the baselines learned from star ratings.

### Development results

**Results on dev** (2,000 reviews; from
[reports/baselines_dev.json](reports/baselines_dev.json),
[reports/bertimbau_dev.json](reports/bertimbau_dev.json) and
[reports/smollm3_base_dev.json](reports/smollm3_base_dev.json)):

| Model | Macro-F1 (95% CI) | Accuracy | F1 `neutro` |
|---|---|---|---|
| Majority class (`positivo`) | 0.195 (0.187–0.202) | 0.413 | 0.000 |
| TF-IDF + logistic regression, text | 0.672 (0.651–0.693) | 0.765 | — |
| TF-IDF + LR, title + text, balanced class weights | 0.762 (0.741–0.779) | 0.793 | 0.560 |
| BERTimbau base, fine-tuned (1 epoch, balanced loss) | 0.787 (0.768–0.804) | 0.807 | 0.621 |
| SmolLM3-3B, no fine-tuning, 6-shot | 0.760 (0.740–0.778) | 0.801 | 0.535 |
| SmolLM3-3B, no fine-tuning, zero-shot | **0.815** (0.795–0.833) | 0.851 | 0.638 |

- The base LLM in zero-shot beats fine-tuned BERTimbau by +0.028 macro-F1 (paired 95% CI
  +0.007 to +0.050). Adding 6 examples to the prompt made it worse (−0.055), so zero-shot was kept.
- Label scoring and free generation agree on 99.85% of reviews; 0% of generated answers were invalid.
- Caveat: B2W and RePro are public, so they may be in SmolLM3's pre-training data
  (not verified); that would inflate the zero-shot score.

**QLoRA fine-tuning, on the 500-review selection part of dev** (the other 1,500 dev reviews are
training data for the human-label run; from
[reports/smollm3_qlora_human_select.json](reports/smollm3_qlora_human_select.json)):

| Model (same 500 reviews) | Macro-F1 (95% CI) | F1 `neutro` |
|---|---|---|
| TF-IDF + LR, best | 0.762 (0.724–0.796) | 0.554 |
| BERTimbau, fine-tuned | 0.792 (0.756–0.827) | 0.620 |
| SmolLM3-3B zero-shot (fp16) | 0.811 (0.775–0.848) | 0.627 |
| SmolLM3-3B in 4 bits, before training | 0.706 (0.666–0.748) | 0.355 |
| SmolLM3-3B + QLoRA, 15,000 star labels | 0.788 (0.752–0.823) | 0.603 |
| **SmolLM3-3B + QLoRA, 1,500 human labels** | **0.899** (0.870–0.927) | **0.822** |

- **1,500 human labels beat 15,000 star labels by +0.112 macro-F1** (paired 95% CI +0.070 to
  +0.151), with a tenth of the examples and under a third of the training time.
- QLoRA on star labels did not beat zero-shot (−0.023, CI −0.059 to +0.013) and tied with
  BERTimbau, agreeing with it on 95.4% of reviews: with the same noisy labels, a 110M encoder and
  a 3B LLM learn almost the same thing.
- QLoRA on human labels adds +0.088 over zero-shot (CI +0.048 to +0.129), almost all on `neutro`;
  its best checkpoint came after only 70 steps (<1 epoch).
- Loading the model in 4 bits alone cost 0.105 macro-F1 before training (neutral nearly vanished).
- The human-label run learned from the same source and annotation as the evaluation, so part of
  its gain may be fitting that annotation rather than "more correct" labels.

- Adding the title and balancing class weights both helped TF-IDF (paired bootstrap, 95% CI of
  the difference above zero).
- Fine-tuned BERTimbau beats the best TF-IDF by +0.026 macro-F1 (paired 95% CI +0.009 to
  +0.042), almost all of it on `neutro`. It trained for ~25 min on a Kaggle T4 vs ~25 s on a
  laptop CPU for TF-IDF.
- A second epoch lowered training loss but also dev macro-F1 (0.787 → 0.774), so epoch 1 was kept.
- Both configurations were selected on dev, so their dev scores are optimistic.
  Details: decisions D13 and D14 in [docs/decisions.md](docs/decisions.md).

## Limitations

To be documented as the project progresses. Known from the start:

- Labels derived from star ratings are noisy: on dev they agree with human labels in 74.4% of reviews.
- The test set's class distribution is artificial (RePro sampled ~2,000 reviews per star rating),
  so accuracy does not reflect real-world traffic; macro-F1 is the main metric.
- Domain: e-commerce product reviews from 2018, not restaurant or food-delivery reviews.
- Only exact text duplicates between training and evaluation were removed; near-duplicates were not.
- TF-IDF hyperparameters were fixed, not tuned, so the classical baseline is reasonable but not optimal.
- BERTimbau was trained once (one seed) with standard, untuned hyperparameters; GPU training is
  not bit-for-bit deterministic, so a rerun may differ slightly. Its training epoch was picked on dev.
- Possible data contamination: both datasets are public, and SmolLM3's pre-training data was not
  checked for them. Few-shot was evaluated with a single draw of 6 examples.
- QLoRA runs were compared on 500 dev reviews (only 20 pure-neutral), so their intervals are about
  twice as wide as on the full dev; checkpoints were chosen on the same 500. Each run used one seed.

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
- **Stage 3b — a stronger model buys a small, measurable gain; the GPU is the bottleneck.**
  Fine-tuned BERTimbau beat TF-IDF by 2.6 macro-F1 points, with a paired bootstrap CI that
  excludes zero, at ~60× the training time. More training was not better: epoch 2 lowered training
  loss but also dev macro-F1. Free Colab had no GPU available, so the notebook was made to run on
  Kaggle too; the data check (rebuilt counts must match the repository) passed under a different
  pandas version, which is what makes the cross-platform run trustworthy.
- **Stage 4 — measure the base model before fine-tuning it; it can be the strongest baseline.**
  SmolLM3-3B with only an instruction beat every trained model on dev, and adding examples to the
  prompt made it worse, which is the opposite of the usual intuition. Reading the chat template
  also mattered: by default it writes today's date into the prompt, which silently breaks
  reproducibility. Scoring labels and generating answers gave almost identical predictions, so
  for this model the answer format was not a problem.
- **Stage 5 — label quality beat data size and model size.** Before training, measuring the label
  noise showed it was systematic (3-star reviews are often positive for humans), so two runs were
  compared: 1,500 human labels won by 11 points over 15,000 star labels. The star-label QLoRA model
  agreed with BERTimbau on 95% of reviews: the labels, more than the architecture, decided what was
  learned. Also unexpected: quantizing to 4 bits cost 10 points before any training.
- **Stage 6 — decide how to read the test before reading it.** The three hypotheses, the Bonferroni
  correction and the verdict rule were committed before the single test run, so no comparison could
  be picked after seeing the numbers. The test agreed with dev within 0.012 for every model, and its
  larger size turned an inconclusive dev result (star-label fine-tuning) into a measurable loss.

## License

To be defined (it must be compatible with the dataset and base model licenses).
