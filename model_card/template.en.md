---
base_model: HuggingFaceTB/SmolLM3-3B
library_name: peft
license: cc-by-nc-sa-4.0
language:
- pt
datasets:
- lucasnil/repro
metrics:
- f1
- accuracy
tags:
- lora
- qlora
- peft
- sentiment-analysis
- text-classification
- brazilian-portuguese
model-index:
- name: smollm3-3b-ptbr-review-sentiment-qlora
  results:
  - task:
      type: text-classification
      name: Sentiment classification (3 classes)
    dataset:
      name: RePro (held-out test split of this project)
      type: lucasnil/repro
    metrics:
    - type: f1
      name: Macro-F1
      value: $test_f1_raw
    - type: accuracy
      name: Accuracy
      value: $test_accuracy_raw
---

# SmolLM3-3B QLoRA — sentiment of Brazilian Portuguese product reviews

> **Resumo em português:** adapter LoRA para o SmolLM3-3B que classifica avaliações de produtos
> em pt-BR como `negativo`, `neutro` ou `positivo`. Treinado com 1.500 rótulos humanos, chega a
> macro-F1 $test_f1 no teste (zero-shot: $zero_shot_f1). Card completo em português:
> [README.pt-BR.md](README.pt-BR.md).

A LoRA adapter for [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) that classifies
Brazilian Portuguese product reviews as **`negativo`**, **`neutro`** or **`positivo`**
(`neutro` includes mixed reviews, with both positive and negative points).
It was trained with QLoRA on 1,500 human-labeled reviews.

On a held-out test set of $test_n human-labeled reviews it reaches **macro-F1 $test_f1**
(95% CI $test_f1_ci), $h1_diff over the same model without fine-tuning.

Code, data preparation, all experiments and negative results:
[github.com/PedroRicci12/ptbr-review-lora](https://github.com/PedroRicci12/ptbr-review-lora).

## Results on the test set

$test_n reviews from [RePro](https://huggingface.co/datasets/lucasnil/repro) with human polarity
labels, never used for training or model selection, scored **once** after the protocol was
committed (commit `$protocol_commit`). Macro-F1 with a 95% bootstrap CI:

| Model | Macro-F1 (95% CI) | Accuracy | F1 `neutro` |
|---|---|---|---|
$test_table

Pre-registered comparisons (paired bootstrap, Bonferroni-corrected $family_confidence% CIs of the
macro-F1 difference):

| | Comparison | Difference (CI) | Verdict |
|---|---|---|---|
$hypotheses_table

The 15,000 star-label run is not published: fine-tuning on star ratings made the model *worse*
than zero-shot (H2), and 1,500 human labels beat 15,000 star labels (H3).

## How to use

Needs a CUDA GPU (4-bit loading uses `bitsandbytes`); it was run on a free Kaggle T4.
The adapter scores the 3 labels as possible answers and returns the most likely one, so the
output is always a valid label:

```python
$usage
```

- **Input format:** the review title and text joined as `"title. text"` (just the text when
  there is no title), exactly as in training.
- **Load the base model in 4 bits (NF4)**, as above: the reported numbers were measured that way.
  Loading it in fp16 or merging the adapter changes the computation and was not evaluated.
- Each review costs 3 forward passes (one per label). Batch them for large volumes.

## Training

- **Data:** 1,500 reviews from the RePro dev split of this project (human polarity labels;
  "mixed" mapped to `neutro`), $training_examples used after dropping $dropped_long longer than
  $max_train_tokens tokens. Another 500 dev reviews were held out to pick the checkpoint.
- **Method:** QLoRA. Base model frozen in 4 bits (NF4, double quantization, fp16 compute); LoRA
  r=$lora_r, alpha=$lora_alpha, dropout $lora_dropout on every linear layer ($trainable_params M
  trainable parameters).
- **Recipe:** loss only on the answer tokens (label + end of turn); learning rate $learning_rate
  with 3% warmup and linear decay; effective batch $effective_batch; $epochs epochs
  ($total_steps steps). The best of 4 checkpoints on the 500 held-out reviews was step
  $best_step (selection macro-F1 $select_f1, CI $select_f1_ci).
- **Compute:** one $gpu, $train_minutes min of training; $versions.
  Base model revision `$model_revision`; training code at commit `$training_commit`.

## Ablations (dev selection set, 500 reviews)

**How many human labels are needed?** Nested, stratified subsets of the 1,500, same recipe:

| Human labels | Macro-F1 (95% CI) |
|---|---|
$learning_curve_table

250 labels already beat zero-shot; gains flatten after about 500.

**Seed variance:** the same 1,500 labels with 3 training seeds gave $seed_f1s
(mean $seed_mean, std $seed_std). The published adapter (seed 42) was the lowest of the three.

## Intended use and limitations

- **Intended use:** research, study and prototyping of sentiment classification for Brazilian
  Portuguese e-commerce product reviews. Non-commercial only (license below).
- **Domain:** product reviews from one Brazilian online store, written in 2018. Other domains
  (restaurants, food delivery, social media) and other varieties of Portuguese were **not**
  evaluated.
- **Same annotation scheme:** the training and test reviews come from the same dataset and
  annotation guidelines (different, non-overlapping splits). Part of the gain over zero-shot may
  come from learning that scheme, not only from "better" labels.
- **Possible contamination:** the datasets are public; whether they appear in SmolLM3's
  pre-training data was not checked.
- **Class balance differs from real traffic:** RePro samples about the same number of reviews
  per star rating, so the share of neutral and negative reviews is higher than in a live store.
- **`neutro` is the hard class** (F1 below the other two), and the selection set had only 20
  purely neutral reviews.
- Not meant for decisions about individuals.

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), inherited from the training
data (RePro). The base model, SmolLM3-3B, is Apache 2.0.

## Data credits

- RePro: a human-annotated subset of B2W-Reviews01, with polarity annotated per aspect; reviews
  with both positive and negative aspects are mapped to `neutro` here ([lucasnil/repro](https://huggingface.co/datasets/lucasnil/repro)).
- B2W-Reviews01 (Americanas S.A.): used only for the star-label comparison runs, not for this adapter.
