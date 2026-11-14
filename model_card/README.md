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
      value: 0.9110
    - type: accuracy
      name: Accuracy
      value: 0.9200
---

# SmolLM3-3B QLoRA — sentiment of Brazilian Portuguese product reviews

> **Resumo em português:** adapter LoRA para o SmolLM3-3B que classifica avaliações de produtos
> em pt-BR como `negativo`, `neutro` ou `positivo`. Treinado com 1.500 rótulos humanos, chega a
> macro-F1 0.911 no teste (zero-shot: 0.814). Card completo em português:
> [README.pt-BR.md](README.pt-BR.md).

A LoRA adapter for [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) that classifies
Brazilian Portuguese product reviews as **`negativo`**, **`neutro`** or **`positivo`**
(`neutro` includes mixed reviews, with both positive and negative points).
It was trained with QLoRA on 1,500 human-labeled reviews.

On a held-out test set of 8,003 human-labeled reviews it reaches **macro-F1 0.911**
(95% CI 0.904–0.917), +0.097 over the same model without fine-tuning.

Code, data preparation, all experiments and negative results:
[github.com/PedroRicci12/ptbr-review-lora](https://github.com/PedroRicci12/ptbr-review-lora).

## Results on the test set

8,003 reviews from [RePro](https://huggingface.co/datasets/lucasnil/repro) with human polarity
labels, never used for training or model selection, scored **once** after the protocol was
committed (commit `763c802`). Macro-F1 with a 95% bootstrap CI:

| Model | Macro-F1 (95% CI) | Accuracy | F1 `neutro` |
|---|---|---|---|
| Majority class (`positivo`) | 0.195 (0.191–0.198) | 0.413 | 0.000 |
| TF-IDF + logistic regression | 0.757 (0.747–0.766) | 0.785 | 0.558 |
| BERTimbau base, fine-tuned (star labels) | 0.784 (0.775–0.793) | 0.803 | 0.618 |
| SmolLM3-3B + QLoRA, 15,000 star labels | 0.789 (0.780–0.798) | 0.810 | 0.618 |
| SmolLM3-3B zero-shot (base model) | 0.814 (0.804–0.823) | 0.850 | 0.634 |
| **This adapter** (QLoRA, 1,500 human labels) | **0.911** (0.904–0.917) | 0.920 | 0.846 |

Pre-registered comparisons (paired bootstrap, Bonferroni-corrected 98.3% CIs of the
macro-F1 difference):

| | Comparison | Difference (CI) | Verdict |
|---|---|---|---|
| H1 | this adapter vs zero-shot | +0.097 (+0.086 to +0.109) | better |
| H2 | QLoRA star labels vs zero-shot | −0.025 (−0.037 to −0.013) | worse |
| H3 | this adapter vs QLoRA star labels | +0.122 (+0.110 to +0.134) | better |

The 15,000 star-label run is not published: fine-tuning on star ratings made the model *worse*
than zero-shot (H2), and 1,500 human labels beat 15,000 star labels (H3).

## How to use

Needs a CUDA GPU (4-bit loading uses `bitsandbytes`); it was run on a free Kaggle T4.
The adapter scores the 3 labels as possible answers and returns the most likely one, so the
output is always a valid label:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"
BASE_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
ADAPTER = "PedroRicci12/smollm3-3b-ptbr-review-sentiment-qlora"
LABELS = ("negativo", "neutro", "positivo")
INSTRUCTION = (
    "Você classifica o sentimento de avaliações de produtos de uma loja online brasileira.\n"
    "Responda com uma única palavra, em minúsculas e sem pontuação:\n"
    "- negativo: o cliente está insatisfeito com o produto ou com a compra;\n"
    "- neutro: a avaliação não é positiva nem negativa, ou tem pontos positivos e negativos "
    "ao mesmo tempo;\n"
    "- positivo: o cliente está satisfeito com o produto ou com a compra."
)

# The adapter was trained and evaluated on top of the base model in 4 bits (NF4), as here.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, revision=BASE_REVISION, quantization_config=quantization,
    dtype=torch.float16, device_map={"": 0},
)
model = PeftModel.from_pretrained(model, ADAPTER).eval()
LABEL_IDS = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in LABELS]


def prompt_ids(review: str) -> list[int]:
    """Chat prompt ending where the answer starts (no date in the system turn, thinking off)."""
    messages = [
        {"role": "system", "content": INSTRUCTION + "\n/system_override"},
        {"role": "user", "content": f"Avaliação: {review.strip()}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def classify(review: str) -> str:
    """Score each label as the answer (sum of its token log-probabilities); the highest wins."""
    prompt = prompt_ids(review)
    scores = []
    for ids in LABEL_IDS:
        input_ids = torch.tensor([prompt + ids], device=model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids).logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        scores.append(sum(log_probs[len(prompt) - 1 + k, t].item() for k, t in enumerate(ids)))
    return LABELS[max(range(len(LABELS)), key=scores.__getitem__)]


# Title and text were joined as "title. text" during training and evaluation.
print(classify("Ótimo. Chegou antes do prazo e funciona muito bem."))
print(classify("Não recomendo. Parou de funcionar depois de uma semana."))
```

- **Input format:** the review title and text joined as `"title. text"` (just the text when
  there is no title), exactly as in training.
- **Load the base model in 4 bits (NF4)**, as above: the reported numbers were measured that way.
  Loading it in fp16 or merging the adapter changes the computation and was not evaluated.
- Each review costs 3 forward passes (one per label). Batch them for large volumes.

## Training

- **Data:** 1,500 reviews from the RePro dev split of this project (human polarity labels;
  "mixed" mapped to `neutro`), 1,498 used after dropping 2 longer than
  512 tokens. Another 500 dev reviews were held out to pick the checkpoint.
- **Method:** QLoRA. Base model frozen in 4 bits (NF4, double quantization, fp16 compute); LoRA
  r=16, alpha=32, dropout 0.05 on every linear layer (30.2 M
  trainable parameters).
- **Recipe:** loss only on the answer tokens (label + end of turn); learning rate 0.0002
  with 3% warmup and linear decay; effective batch 16; 3 epochs
  (279 steps). The best of 4 checkpoints on the 500 held-out reviews was step
  70 (selection macro-F1 0.899, CI 0.870–0.927).
- **Compute:** one Tesla T4, 41 min of training; torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2.
  Base model revision `a07cc9a04f16550a088caea529712d1d335b0ac1`; training code at commit `ed4c06b`.

## Ablations (dev selection set, 500 reviews)

**How many human labels are needed?** Nested, stratified subsets of the 1,500, same recipe:

| Human labels | Macro-F1 (95% CI) |
|---|---|
| 0 | 0.811 (0.775–0.848) |
| 250 | 0.871 (0.842–0.899) |
| 500 | 0.893 (0.865–0.919) |
| 1,000 | 0.913 (0.887–0.938) |
| 1,500 | 0.899 (0.870–0.927) |

250 labels already beat zero-shot; gains flatten after about 500.

**Seed variance:** the same 1,500 labels with 3 training seeds gave 0.899 / 0.908 / 0.909
(mean 0.906, std 0.005). The published adapter (seed 42) was the lowest of the three.

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
