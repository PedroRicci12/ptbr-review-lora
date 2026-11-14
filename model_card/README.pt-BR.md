# SmolLM3-3B QLoRA — sentimento de avaliações de produtos em português do Brasil

> English version: [README.md](README.md).

Adapter LoRA para o [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) que classifica
avaliações de produtos em português do Brasil como **`negativo`**, **`neutro`** ou **`positivo`**
(`neutro` inclui avaliações mistas, com pontos positivos e negativos).
Foi treinado com QLoRA em 1.500 avaliações rotuladas por humanos.

Num conjunto de teste separado, com 8.003 avaliações rotuladas por humanos, chega a
**macro-F1 0,911** (IC de 95%: 0,904–0,917), +0,097 acima do mesmo modelo sem fine-tuning.

Código, preparação dos dados, todos os experimentos e os resultados negativos:
[github.com/PedroRicci12/ptbr-review-lora](https://github.com/PedroRicci12/ptbr-review-lora).

## Resultados no teste

8.003 avaliações do [RePro](https://huggingface.co/datasets/lucasnil/repro) com polaridade
rotulada por humanos, nunca usadas em treino nem em escolha de modelo, avaliadas **uma única vez**
depois de o protocolo ser commitado (commit `763c802`). Macro-F1 com IC de 95% por
bootstrap:

| Modelo | Macro-F1 (IC 95%) | Acurácia | F1 `neutro` |
|---|---|---|---|
| Classe majoritária (`positivo`) | 0,195 (0,191–0,198) | 0,413 | 0,000 |
| TF-IDF + regressão logística | 0,757 (0,747–0,766) | 0,785 | 0,558 |
| BERTimbau base ajustado (rótulos de estrelas) | 0,784 (0,775–0,793) | 0,803 | 0,618 |
| SmolLM3-3B + QLoRA, 15.000 rótulos de estrelas | 0,789 (0,780–0,798) | 0,810 | 0,618 |
| SmolLM3-3B zero-shot (modelo base) | 0,814 (0,804–0,823) | 0,850 | 0,634 |
| **Este adapter** (QLoRA, 1.500 rótulos humanos) | **0,911** (0,904–0,917) | 0,920 | 0,846 |

Comparações pré-registradas (bootstrap pareado, IC de 98,3% da diferença de macro-F1,
com correção de Bonferroni):

| | Comparação | Diferença (IC) | Veredito |
|---|---|---|---|
| H1 | este adapter vs zero-shot | +0,097 (+0,086 a +0,109) | melhor |
| H2 | QLoRA estrelas vs zero-shot | −0,025 (−0,037 a −0,013) | pior |
| H3 | este adapter vs QLoRA estrelas | +0,122 (+0,110 a +0,134) | melhor |

O treino com 15.000 rótulos de estrelas não foi publicado: o fine-tuning com estrelas deixou o
modelo *pior* que o zero-shot (H2), e 1.500 rótulos humanos superaram 15.000 de estrelas (H3).

## Como usar

Precisa de uma GPU CUDA (o carregamento em 4 bits usa `bitsandbytes`); foi executado numa T4
gratuita do Kaggle. O adapter pontua os 3 rótulos como respostas possíveis e devolve o mais
provável, então a saída é sempre um rótulo válido:

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

- **Formato da entrada:** título e texto da avaliação unidos como `"título. texto"` (só o texto
  quando não há título), exatamente como no treino.
- **Carregue o modelo base em 4 bits (NF4)**, como acima: os números foram medidos assim.
  Carregar em fp16 ou fundir o adapter muda as contas e não foi avaliado.
- Cada avaliação custa 3 passadas pelo modelo (uma por rótulo). Use lotes para grandes volumes.

## Treino

- **Dados:** 1.500 avaliações da parte de dev do RePro neste projeto (polaridade rotulada por
  humanos; "misto" vira `neutro`), 1.498 usadas depois de descartar 2 com
  mais de 512 tokens. Outras 500 do dev ficaram de fora para escolher o checkpoint.
- **Método:** QLoRA. Modelo base congelado em 4 bits (NF4, quantização dupla, cálculo em fp16);
  LoRA r=16, alpha=32, dropout 0,05 em todas as camadas lineares
  (30,2 M parâmetros treináveis).
- **Receita:** perda só nos tokens da resposta (rótulo + fim de turno); taxa de aprendizado
  0.0002 com 3% de aquecimento e decaimento linear; lote efetivo 16;
  3 épocas (279 passos). O melhor de 4 checkpoints nas 500 separadas foi o passo
  70 (macro-F1 de seleção 0,899, IC 0,870–0,927).
- **Computação:** uma Tesla T4, 41 min de treino; torch 2.11.0+cu130, transformers 5.20.2, peft 0.21.2, bitsandbytes 0.50.2.
  Revisão do modelo base `a07cc9a04f16550a088caea529712d1d335b0ac1`; código de treino no commit `ed4c06b`.

## Ablações (parte de seleção do dev, 500 avaliações)

**Quantos rótulos humanos são necessários?** Subconjuntos aninhados e estratificados dos 1.500,
mesma receita:

| Rótulos humanos | Macro-F1 (IC 95%) |
|---|---|
| 0 | 0,811 (0,775–0,848) |
| 250 | 0,871 (0,842–0,899) |
| 500 | 0,893 (0,865–0,919) |
| 1.000 | 0,913 (0,887–0,938) |
| 1.500 | 0,899 (0,870–0,927) |

Com 250 rótulos já supera o zero-shot; o ganho achata depois de cerca de 500.

**Variância entre seeds:** os mesmos 1.500 rótulos com 3 seeds de treino deram 0,899 / 0,908 / 0,909
(média 0,906, desvio-padrão 0,005). O adapter publicado (seed 42) foi o mais baixo dos três.

## Uso pretendido e limitações

- **Uso pretendido:** pesquisa, estudo e protótipos de classificação de sentimento para
  avaliações de produtos de e-commerce em português do Brasil. Só uso não comercial (licença abaixo).
- **Domínio:** avaliações de produtos de uma loja online brasileira, escritas em 2018. Outros
  domínios (restaurantes, delivery de comida, redes sociais) e outras variedades do português
  **não** foram avaliados.
- **Mesmo esquema de anotação:** as avaliações de treino e de teste vêm do mesmo dataset e das
  mesmas regras de anotação (partes diferentes, sem sobreposição). Parte do ganho sobre o zero-shot
  pode vir de aprender esse esquema, e não só de rótulos "melhores".
- **Possível contaminação:** os datasets são públicos; não foi verificado se estão nos dados de
  pré-treino do SmolLM3.
- **O equilíbrio de classes difere do tráfego real:** o RePro tem quase o mesmo número de
  avaliações por nota, então há mais neutras e negativas que numa loja de verdade.
- **`neutro` é a classe difícil** (F1 abaixo das outras duas), e a parte de seleção tinha só 20
  avaliações neutras puras.
- Não é feito para decisões sobre pessoas.

## Licença

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), herdada dos dados de treino
(RePro). O modelo base, SmolLM3-3B, é Apache 2.0.

## Créditos dos dados

- RePro: subconjunto da B2W-Reviews01 com polaridade anotada por humanos, por aspecto; avaliações
  com aspectos positivos e negativos viram `neutro` aqui
  ([lucasnil/repro](https://huggingface.co/datasets/lucasnil/repro)).
- B2W-Reviews01 (Americanas S.A.): usada só nos treinos de comparação com estrelas, não neste adapter.
