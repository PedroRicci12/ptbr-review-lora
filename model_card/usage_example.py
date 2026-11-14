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
