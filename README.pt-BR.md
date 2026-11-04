# ptbr-review-lora

[English](README.md) | **Português**

Fine-tuning com QLoRA de um LLM aberto pequeno (1B–3B parâmetros) para classificar avaliações
em português do Brasil como **negativa**, **neutra** ou **positiva**, comparado com o modelo
base e com baselines mais simples.

> **Status:** em andamento — Etapa 1 (estrutura do projeto). Nenhum modelo foi treinado ainda
> e não há resultados.

## Objetivo

Medir se o fine-tuning eficiente em parâmetros (LoRA/QLoRA) de um LLM pequeno melhora a
classificação de sentimento em pt-BR em relação a:

1. o mesmo modelo sem fine-tuning (prompting zero-shot / few-shot);
2. baselines clássicas e baseadas em encoder (definidas na Etapa 3).

## Estrutura do repositório

```
├── src/ptbr_review_lora/   # código reutilizável (dados, prompts, avaliação)
├── notebooks/              # notebooks para Colab/Kaggle (treino e inferência em GPU)
├── tests/                  # testes com pytest
└── docs/decisions.md       # decisões de design e alternativas consideradas
```

## Instalação

Requisitos: Python 3.12 e [uv](https://docs.astral.sh/uv/).

```bash
git clone <url-do-repo>
cd ptbr-review-lora
uv sync
cp .env.example .env   # depois adicione seu token do Hugging Face
```

## Uso

Rodar os testes:

```bash
uv run pytest
```

O treino em GPU roda no Google Colab ou no Kaggle; os notebooks entram na Etapa 5.

## Resultados

Ainda não há. Todo número desta seção virá de uma execução real, com o notebook que o gerou.

## Limitações

Serão documentadas ao longo do projeto. Já conhecida desde o início:

- Rótulos derivados de estrelas têm ruído (o texto e a nota podem discordar).

## Aprendizados

Preenchidos ao final de cada etapa.

- **Etapa 1 — caminhos com acento quebram a instalação editável no Python 3.12 / Windows.**
  Com o projeto em `...\Área de Trabalho\...`, o `uv run` falhou com
  `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. A instalação editável grava
  o caminho de `src/` num arquivo `.pth` em UTF-8 (`Á` = `C3 81`), mas o Python 3.12 no Windows
  lê arquivos `.pth` com a codificação do sistema (cp1252), na qual `0x81` não existe.
  Correção: mover o projeto para um caminho sem acento (e fora do OneDrive). Ver D4 em
  [docs/decisions.md](docs/decisions.md).

## Licença

A definir (precisa ser compatível com as licenças do dataset e do modelo base).
