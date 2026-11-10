# ptbr-review-lora

[English](README.md) | **Português**

Fine-tuning com QLoRA de um LLM aberto pequeno (1B–3B parâmetros) para classificar avaliações
em português do Brasil como **negativa**, **neutra** ou **positiva**, comparado com o modelo
base e com baselines mais simples.

> **Status:** em andamento — Etapa 4 (LLM base, zero-shot e few-shot) concluída.
> Nenhum LLM passou por fine-tuning ainda. Todos os números abaixo são do dev; o teste ainda não foi usado.

## Objetivo

Medir se o fine-tuning eficiente em parâmetros (LoRA/QLoRA) de um LLM pequeno melhora a
classificação de sentimento em pt-BR em relação a:

1. o mesmo modelo sem fine-tuning (prompting zero-shot / few-shot);
2. baselines clássicas e baseadas em encoder (definidas na Etapa 3).

## Dados

| Papel | Dataset | Rótulos | Tamanho |
|---|---|---|---|
| Pool de treino | [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) (Americanas.com, 2018) | 1–5 estrelas → 3 classes | 118.093 |
| Dev (análise e ajustes) | [RePro](https://huggingface.co/datasets/lucasnil/repro), divisão estratificada | polaridade anotada por humanos | 2.000 |
| Teste (só avaliação final) | RePro, divisão estratificada | polaridade anotada por humanos | 8.003 |

- Estrelas: 1–2 → `negativo`, 3 → `neutro`, 4–5 → `positivo`. RePro: avaliações anotadas como
  negativas e positivas ao mesmo tempo ("misto") contam como `neutro`.
- O RePro é um subconjunto da B2W: as 10.003 avaliações do RePro foram removidas do pool de
  treino, além de 130 avaliações do pool com texto idêntico a um texto do RePro.
- Os campos de perfil do avaliador (ano de nascimento, gênero, estado) são descartados no carregamento.
- Os dois datasets são CC BY-NC-SA 4.0, então os modelos treinados aqui são não comerciais.
- No dev, o rótulo das estrelas concorda com o rótulo humano em **74,4%** das avaliações (medido).

Contagens por split: [reports/data_summary.json](reports/data_summary.json).
Justificativas: decisões D5–D9 em [docs/decisions.md](docs/decisions.md).

## Avaliação

- **Métrica principal:** macro-F1 sobre as 3 classes. Secundárias: acurácia, F1 por classe,
  matriz de confusão e recall do `neutro` separado em neutro puro e misto.
- **Incerteza:** bootstrap de percentis com 95% (1.000 reamostras); modelos são comparados com
  bootstrap pareado nas mesmas reamostras.
- **Protocolo:** toda escolha é feita no dev. O teste é usado uma única vez, no fim, para todos
  os modelos congelados juntos.

## Estrutura do repositório

```
├── src/ptbr_review_lora/   # código reutilizável (dados, prompts, avaliação)
├── scripts/                # pontos de entrada de linha de comando (ex.: preparação dos dados)
├── notebooks/              # notebooks para Colab/Kaggle (treino e inferência em GPU)
├── reports/                # arquivos pequenos de resultados (só contagens e métricas, sem texto)
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

Baixar (~55 MB, com versão fixada e checksum verificado) e preparar os dados:

```bash
uv run python scripts/prepare_data.py
```

Treinar as baselines clássicas e avaliá-las no dev (~2 minutos na CPU de um notebook):

```bash
uv run python scripts/run_baselines.py
```

Ajustar a baseline BERTimbau (GPU) com
[notebooks/03b_bertimbau.ipynb](notebooks/03b_bertimbau.ipynb), em qualquer uma das plataformas:

- **Google Colab:** [abrir no Colab](https://colab.research.google.com/github/PedroRicci12/ptbr-review-lora/blob/main/notebooks/03b_bertimbau.ipynb),
  escolha uma GPU T4 e rode todas as células; o navegador baixa o `bertimbau_results.zip`.
- **Kaggle:** importe o notebook do GitHub (*File → Import Notebook*), ajuste *Accelerator →
  GPU T4 x2* e *Internet → on* e clique em *Save Version → Save & Run All*; o zip fica na aba
  *Output* da versão. GPUs no Kaggle exigem conta verificada por celular.

Descompacte os resultados na raiz do repositório: os arquivos vão para `reports/` e `predictions/`.

Avaliar o LLM base (SmolLM3-3B, zero-shot e 6-shot, GPU) com
[notebooks/04_base_llm.ipynb](notebooks/04_base_llm.ipynb), no Colab
([abrir no Colab](https://colab.research.google.com/github/PedroRicci12/ptbr-review-lora/blob/main/notebooks/04_base_llm.ipynb))
ou no Kaggle, do mesmo jeito; ele gera o `smollm3_base_results.zip`.

## Resultados

Os resultados finais (no teste) ainda não existem. Todo número desta seção vem de uma execução
real, com o script ou notebook que o gerou.

**Resultados no dev** (2.000 avaliações; de
[reports/baselines_dev.json](reports/baselines_dev.json),
[reports/bertimbau_dev.json](reports/bertimbau_dev.json) e
[reports/smollm3_base_dev.json](reports/smollm3_base_dev.json)):

| Modelo | Macro-F1 (IC 95%) | Acurácia | F1 `neutro` |
|---|---|---|---|
| Classe majoritária (`positivo`) | 0,195 (0,187–0,202) | 0,413 | 0,000 |
| TF-IDF + regressão logística, texto | 0,672 (0,651–0,693) | 0,765 | — |
| TF-IDF + RL, título + texto, pesos balanceados | 0,762 (0,741–0,779) | 0,793 | 0,560 |
| BERTimbau base ajustado (1 época, perda balanceada) | 0,787 (0,768–0,804) | 0,807 | 0,621 |
| SmolLM3-3B sem fine-tuning, 6-shot | 0,760 (0,740–0,778) | 0,801 | 0,535 |
| SmolLM3-3B sem fine-tuning, zero-shot | **0,815** (0,795–0,833) | 0,851 | 0,638 |

- O LLM base em zero-shot supera o BERTimbau ajustado em +0,028 de macro-F1 (IC pareado de 95%:
  +0,007 a +0,050). Colocar 6 exemplos no prompt piorou (−0,055), então ficou o zero-shot.
- Pontuar os rótulos e gerar a resposta concordam em 99,85% das avaliações; 0% das respostas
  geradas foram inválidas.
- Cuidado: a B2W e o RePro são públicos, então podem estar nos dados de pré-treino do SmolLM3
  (não verificado); isso inflaria o resultado do zero-shot.

- Incluir o título e balancear os pesos das classes ajudou a TF-IDF (bootstrap pareado, IC de
  95% da diferença acima de zero).
- O BERTimbau ajustado supera a melhor TF-IDF em +0,026 de macro-F1 (IC pareado de 95%: +0,009
  a +0,042), quase tudo no `neutro`. Treinou em ~25 min numa T4 do Kaggle, contra ~25 s da TF-IDF
  na CPU de um notebook.
- Uma segunda época baixou a perda de treino, mas também o macro-F1 no dev (0,787 → 0,774),
  então ficou a época 1.
- As duas configurações foram escolhidas no dev, então os números delas no dev são otimistas.
  Detalhes: decisões D13 e D14 em [docs/decisions.md](docs/decisions.md).

## Limitações

Serão documentadas ao longo do projeto. Já conhecida desde o início:

- Rótulos derivados de estrelas têm ruído: no dev, concordam com o rótulo humano em 74,4% das avaliações.
- A distribuição de classes do teste é artificial (o RePro amostrou ~2.000 avaliações por nota),
  então a acurácia não reflete o tráfego real; a métrica principal é o macro-F1.
- Domínio: avaliações de produtos de e-commerce de 2018, não de restaurantes ou delivery.
- Só duplicatas exatas de texto entre treino e avaliação foram removidas; quase-duplicatas não.
- Os hiperparâmetros do TF-IDF foram fixados, não ajustados: a baseline clássica é razoável, mas não ótima.
- O BERTimbau foi treinado uma vez (uma seed) com hiperparâmetros padrão, não ajustados; treino em
  GPU não é determinístico bit a bit, então uma nova execução pode variar um pouco. A época foi escolhida no dev.
- Possível contaminação: os dois datasets são públicos, e não foi verificado se estão nos dados de
  pré-treino do SmolLM3. O few-shot foi avaliado com um único sorteio de 6 exemplos.

## Aprendizados

Preenchidos ao final de cada etapa.

- **Etapa 1 — caminhos com acento quebram a instalação editável no Python 3.12 / Windows.**
  Com o projeto em `...\Área de Trabalho\...`, o `uv run` falhou com
  `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. A instalação editável grava
  o caminho de `src/` num arquivo `.pth` em UTF-8 (`Á` = `C3 81`), mas o Python 3.12 no Windows
  lê arquivos `.pth` com a codificação do sistema (cp1252), na qual `0x81` não existe.
  Correção: mover o projeto para um caminho sem acento (e fora do OneDrive). Ver D4 em
  [docs/decisions.md](docs/decisions.md).
- **Etapa 2 — confirmar os fatos sobre os dados nos próprios dados.** O "dataset" da B2W no
  Hugging Face Hub é só um script de carregamento (sem dados), então o CSV é baixado da fonte
  original no GitHub. A descrição da fonte sugeria separador `;`, mas o arquivo usa `,`: ler o
  cabeçalho resolveu. Remover o RePro do pool de treino pelo ID também não bastou: 92 textos do
  RePro ainda apareciam idênticos no pool (provavelmente a mesma avaliação publicada para produtos diferentes).
- **Etapa 3 — uma baseline barata define a régua.** TF-IDF + regressão logística, treinada em
  ~25 s, chega a 0,762 de macro-F1 no dev, e seus 79% de acurácia já superam os 74% de
  concordância entre estrelas e rótulos humanos. O que mais pesou foi balancear os pesos das
  classes: o recall do neutro puro subiu de 7% para 38%, porque o neutro é 12% do treino e 24% da avaliação.
- **Etapa 3b — um modelo mais forte compra um ganho pequeno e mensurável; a GPU é o gargalo.**
  O BERTimbau ajustado superou a TF-IDF em 2,6 pontos de macro-F1, com IC do bootstrap pareado
  excluindo o zero, ao custo de ~60× o tempo de treino. Treinar mais não foi melhor: a época 2
  baixou a perda de treino, mas também o macro-F1 no dev. O Colab gratuito não tinha GPU, então o
  notebook passou a rodar também no Kaggle; a checagem dos dados (contagens reconstruídas iguais
  às do repositório) passou com outra versão do pandas, e é isso que torna confiável rodar em outra plataforma.
- **Etapa 4 — medir o modelo base antes de ajustá-lo; ele pode ser a baseline mais forte.**
  O SmolLM3-3B, só com uma instrução, superou todos os modelos treinados no dev, e colocar exemplos
  no prompt piorou o resultado, o contrário da intuição comum. Ler o chat template também fez
  diferença: por padrão ele escreve a data de hoje no prompt, o que quebra a reprodutibilidade sem
  avisar. Pontuar os rótulos e gerar a resposta deram previsões quase idênticas, então, para este
  modelo, o formato da resposta não foi problema.

## Licença

A definir (precisa ser compatível com as licenças do dataset e do modelo base).
