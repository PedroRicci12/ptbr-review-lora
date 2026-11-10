# Registro de decisões

Cada decisão de design do projeto, com as alternativas consideradas e o motivo da escolha.
Formato inspirado em ADRs (Architecture Decision Records).

---

## D1 — Tarefa: sentimento em 3 classes

- **Data:** 2026-11-03 (Etapa 1)
- **Decisão:** classificar avaliações em pt-BR como `negativo`, `neutro` ou `positivo`,
  com rótulos derivados da nota em estrelas (1–2 → negativo, 3 → neutro, 4–5 → positivo).
  O mapeamento exato será confirmado na Etapa 2, depois de olhar os dados.
- **Alternativas:**
  - Nota de 1 a 5: mais granular, mas classes vizinhas se confundem e a métrica fica difícil de interpretar.
  - Classificação por aspecto (entrega, comida, preço, atendimento): mais próxima do negócio de
    delivery, mas não há dataset público rotulado conhecido; exigiria rotulagem manual ou por LLM.
- **Motivo:** rótulos já disponíveis, avaliação objetiva e escopo que cabe no projeto.
  A classificação por aspecto fica como extensão futura.
- **Riscos conhecidos:** rótulo vindo de estrelas tem ruído (texto negativo com nota 4, por exemplo);
  a classe `neutro` tende a ser a mais difícil e a menos frequente.

## D2 — Gerenciador de dependências: uv

- **Data:** 2026-11-03 (Etapa 1)
- **Decisão:** `uv` com `pyproject.toml` + `uv.lock`.
- **Alternativa:** `pip` + `requirements.txt` fixado. Mais simples e nativo no Colab.
- **Motivo:** o `uv.lock` fixa versões exatas de todas as dependências (inclusive as transitivas)
  e o uv é muito mais rápido que o pip. No Colab, o notebook instalará as dependências de GPU
  com versões fixadas explicitamente.

## D3 — Documentação bilíngue

- **Data:** 2026-11-03 (Etapa 1)
- **Decisão:** `README.md` em inglês (versão principal) e `README.pt-BR.md` em português,
  com link entre eles no topo. Este registro de decisões fica em português.
- **Custo:** os dois READMEs precisam ser atualizados juntos a cada etapa.

## D4 — Local do projeto: caminho sem acento, fora do OneDrive

- **Data:** 2026-11-03 (Etapa 1)
- **Problema (medido):** em `C:\Users\pedro\OneDrive\Área de Trabalho\Projetos\...`, o `uv sync`
  funcionou, mas o `uv run pytest` falhou ao iniciar o Python da `.venv` com
  `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 25`.
  O arquivo `_editable_impl_ptbr_review_lora.pth` continha o caminho em UTF-8; o byte 25 é o
  `0x81` do `Á` (`C3 81`). O Python 3.12 no Windows lê `.pth` em cp1252, onde `0x81` é indefinido.
- **Decisão:** mover o projeto para `C:\Users\pedro\Projetos\ptbr-review-lora`, com um atalho
  na Área de Trabalho.
- **Alternativas:**
  - `PYTHONUTF8=1` no sistema: só contorna o problema na máquina local; quem clonar num caminho
    com acento cai no mesmo erro.
  - Python 3.13 localmente: diverge do Python 3.12 usado no Colab.
- **Motivo:** resolve a causa e evita também que o OneDrive sincronize os milhares de arquivos da `.venv`.
- **Resultado (medido):** no novo caminho, `uv sync` e `uv run pytest` passaram (1 teste).

## D5 — Dados: B2W-Reviews01 para treino, RePro para avaliação

- **Data:** 2026-11-06 (Etapa 2)
- **Decisão:** treinar com [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01)
  (~130 mil avaliações, rótulo = estrelas) e avaliar com [RePro](https://huggingface.co/datasets/lucasnil/repro)
  (10.003 avaliações da própria B2W com polaridade anotada por humanos).
- **Alternativas consideradas:**
  - Zomato PT-EN (Mendeley, CC BY 4.0): restaurantes, mas vindo do Zomato Portugal,
    provavelmente pt-PT. Fica como possível teste fora do domínio no futuro.
  - Olist (Kaggle): sem vantagem clara sobre a B2W e o download exige login.
- **Motivo:** rótulos humanos no teste permitem medir o modelo contra a verdade humana, e não
  contra o ruído das estrelas.
- **Riscos e cuidados:**
  - **Vazamento:** o RePro é um subconjunto da B2W. As avaliações do RePro serão removidas
    do treino (chave: `reviewer_id`, `product_id`, `submission_date`, `review_text`).
  - **Licença:** ambos são CC BY-NC-SA 4.0 (o card do Hugging Face da B2W diz CC BY 4.0, mas o
    repositório original diz NC-SA). O modelo publicado herda CC BY-NC-SA 4.0.
  - **Dados pessoais:** `reviewer_birth_year`, `reviewer_gender` e `reviewer_state` são descartados
    no carregamento e nunca entram no repositório.
  - **Distribuição artificial:** o RePro tem ~2.000 avaliações por estrela (fonte: estatísticas
    do Hugging Face), então a distribuição de classes do teste não reflete o mundo real.
    Métrica principal: macro-F1.

## D6 — Rótulo "misto" do RePro vira `neutro`

- **Data:** 2026-11-06 (Etapa 2)
- **Contexto (fonte: estatísticas do Hugging Face):** a polaridade do RePro é por aspecto.
  Distribuição: `POSITIVO` 4.127, `NEGATIVO` 3.449, `NEGATIVO`+`POSITIVO` (misto) 2.018, `NEUTRO` 409.
- **Decisão:** `POSITIVO` → positivo; `NEGATIVO` → negativo; `NEUTRO` e misto → neutro.
- **Alternativas:**
  - 4 classes: mais fiel à anotação, mas o treino com estrelas não tem rótulo "misto".
  - Excluir o misto do teste: remove 20% dos casos difíceis e infla a métrica.
- **Motivo:** coerente com o treino, em que 3 estrelas vira neutro.
- **Mitigação:** reportar o desempenho na classe neutro separado em "neutro puro" e "misto".

## D7 — RePro dividido em dev (2.000) e teste (8.003)

- **Data:** 2026-11-06 (Etapa 2)
- **Decisão:** divisão estratificada pelos 4 grupos de polaridade originais (positivo, negativo,
  neutro puro, misto), com seed fixa. O **dev** serve para toda análise e ajuste (mapeamento
  de estrelas, prompts, limiares). O **teste** é usado só no fim, uma vez por modelo.
- **Alternativa:** 1.000 / 9.003. Teste um pouco maior, mas só ~40 neutros puros no dev.
- **Motivo:** olhar o teste para tomar decisões tornaria a métrica final otimista.
  Com 2.000 exemplos, o dev tem ~80 neutros puros (estimativa pela proporção do RePro).
- **Resultado (medido):** dev = 825 positivo, 690 negativo, 403 misto, 82 neutro puro;
  teste = 3.302 positivo, 2.759 negativo, 1.615 misto, 327 neutro puro.
- **Atualização (Etapa 5, D16):** para o fine-tuning com rótulos humanos, o dev foi dividido em
  1.500 de treino e 500 de seleção. O teste continua intocado.

## D8 — Mapeamento estrelas → classe confirmado no dev

- **Data:** 2026-11-06 (Etapa 2)
- **Evidência (medida no dev, 2.000 avaliações):** porcentagem de cada polaridade humana por nota.

  | Estrelas | negativo | neutro | misto | positivo |
  |---|---|---|---|---|
  | 1 | 88,5 | 3,7 | 7,6 | 0,2 |
  | 2 | 62,6 | 4,8 | 30,7 | 1,9 |
  | 3 | 17,3 | 10,5 | 37,7 | 34,6 |
  | 4 | 0,5 | 0,8 | 18,7 | 80,0 |
  | 5 | 0,7 | 1,0 | 6,9 | 91,4 |

- **Decisão:** manter 1–2 → negativo, 3 → neutro, 4–5 → positivo. Em cada nota, a classe
  majoritária é a escolhida (na nota 3, neutro + misto = 48,2%).
- **Consequência:** o rótulo das estrelas concorda com o rótulo humano em **74,4%** do dev
  (1.487/2.000). Um modelo que só imitasse as estrelas ficaria perto desse valor; é uma
  referência para interpretar os resultados.
- **Cuidado:** a análise foi feita só no dev; o teste não foi consultado.

## D9 — Remover do treino textos idênticos aos do RePro

- **Data:** 2026-11-06 (Etapa 2)
- **Problema (medido):** depois de remover as avaliações do RePro pela chave, 21 textos do dev
  (1,1%) e 71 do teste (0,9%) ainda tinham texto idêntico no pool de treino (comparação sem
  diferenciar maiúsculas e ignorando espaços nas pontas). Eram textos longos e específicos
  (mediana de 83–97 caracteres). Hipótese, não verificada: a mesma pessoa publicou a mesma
  avaliação para produtos diferentes.
- **Decisão:** remover do pool toda avaliação com texto igual a um texto do RePro.
- **Alternativa:** manter. Custo zero, mas o modelo veria no treino textos exatos do teste.
- **Resultado (medido):** 130 linhas removidas; pool final com 118.093 avaliações.
- **Não tratado:** quase-duplicatas (texto com pequenas diferenças). Ficam como limitação.
  Também há 2 pares (revisor, produto) com avaliações diferentes no dev e no teste;
  efeito considerado desprezível.

## D10 — Métrica principal: macro-F1

- **Data:** 2026-11-06 (Etapa 3)
- **Decisão:** macro-F1 sobre as 3 classes; uma classe nunca prevista conta como F1 = 0.
  Secundárias: acurácia, F1 por classe, matriz de confusão e recall do neutro separado em
  "neutro puro" e "misto".
- **Alternativas:** acurácia e F1 ponderado (favorecem as classes grandes, e a distribuição do
  teste é artificial); kappa de Cohen / MCC (corretas, mas mais difíceis de comunicar).
- **Correção em relação ao plano:** o plano falava em "F1 do neutro por subgrupo". F1 por subgrupo
  não é bem definido (um falso `neutro` não pertence a nenhum subgrupo), então reportamos recall.

## D11 — Incerteza: bootstrap pareado

- **Data:** 2026-11-06 (Etapa 3)
- **Decisão:** intervalo de 95% por bootstrap de percentis (1.000 reamostras, seed 42) para cada
  modelo, e bootstrap **pareado** para comparar dois modelos (mesmos índices para os dois).
- **Alternativa:** comparar os intervalos individuais. Menos sensível, porque ignora que os dois
  modelos erram em exemplos parecidos.

## D12 — O teste só é usado na Etapa 6

- **Data:** 2026-11-06 (Etapa 3)
- **Decisão:** baselines, LLM base e LLM ajustado são avaliados no teste uma única vez, juntos,
  com as configurações já congeladas. Até lá, todas as escolhas usam só o dev.

## D13 — Baselines e configuração escolhida no dev

- **Data:** 2026-11-06 (Etapa 3)
- **Decisão:** baselines = classe majoritária, TF-IDF + regressão logística e, numa Etapa 3b,
  BERTimbau ajustado (GPU, Colab).
- **Hiperparâmetros fixos (não ajustados):** TF-IDF de 1–2 gramas, `min_df=2`, `sublinear_tf`;
  regressão logística com `C=1`.
- **Resultados no dev (medidos, 2.000 avaliações; IC de 95%):**

  | Modelo | Macro-F1 | IC 95% | Acurácia | F1 neutro | Recall neutro puro | Recall misto |
  |---|---|---|---|---|---|---|
  | Majoritária (`positivo`) | 0,195 | 0,187–0,202 | 0,413 | 0,000 | 0,000 | 0,000 |
  | TF-IDF, texto | 0,672 | 0,651–0,693 | 0,765 | 0,323 | 0,073 | 0,233 |
  | TF-IDF, texto, balanceado | 0,740 | 0,720–0,759 | 0,772 | 0,532 | 0,378 | 0,533 |
  | TF-IDF, título + texto | 0,702 | 0,680–0,724 | 0,783 | 0,380 | 0,098 | 0,293 |
  | **TF-IDF, título + texto, balanceado** | **0,762** | 0,741–0,779 | 0,793 | 0,560 | 0,427 | 0,553 |

- **Comparação pareada contra "TF-IDF, texto" (diferença de macro-F1, IC 95%):**
  balanceado +0,068 (0,048–0,089); título +0,030 (0,015–0,045); título + balanceado +0,090
  (0,068–0,110). Nenhuma das 1.000 reamostras teve diferença ≤ 0 nesses três casos.
- **Escolha:** entrada = **título + texto**, com **pesos balanceados**. Vale também para os
  próximos modelos (o título ajuda e o neutro é sub-representado no treino).
- **Cuidado:** como a configuração foi escolhida olhando o dev, o 0,762 é otimista; o número
  que vale é o do teste, na Etapa 6.

## D14 — Etapa 3b: BERTimbau base ajustado (fine-tuning completo)

- **Data:** 2026-11-09 (Etapa 3b)
- **Modelo:** `neuralmind/bert-base-portuguese-cased` (110M parâmetros, MIT, pré-treinado no brWaC),
  revisão fixada `94d69c9`. Os pesos só existem como `pytorch_model.bin` (pickle, que pode executar
  código ao carregar); aceito por vir do próprio grupo autor e com revisão fixada.
- **Decisões (escolhidas por Pedro):**
  - base, não large (large: ~3× parâmetros, mais tempo e memória);
  - pool inteiro (118.093), mesmo treino da regressão logística, para a comparação ser justa;
  - `MAX_LEN` = menor valor entre 128/256/384/512 que cobre 95% das avaliações (medido no notebook);
  - 2 épocas, ficando com a de melhor macro-F1 no dev;
  - perda com pesos balanceados por classe (mesma escolha da D13);
  - Google Colab, GPU T4. **Atualização:** na primeira tentativa o Colab gratuito não tinha GPU
    disponível ("Nenhum back-end com GPU disponível"). O notebook passou a rodar também no Kaggle
    (GPU T4), detectando a plataforma sozinho; a plataforma usada fica registrada no relatório.
    No Kaggle, a P100 é evitada porque versões recentes do PyTorch não suportam mais a arquitetura dela.
- **Hiperparâmetros fixos (padrão de fine-tuning de BERT, não ajustados):** lr 2e-5, batch 32,
  10% de warmup linear, weight decay 0,01 (exceto bias e LayerNorm), clipping 1,0, fp16, seed 42.
- **Teste:** o notebook gera as previsões do teste, mas não calcula métricas (D12). As previsões são
  commitadas antes da Etapa 6, o que mostra que não foram alteradas depois.
- **Mudança em relação ao plano:** não há checkpoints no Google Drive. Com as previsões do teste
  salvas, os pesos não são necessários depois; se a sessão cair, o custo é rodar de novo.
- **Limitação:** treino em GPU não é determinístico bit a bit; uma nova execução pode variar um pouco.
- **Resultados (medidos, dev, 2.000 avaliações; `reports/bertimbau_dev.json`):**
  - Execução: Kaggle, Tesla T4, commit `d0a9765`; pandas 2.3.3 no Kaggle, e a checagem de
    `data_summary.json` passou, então os dados são os mesmos do notebook local (pandas 3.1.1).
  - `MAX_LEN` = 128 (cobre 97,2% da amostra; mediana 32 tokens, p95 103, p99 184).
  - Treino: 1.478 s (~25 min) para as 2 épocas.

    | Época | Perda de treino | Macro-F1 dev |
    |---|---|---|
    | 1 | 0,487 | **0,787** |
    | 2 | 0,376 | 0,774 |

  - Melhor época: 1. A segunda baixou a perda de treino mas piorou o dev (resultado negativo
    registrado). Hipótese não testada: com rótulos de estrelas ruidosos (74% de concordância
    com humanos), treinar mais ajusta o modelo ao ruído. A diferença entre as épocas também pode
    ser variação normal; não foi medida com bootstrap.
  - Macro-F1 **0,787** (IC 95% 0,768–0,804), acurácia 0,807. F1 por classe: negativo 0,889,
    neutro 0,621, positivo 0,852. Recall do neutro: puro 0,585, misto 0,663.
  - Contra a melhor TF-IDF (0,762): diferença **+0,026** (IC 95% +0,009 a +0,042), `p_b_not_better`
    0,001 (bootstrap pareado). Os dois modelos concordam em 88% das previsões do dev.
  - O ganho veio quase todo do neutro (F1 0,560 → 0,621; recall do neutro puro 0,427 → 0,585);
    positivo ficou igual (0,851 → 0,852).
  - Maior erro: 163 positivos previstos como neutro. Nenhum positivo foi previsto como negativo,
    e só 2 negativos como positivo.
  - **Cuidado:** a época foi escolhida pelo dev, então o 0,787 é otimista, como o 0,762 da TF-IDF.
    As previsões do teste (8.003) foram salvas sem calcular métricas (D12).

## D15 — Etapa 4: LLM base sem fine-tuning (zero-shot e few-shot)

- **Data:** 2026-11-10 (Etapa 4)
- **Modelo (escolhido por Pedro):** `HuggingFaceTB/SmolLM3-3B` (Apache 2.0, sem gate). Português é
  um dos 6 idiomas suportados oficialmente; arquitetura só de texto, o caminho mais testado para
  QLoRA na Etapa 5; dados e receita de treino públicos.
  - Alternativas: Qwen3-1.7B (Apache 2.0, mas o card não cita português), Qwen3.5-2B (multimodal e
    com atenção híbrida, suporte a QLoRA menos testado), Llama-3.2-3B (licença exige "Llama" no nome
    de modelos derivados), Gemma-3-1B (notas multilíngues baixas no card), Tucano 2 (licença dos
    pesos não verificada).
- **Como extrair o rótulo (escolhido por Pedro):**
  - **Principal — pontuar os rótulos:** para cada avaliação, soma-se a log-probabilidade dos tokens
    de cada rótulo como resposta do assistente e fica o maior. Toda previsão é válida, é
    determinística e mede só a classificação, não a obediência ao formato. O modelo ajustado da
    Etapa 5 será medido do mesmo jeito.
  - **Secundário (só dev) — gerar e interpretar:** `parse_label` aceita variações ("Negativa.",
    "é neutra") e devolve "inválido" se a resposta não cita nenhum rótulo ou cita mais de um.
    Registra-se a taxa de respostas fora do formato.
  - Viés conhecido da pontuação: rótulos com mais tokens somam mais log-probabilidades negativas.
    Depois do primeiro token o resto da palavra costuma ser quase certo, então o efeito deve ser
    pequeno; não foi medido.
- **Prompts (escolhido por Pedro):** zero-shot e 6-shot (2 por classe).
  - Exemplos sorteados com seed 42 **do pool de treino da B2W** (nunca do dev ou do teste), só
    entre avaliações de até 280 caracteres, em ordem embaralhada; os mesmos 6 para todas as
    avaliações. O relatório guarda os índices no pool, não os textos.
  - Os rótulos do pool vêm das estrelas, então algum exemplo pode estar com rótulo errado.
- **Instrução:** define `neutro` incluindo avaliações com pontos positivos e negativos ao mesmo
  tempo. É a definição da tarefa (D6), não um vazamento do teste.
- **Modo *thinking* desligado:** não serve para pontuar rótulos e multiplica o custo.
- **Teste:** como na TF-IDF, a configuração (zero-shot ou 6-shot) é escolhida no dev; só a escolhida
  gera previsões do teste, sem métricas (D12).
- **Precisão numérica:** a T4 não suporta bf16, então a inferência roda em fp16; problemas
  numéricos são um risco a medir, não a supor. O notebook para se alguma pontuação não for finita.
- **Implementação da pontuação (escolhida por Pedro): simples, 3 passadas por avaliação**
  (prompt + cada rótulo). Alternativa: reaproveitar o KV cache do prompt para os 3 rótulos
  (~1/3 do tempo, estimativa não medida), descartada por ter mais código e mais pontos de erro.
  Duas otimizações sem mudar o resultado: a camada de saída só é aplicada nas posições que preveem
  tokens dos rótulos, e o notebook confere em 4 avaliações que esse atalho bate com o cálculo
  direto (`model(...).logits`).
- **Reprodutibilidade do prompt:** o chat template do SmolLM3 escreve a data de hoje no prompt de
  sistema; a flag `/system_override` do próprio template remove isso, deixando só a instrução.
- **Revisão fixada:** `a07cc9a04f16550a088caea529712d1d335b0ac1` (pesos em safetensors).
- **Resultados (medidos, dev, 2.000 avaliações; `reports/smollm3_base_dev.json`):**
  - Execução: Kaggle, Tesla T4, fp16, commit `8b1a763`. As checagens passaram: atalho = cálculo
    direto e nenhuma pontuação não finita, então o fp16 funcionou com este modelo.
  - Tempo: pontuar o dev levou 340 s (zero-shot) e 1.038 s (6-shot); gerar, 137 s e 398 s;
    pontuar o teste, 1.377 s. Prompt mediano: 158 tokens (zero-shot) e 480 (6-shot).

    | Configuração | Macro-F1 (IC 95%) | Acurácia | F1 neutro |
    |---|---|---|---|
    | Zero-shot, pontuação | **0,815** (0,795–0,833) | 0,851 | 0,638 |
    | 6-shot, pontuação | 0,760 (0,740–0,778) | 0,801 | 0,535 |
    | Zero-shot, geração | 0,816 | 0,851 | 0,640 |
    | 6-shot, geração | 0,760 | 0,800 | 0,536 |

  - **Escolhido: zero-shot.** O 6-shot foi **pior** (−0,055, IC −0,073 a −0,038): resultado
    negativo registrado. Ele empurrou previsões para `negativo` (neutros previstos como negativo:
    168 → 218) e tirou positivos do `positivo` (positivos previstos como neutro: 47 → 119).
    Hipóteses não testadas: os exemplos têm rótulos de estrelas (ruidosos) ou o efeito é deste
    conjunto específico de 6 exemplos; só um sorteio foi avaliado.
  - **Geração:** 0% de respostas inválidas e 99,85% de concordância com a pontuação. Só 2 respostas
    (de 2.000) continuaram depois do rótulo, e o rótulo inicial ainda foi lido corretamente.
  - **Contra as baselines (bootstrap pareado, zero-shot):** +0,054 sobre a melhor TF-IDF
    (IC +0,035 a +0,074) e **+0,028 sobre o BERTimbau ajustado** (IC +0,007 a +0,050,
    `p_b_not_better` 0,004).
  - Erros: o principal são 168 neutros previstos como negativo; nenhum erro entre positivo e
    negativo. O BERTimbau erra mais o contrário (positivos previstos como neutro). Os dois
    concordam em 82% das previsões; 170 avaliações são erradas pelos dois.
  - **Cuidados:**
    - o prompt foi escolhido no dev, então o 0,815 é otimista, como os das baselines;
    - **possível contaminação:** a B2W está pública no GitHub desde 2018 e o RePro no Hugging Face;
      se estiverem nos dados de pré-treino do SmolLM3, o zero-shot fica inflado. Não foi verificado.
    - Hipótese (não testada) para o LLM sem ajuste superar os modelos treinados: TF-IDF e BERTimbau
      aprenderam com rótulos de estrelas, que concordam com os humanos em só 74%; o LLM segue a
      definição da instrução, mais próxima da anotação humana do RePro.

## D16 — Etapa 5: QLoRA com rótulos de estrelas (A) e com rótulos humanos (B)

- **Data:** 2026-11-10 (Etapa 5)
- **Motivo para dois treinos:** o ruído das estrelas é sistemático, não aleatório. Medido no dev:

  | Estrelas → rótulo | Humano: negativo | neutro | positivo | Acerto |
  |---|---|---|---|---|
  | 1–2 → negativo | 619 | 193 | 9 | 75% |
  | 3 → neutro | 66 | 184 | 132 | **48%** |
  | 4–5 → positivo | 5 | 108 | 684 | 86% |

  Avaliações de 3 estrelas são positivas para os humanos em 35% dos casos, e o maior erro do
  BERTimbau (treinado com estrelas) foi justamente positivo → neutro. Inversões negativo ↔ positivo
  são raras (14). O RePro sorteou ~2.000 avaliações por nota, então essas taxas valem para uma
  amostra equilibrada por estrelas, não para o pool.
- **Decisão (escolhida por Pedro): treinar os dois e comparar** — "muitos rótulos ruidosos ou
  poucos rótulos limpos?"
  - **A (estrelas):** 15.000 avaliações do pool, 5.000 por classe, 1 época. Mesma fonte das
    baselines, então a comparação com elas é direta.
  - **B (humanos):** 1.500 avaliações do dev (`split_dev`, estratificado pelos 4 grupos, seed 42),
    todas, sem balancear (364 neutros, dos quais 302 mistos). Número de épocas decidido depois
    do piloto de velocidade.
  - Alternativa descartada: filtrar o pool pelo próprio SmolLM3 (circular).
- **Seleção e comparação no dev:** nas **500 avaliações restantes** do dev (206 positivo,
  173 negativo, 101 misto, 20 neutro puro). TF-IDF, BERTimbau e zero-shot são recalculados nessas
  mesmas linhas, a partir das previsões salvas. Custo: IC cerca de 2× mais largo (~±0,04) e só 20
  neutros puros, então o recall do neutro puro nesse conjunto é muito instável. O teste (8.003)
  continua sendo a avaliação final.
- **Formato:** o mesmo prompt do zero-shot (Etapa 4) + rótulo e fim de turno como resposta. A perda
  é só nos tokens da resposta (`build_example`); o prompt e o padding ficam com -100.
- **QLoRA (escolhido por Pedro, padrão do artigo do QLoRA, não ajustado):** base em 4 bits NF4 com
  double quantization, cálculo em fp16 (T4); LoRA em todas as camadas lineares, r 16, alpha 32,
  dropout 0,05; lr 2e-4, warmup 3%, decaimento linear, batch efetivo 16.
- **Checkpoints (escolhido por Pedro):** avaliação nas 500 de seleção em 25%, 50%, 75% e 100% do
  treino; fica o melhor. O passo 0 é o zero-shot.
- **Piloto:** o notebook mede a velocidade em poucos passos antes do treino; se A passar de ~3 h,
  o tamanho é reduzido.
