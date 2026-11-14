# SmolLM3-3B QLoRA — sentimento de avaliações de produtos em português do Brasil

> English version: [README.md](README.md).

Adapter LoRA para o [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) que classifica
avaliações de produtos em português do Brasil como **`negativo`**, **`neutro`** ou **`positivo`**
(`neutro` inclui avaliações mistas, com pontos positivos e negativos).
Foi treinado com QLoRA em 1.500 avaliações rotuladas por humanos.

Num conjunto de teste separado, com $test_n avaliações rotuladas por humanos, chega a
**macro-F1 $test_f1** (IC de 95%: $test_f1_ci), $h1_diff acima do mesmo modelo sem fine-tuning.

Código, preparação dos dados, todos os experimentos e os resultados negativos:
[github.com/PedroRicci12/ptbr-review-lora](https://github.com/PedroRicci12/ptbr-review-lora).

## Resultados no teste

$test_n avaliações do [RePro](https://huggingface.co/datasets/lucasnil/repro) com polaridade
rotulada por humanos, nunca usadas em treino nem em escolha de modelo, avaliadas **uma única vez**
depois de o protocolo ser commitado (commit `$protocol_commit`). Macro-F1 com IC de 95% por
bootstrap:

| Modelo | Macro-F1 (IC 95%) | Acurácia | F1 `neutro` |
|---|---|---|---|
$test_table

Comparações pré-registradas (bootstrap pareado, IC de $family_confidence% da diferença de macro-F1,
com correção de Bonferroni):

| | Comparação | Diferença (IC) | Veredito |
|---|---|---|---|
$hypotheses_table

O treino com 15.000 rótulos de estrelas não foi publicado: o fine-tuning com estrelas deixou o
modelo *pior* que o zero-shot (H2), e 1.500 rótulos humanos superaram 15.000 de estrelas (H3).

## Como usar

Precisa de uma GPU CUDA (o carregamento em 4 bits usa `bitsandbytes`); foi executado numa T4
gratuita do Kaggle. O adapter pontua os 3 rótulos como respostas possíveis e devolve o mais
provável, então a saída é sempre um rótulo válido:

```python
$usage
```

- **Formato da entrada:** título e texto da avaliação unidos como `"título. texto"` (só o texto
  quando não há título), exatamente como no treino.
- **Carregue o modelo base em 4 bits (NF4)**, como acima: os números foram medidos assim.
  Carregar em fp16 ou fundir o adapter muda as contas e não foi avaliado.
- Cada avaliação custa 3 passadas pelo modelo (uma por rótulo). Use lotes para grandes volumes.

## Treino

- **Dados:** 1.500 avaliações da parte de dev do RePro neste projeto (polaridade rotulada por
  humanos; "misto" vira `neutro`), $training_examples usadas depois de descartar $dropped_long com
  mais de $max_train_tokens tokens. Outras 500 do dev ficaram de fora para escolher o checkpoint.
- **Método:** QLoRA. Modelo base congelado em 4 bits (NF4, quantização dupla, cálculo em fp16);
  LoRA r=$lora_r, alpha=$lora_alpha, dropout $lora_dropout em todas as camadas lineares
  ($trainable_params M parâmetros treináveis).
- **Receita:** perda só nos tokens da resposta (rótulo + fim de turno); taxa de aprendizado
  $learning_rate com 3% de aquecimento e decaimento linear; lote efetivo $effective_batch;
  $epochs épocas ($total_steps passos). O melhor de 4 checkpoints nas 500 separadas foi o passo
  $best_step (macro-F1 de seleção $select_f1, IC $select_f1_ci).
- **Computação:** uma $gpu, $train_minutes min de treino; $versions.
  Revisão do modelo base `$model_revision`; código de treino no commit `$training_commit`.

## Ablações (parte de seleção do dev, 500 avaliações)

**Quantos rótulos humanos são necessários?** Subconjuntos aninhados e estratificados dos 1.500,
mesma receita:

| Rótulos humanos | Macro-F1 (IC 95%) |
|---|---|
$learning_curve_table

Com 250 rótulos já supera o zero-shot; o ganho achata depois de cerca de 500.

**Variância entre seeds:** os mesmos 1.500 rótulos com 3 seeds de treino deram $seed_f1s
(média $seed_mean, desvio-padrão $seed_std). O adapter publicado (seed 42) foi o mais baixo dos três.

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
