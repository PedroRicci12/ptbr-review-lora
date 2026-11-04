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
