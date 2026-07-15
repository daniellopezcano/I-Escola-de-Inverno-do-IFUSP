# Course Manifest — I Escola de Inverno do IFUSP
# «Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos»
# Generated: 2026-07-15 | Status: authoritative for all downstream agents

---

## DECISIONS (flagged for human review)

| # | DECISION | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | NB2 primary dataset | MNIST subset (cached .npz, ~4k samples); Galaxy10-DECaLS as 🟣 optional variant | Master Plan §6.3 explicitly names MNIST as primary and Galaxy10 as optional; inverting for "astro flavor" risks runtime overruns and wifi dependency — the particle sandbox in Act 1 already provides physics flavor without data risk. |
| D2 | L04 internal block order | L04_B01 = halos (A&A 685 A37), L04_B02 = J-PAS SSDA (arXiv:2602.13902) + closing | Confirmed per Master Plan §0 design decision 4 and narrative chiasmus reasoning: L3→L4_B01 (instance segmentation recognition) and L2→L4_B02 (DA recognition), closing with J-PAS outlook (WEAVE, careers) as natural course finale. |
| D3 | NB1 class count | 4 classes (simplified from research notebook's 6) | Master Plan §6.2 states explicitly "6→4 classes (less visual clutter, keeps one rare class)"; 4 classes preserve the imbalance lesson while reducing visual and computational complexity. |
| D4 | InfoNCE/temperature material depth in L03_B01 | One slide in theory block + one 🟣 section in NB2 | Pedagogically safer: the discriminative (Weinberger-style) pull/push loss is the load-bearing concept connected to Day 4's halo paper; InfoNCE is enrichment, not a second foundation. Matches Master Plan §3 and §6.3. |
| D5 | Poll mechanism for ~130 students | Show of hands | No tech dependency, no registration barrier, works in any venue, instructor reads the room instantly; Mentimeter adds friction and a tech-failure risk. Each block retains 2 polls as show-of-hands. |
| D6 | NB1 dataset: source/target cloud displacement magnitude | Moderate shift (cluster centers displaced by ~2–3 std of source clusters) so that zero-shot failure is visually obvious but not total; exact value set in the config cell | Ensures the "confident failure" visual is dramatic without making target-only training trivially easy, matching the narrative of "B beats C for large K." |
| D7 | NB0 1D function | Damped sinusoid: y = A·sin(2πx/λ)·exp(-x/τ) + ε, with ε ~ N(0,σ²) | Master Plan §6.1 suggests "senoide amortecida, sabor físico"; this is the explicit parametric form for the notebook-builder. |
| D8 | Accepted NB runtime budget | < 3 minutes wall-clock end-to-end on free Colab CPU with PRETRAINED=True | Direct quote from Master Plan §6 shared conventions. |

---

## GLOBAL CONVENTIONS

### Language Policy
- ALL student-facing content: Portuguese (pt-BR). This covers: all Markdown cells in notebooks, all code comments that students will read, all student-facing sections of block files, and 00_INDEX.md.
- ALL instructor/meta content: English. This covers: [!instructor] callouts in block files, this manifest, build logs, coherence report, code-internal logic comments (not tutorial-style).
- Block files are DUAL-PURPOSE: student sections in pt-BR, instructor callout in English.

### Traffic-Light Cell Taxonomy (exact definitions)
| Tag | Symbol | Meaning |
|-----|--------|---------|
| 🟢 | Verde | Concept / narrative cell — explains an idea in plain language; no code execution required. Students read and listen. |
| 🔵 | Azul | Core code cell — the main implementation step for the session; every student (and the instructor) runs this. |
| 🟡 | Amarelo | Lightning-poll / reflection cell — a prompt for a show-of-hands or a moment of prediction before running the next cell; must always be paired with a reveal (the next 🔵 cell). |
| 🟣 | Roxo | "Para quem quer mais" — optional extension cell; advanced students self-serve without derailing pacing; instructor explicitly tells the room "vocês não precisam fazer agora." |

### "Mapa do Curso" Recurring Element
- Every block opens with a static course-map graphic (or slide) showing all 4 days and 8 blocks, with the CURRENT block highlighted.
- In notebook cells: a 🟢 Markdown cell at position 0 with the map reproduced as a Markdown table and the current block in **bold**.
- In block files: referenced in the [!instructor] callout under "Preparação" as the first bullet.
- Purpose: continuous orientation for a heterogeneous 130-person room.

### Per-Block Timing Envelope
- Total block: 40 minutes.
- Content: 36 minutes.
- Buffer (questions): 4 minutes.
- Chronograph tables in block files are mandatory and must match the Master Plan verbatim.

### File Naming
| Asset type | Pattern | Example |
|---|---|---|
| Notebook | `NN_slug.ipynb` | `00_caixa_de_ferramentas.ipynb` |
| Block file | `LXX_BYY.md` | `L01_B01.md` |
| Index | `00_INDEX.md` | — |
| Build log | `.dev/agents/work/build_logs/<name>.log` | `build_logs/00_caixa_de_ferramentas.log` |
| Cached data | `jax-examples/assets/<name>.npz` | `assets/mnist_4k.npz` |
| Pretrained checkpoints | `jax-examples/assets/<name>_<stage>.pkl` | `assets/nb2_encoder_epoch_late.pkl` |
| Static PNG fallbacks | `jax-examples/assets/<name>_fig_<N>.png` | `assets/nb1_decision_map_source.png` |

### Block Template Section Skeleton (verbatim — block-writers must follow exactly)

```
---
tags: [lecture-block, published]
lecture_number: 
block_number: 
duration_mins: 40
date: 
colab_badge: "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](COOLAB_LINK_HERE)"
---

# Bloco L{{lecture_number}}.B{{block_number}}: [Título do Conceito]

> [!instructor] 🛠️ Notas do Instrutor (Preparação)
> - **Tempo estimado**: 40 min (inclui 5-10 min para dúvidas)
> - **Pré-requisitos para esta aula**: 
> - **Armadilhas comuns dos alunos**: 
> - **Link do Notebook**: 

---

## 🎯 Objetivos de Aprendizagem
Ao final deste bloco, você será capaz de:
- [ ] 
- [ ] 

## 🧠 Intuição e Conceito-Chave
*(Explicação acessível, focada na intuição física e no fluxo de trabalho de ML, sem derivações matemáticas densas inicialmente.)*
- 

## ⚙️ Formulação e Conexão com a Física
*(Aqui entra a "carne" do conceito: a função de perda, a arquitetura ou o princípio físico, explicado de forma clara.)*
- 

## 🖼️ Visualização e Slides
*Para acompanhar esta seção, consulte os slides da aula. O conceito principal é ilustrado por:*
- [Descrever brevemente o diagrama ou figura que será mostrado no slide]

## 💻 Demonstração Prática
Este conceito é colocado em prática no seguinte notebook:
{{colab_badge}}

**O que observar no código:**
1. 
2. 

## 📝 Resumo e Próximos Passos
- **Ideia principal**: 
- **Próximo bloco**: Vamos conectar isso a [[L{{lecture_number}}_B{{block_number+1}}_Proximo_Tema]].

## 🔗 Referências
- [Título do Paper/Recurso](link) - *Breve nota sobre o que este recurso cobre deste tópico.*
```

### Notebook Runtime Budget
- Hard ceiling: < 3 minutes wall-clock end-to-end on free Google Colab CPU tier, with `PRETRAINED = True`.
- `PRETRAINED = True` must be the default. When True, all model training and expensive computations are skipped and pre-computed results (weights, figures) are loaded from `jax-examples/assets/`.
- Every cell that trains a model or generates a heavy computation must check `if not PRETRAINED:` before executing.
- Test command (for CI): `/home/dlopez/miniconda3/envs/WinterSchool/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 <notebook>.ipynb`

### Assets Conventions
- **Cache directory**: `jax-examples/assets/` (committed to the repo; no .gitignore exceptions needed for small files).
- **Data caches** (`.npz`): MNIST 4k subset (`mnist_4k.npz`, ~2 MB), Galaxy10 1k subset (`galaxy10_1k.npz`, optional), NB1 2D Gaussian mixture (`toy_2d_4class.npz`, <100 KB).
- **Pretrained checkpoints** (`.pkl` via `pickle` or JAX array serialization): one per notebook per training stage. Naming: `nb0_fcnn_params.pkl`, `nb1_encoder_source.pkl`, `nb1_encoder_ssda.pkl`, `nb2_encoder_epoch0.pkl`, `nb2_encoder_early.pkl`, `nb2_encoder_late.pkl`.
- **Static PNG fallbacks**: one PNG per key figure, named `<notebook_id>_fig<N>.png`. Committed. Used when `PRETRAINED=True` and live matplotlib rendering fails (venue wifi, Colab instability). Notebooks must include a fallback display pattern: `try: <live_plot>; except: display(Image('assets/<name>.png'))`.
- All assets must be producible by a single `build_assets.py` script (to be written by notebook-builder agents). That script must also run in < 10 min on a GPU.

---

## NOTEBOOK BRIEF: 00_caixa_de_ferramentas.ipynb

### Purpose and Block Served
NB0 serves L01_B02 ("A caixa de ferramentas: Python, Colab e JAX na prática") as an **instructor-driven guided demo**. Students watch; they receive the notebook afterwards. The notebook demystifies the full working environment (Colab, GitHub, hardware) and consolidates the L01_B01 introduction by building a from-scratch fully-connected neural network in JAX and fitting a noisy 1D function via gradient descent — the "hello world" of everything the course uses for three more days. The key pedagogical payoff is the overfit cell, which plants the seed of generalization that flowers in Day 2.

### Complete Cell-by-Cell Outline (from Master Plan §6.1, expanded)

| Cell # | Type | Tag | Content Spec | Key Functions / Plots |
|--------|------|-----|--------------|----------------------|
| 0 | Markdown | 🟢 | **Mapa do Curso** — reproduz o mapa dos 4 dias em tabela Markdown; destaca L01_B02 em negrito. Título do notebook, aviso de que é demo guiada. | — |
| 1 | Markdown | 🟢 | **O que é este ambiente** — Explica em pt-BR o que é Google Colab (navegador, servidor remoto, célula de código vs. texto), o que é um kernel, e onde o notebook vive (GitHub). Menção de 60s ao que é um repositório para quem nunca viu. | — |
| 2 | Code | 🔵 | **Hardware timing** — Célula de timing: multiplicação de matrizes grandes em CPU vs GPU usando `jax.numpy`. Imprime tempo de execução. A célula deve imprimir algo como "CPU: 2.3s / GPU: 0.04s" (valores ilustrativos dependem do hardware). | `jax.numpy.dot`, `time.time()`, print comparison |
| 3 | Markdown | 🟢 | **O stack Python científico** — Um parágrafo introduzindo NumPy como língua franca, matplotlib para visualização, e posicionando JAX como "NumPy que sabe derivar e roda em GPU." | — |
| 4 | Code | 🔵 | **NumPy essencial — célula 1** — Criação de arrays, dtype, shape. Broadcasting exemplificado com operação física (e.g., vetores de posição). | `np.array`, `.shape`, broadcasting demo |
| 5 | Code | 🔵 | **NumPy essencial — célula 2** — Slicing e indexação (1D e 2D). Um exemplo de slicing de um "espectro" (vetor de 54 fluxos — gancho para J-PAS, sem explicar ainda). | `arr[::2]`, fancy indexing |
| 6 | Code | 🔵 | **matplotlib em 1 célula** — Plota uma senoide simples com rótulos pt-BR. Demonstra `plt.figure`, `plt.plot`, `plt.xlabel`, `plt.ylabel`, `plt.title`, `plt.savefig`. | `matplotlib.pyplot` basics |
| 7 | Markdown | 🟢 | **JAX = "NumPy que sabe derivar"** — Os três superpoderes em prosa: `jax.numpy` (drop-in), `jax.grad` (autodiff exato), `jit`/`vmap` (velocidade). Analogia: "derivadas exatas de código arbitrário — é isto que torna o treino possível." | — |
| 8 | Code | 🔵 | **jax.numpy drop-in** — Refaz as operações NumPy das células 4–5 com `jnp`. Mostra que a API é idêntica. | `import jax.numpy as jnp` |
| 9 | Code | 🔵 | **jax.grad de x² e de um potencial** — Define `f = lambda x: x**2`, computa `jax.grad(f)(3.0)` → 6.0. Em seguida, define um potencial de Lennard-Jones simplificado (ou poço de potencial 1D) e mostra `jax.grad` funcionando. Punchline em comentário: "derivadas exatas de _qualquer_ função Python." | `jax.grad`, scalar functions |
| 10 | Code | 🟣 | **(Opcional) jit e vmap com micro-benchmark** — Demonstra aceleração com `@jax.jit` e paralelismo de batch com `jax.vmap`. Só executar se houver GPU. Inclui `%timeit`. | `jax.jit`, `jax.vmap` |
| 11 | Markdown | 🟢 | **O Exercício Central: regressão com FCNN do zero** — Anuncia o que vem a seguir. Explica a função alvo: y = A·sin(2πx/λ)·exp(-x/τ) + ε ("senoide amortecida — sabor físico"). Apresenta a estrutura da FCNN: params = lista de tuplas (W, b). | — |
| 12 | Code | 🔵 | **Dados** — Amostra 200 pontos x ~ Uniform(0, 4π); computa y = A·sin(2πx/λ)·exp(-x/τ) + ε. Plota dado ruidoso + curva verdadeira. Parâmetros sugeridos: A=1.5, λ=2, τ=6, σ_ε=0.15. | `jnp.linspace`, `jax.random.normal`, `plt.scatter` |
| 13 | Code | 🔵 | **MLP do zero** — Define `init_params(layer_sizes, key)` retornando lista de `(W, b)`. Define `forward(params, x)` em ~10 linhas usando `tanh` como ativação (loop sobre camadas, última sem ativação). Tamanho sugerido: [1, 32, 32, 1]. | `init_params()`, `forward()` |
| 14 | Code | 🔵 | **Perda MSE** — Define `mse_loss(params, x_batch, y_batch)` = média de (forward(params, x) - y)². | `mse_loss()` |
| 15 | Code | 🔵 | **Loop de treino** — Config cell: `PRETRAINED = True`. Se `not PRETRAINED`: roda o loop (1000 épocas, SGD explícito: `params = [(W - lr*dW, b - lr*db) for (W,b),(dW,db) in zip(params, grads)]`). Se `PRETRAINED`: carrega `assets/nb0_fcnn_params.pkl`. Imprime perda a cada 100 épocas. | `jax.grad(mse_loss)`, explicit SGD update, `pickle.load` |
| 16 | Code | 🔵 | **Figura-troféu** — Plota o ajuste em 4 painéis: épocas 0, 10, 100, 1000 (usa checkpoints intermediários, também salvos em assets/). Cada painel mostra a curva do modelo sobre os dados ruidosos e a curva verdadeira. Legenda e título em pt-BR. Se `PRETRAINED`: exibe `assets/nb0_fig_trophy.png` como fallback. | `plt.subplot`, checkpoint loading, static PNG fallback |
| 17 | Code | 🟡 | **Poll "o que acontece com rede grande?"** — Célula Markdown com a pergunta: "O que acontece se usarmos uma rede muito grande e treinarmos por muitas épocas?" + instrução: "Levante a mão antes de rodar a próxima célula." | — (poll prompt only) |
| 18 | Code | 🔵 | **Célula de overfitting** — Redefine a rede com [1, 128, 128, 128, 1], treina por 5000 épocas (ou carrega checkpoint `assets/nb0_overfit_params.pkl`). Plota: ajuste interpolando o ruído. Comentário em pt-BR: "a rede memorizou o ruído — não generalizou." Planta a semente: "na quarta-feira, veremos o que acontece quando a distribuição de teste é diferente da de treino." | overfit checkpoint, side-by-side plot |
| 19 | Markdown | 🟢 | **Mapa de vocabulário** — Tabela 3 colunas: O que fizemos | Jargão padrão | O que a aula de hoje ensinou. Linhas: função alvo → tarefa; ruído → distribuição dos dados; rede → modelo; forward() → inferência; mse_loss() → função de perda; jax.grad() → gradiente; SGD update → descida do gradiente; épocas → época; curva ajustada bem → generalização; interpolação do ruído → overfitting. | Markdown table, pt-BR |
| 20 | Code | 🟣 | **(Opcional) Exploração: ativação / largura / learning rate** — 3 mini-experimentos parametrizados: (a) trocar tanh por ReLU; (b) variar largura [1, N, N, 1] com N ∈ {4,16,64,256}; (c) variar lr. Cada um re-executa o treino e plota o resultado. | Parametrized training loop, comparison plots |
| 21 | Markdown | 🟡 | **Para casa** — Lista de 2–3 exercícios em pt-BR: (1) Ajustar outra função f(x) de sua escolha (ex: |sin(x)|, x³·exp(-x)); (2) Dividir os dados em treino/validação, plotar ambas as perdas ao longo das épocas e identificar o ponto de overfitting; (3) (🟣) Trocar SGD por momentum manual. | — |

### Data and Asset Requirements
- `jax-examples/assets/nb0_fcnn_params.pkl` — trained weights after 1000 epochs on the damped sinusoid regression task.
- `jax-examples/assets/nb0_overfit_params.pkl` — weights of the over-wide network after 5000 epochs.
- `jax-examples/assets/nb0_epoch0_params.pkl`, `nb0_epoch10_params.pkl`, `nb0_epoch100_params.pkl` — intermediate checkpoints for the trophy figure.
- `jax-examples/assets/nb0_fig_trophy.png` — pre-rendered 4-panel trophy figure (static PNG fallback).
- `jax-examples/assets/nb0_fig_overfit.png` — pre-rendered overfit figure (static PNG fallback).
- No external dataset downloads required.

### PRETRAINED Checkpoints to Produce During Build
- `nb0_epoch0_params.pkl`, `nb0_epoch10_params.pkl`, `nb0_epoch100_params.pkl`, `nb0_fcnn_params.pkl` (1000 epochs), `nb0_overfit_params.pkl` (5000 epochs, wide net).
- Produced by `build_assets.py`, committed to `jax-examples/assets/`.

### Shared Utils
- No external utils module required for NB0 (all functions defined inline to maximize pedagogical transparency).
- If a `utils/` module is created later: `utils/plotting.py` with `plot_fit_comparison(params_list, x, y_noisy, y_true, epochs)` may be reused by later notebooks.

### Explicit DON'Ts
- Do NOT import `optax`, `flax`, `torch`, or any optimizer library. Raw pytrees and explicit SGD only.
- Do NOT import from external repos. All code self-contained.
- Do NOT include t-SNE or UMAP (Day 3 material).
- Do NOT explain the encoder+head split or domain shift (Day 2 material). The overfitting cell plants the seed with one sentence only.
- Do NOT use `jax.jit` in the main training loop (keep magic visible; jit is 🟣 optional).
- The colab_badge URL in the YAML front matter must link to the actual repo path once committed.

### Acceptance Criteria
1. Runs end-to-end headless in < 3 min (CPU, `PRETRAINED=True`): `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 00_caixa_de_ferramentas.ipynb`.
2. All figures render (no blank axes, no FileNotFoundError).
3. All Markdown cells are in pt-BR.
4. The trophy figure (cell 16) shows clearly distinguishable fits at all 4 epochs.
5. The overfit cell (cell 18) produces a visibly wiggly interpolation.
6. Take-home exercises present in the last Markdown cell.
7. The vocabulary map (cell 19) is complete (all 10 rows).
8. No `import torch`, no `import optax`, no external repo imports.

---

## NOTEBOOK BRIEF: 01_domain_shift_toy.ipynb

### Purpose and Block Served
NB1 serves L02_B02 ("Mão na massa I: quebrar e consertar um classificador") as a **hands-on lab notebook**. It is a simplified JAX port of the instructor's `06_training_tools.ipynb` from the `JPAS_Domain_Adaptation` repo. It enacts the full domain-shift lifecycle in 2D where everything is visible: source training, confident failure on target, shift diagnosis without target labels, and the three-regime experiment (zero-shot / target-only / SSDA) that directly mirrors the J-PAS paper students will see on Day 4. Four classes (not 6) are used to reduce visual clutter while preserving one rare class for the imbalance lesson.

### Complete Cell-by-Cell Outline (from Master Plan §6.2, expanded)

| Cell # | Type | Tag | Content Spec | Key Functions / Plots |
|--------|------|-----|--------------|----------------------|
| 0 | Markdown | 🟢 | **Mapa do Curso** — mapa em Markdown, L02_B02 em negrito. Título, subtítulo "O ciclo de vida do domain shift em 4 atos." | — |
| 1 | Markdown | 🟢 | **Orientação narrativa** — Explica os 4 atos: (1) criar o universo de brinquedo; (2) treinar na fonte, quebrar no alvo; (3) diagnosticar o shift sem rótulos do alvo; (4) comparar os três regimes de adaptação. Menciona que o design experimental do Ato 4 é o mesmo do artigo do J-PAS (sexta-feira). | — |
| 2 | Code | ⚙️ | **Setup** — Importações (`jax`, `jax.numpy as jnp`, `jax.random`, `sklearn.metrics`, `matplotlib`). Config cell: `PRETRAINED = True`. Constante `N_CLASSES = 4`. Semente global `KEY = jax.random.PRNGKey(42)`. | Config cell, no heavy computation |
| 3 | Code | 🔵 | **ATO 1.1 — Universo de brinquedo: dados** — `generate_gmm_data(n_per_class, class_weights, means_source, means_target, cov, key)` → arrays (X, y, domain). 4 classes; pesos de classe desbalanceados (e.g., [0.5, 0.25, 0.15, 0.10]); médias do alvo = médias da fonte deslocadas por ~2 std em duas classes. Retorna `(X_source, y_source, X_target, y_target)`. | `generate_gmm_data()`, `jax.random.multivariate_normal` |
| 4 | Code | 🔵 | **ATO 1.2 — Visualização Source vs Target** — Dois subplots lado a lado: scatter 2D, cada classe com cor distinta. Source à esquerda, Target à direita. KDE ou elipses de contorno sobre os pontos. Legenda em pt-BR. Se `PRETRAINED`: exibe `assets/nb1_fig_gmm.png`. | `plt.scatter`, KDE contours, static PNG fallback |
| 5 | Markdown | 🟡 | **Poll Ato 1** — "Olhe as duas figuras: quais classes vão sofrer mais com o shift? Levante a mão." (A resposta: as classes cujas nuvens foram deslocadas, especialmente a classe rara.) | — |
| 6 | Markdown | 🟢 | **Arquitetura: Encoder + Head** — Explica a decomposição em pt-BR: o encoder mapeia dados 2D → espaço latente 2D; a cabeça classifica o espaço latente → 4 classes. Analogia: encoder = percepção, cabeça = vocabulário. | — |
| 7 | Code | 🔵 | **ATO 2.1 — Definição da arquitetura** — `init_encoder(key)` → params para MLP [2→32→32→2] com tanh. `init_head(key)` → params para MLP linear [2→4]. `forward_encoder(enc_params, x)` e `forward_head(head_params, z)`. Reutiliza o estilo de NB0 (params como lista de tuplas (W,b), forward em ~10 linhas). | `init_encoder()`, `init_head()`, `forward_encoder()`, `forward_head()` |
| 8 | Code | 🔵 | **ATO 2.2 — Cross-entropy ponderada** — `weighted_ce_loss(enc_params, head_params, X, y, class_weights)`. Uma célula com 3 linhas de comentário em pt-BR: "por que pesos ∝ 1/frequência — sem eles, o modelo ignora a classe rara." | `weighted_ce_loss()`, `jax.nn.log_softmax` |
| 9 | Code | 🔵 | **ATO 2.3 — Treino no Source** — Config check `if not PRETRAINED:` → loop de treino (~500 épocas, Adam-like com updates explícitos ou `optax.adam` — DECISION: use `optax.adam` here since NB0 showed raw SGD; the import must be declared in Setup). Else: carrega `assets/nb1_encoder_source.pkl` e `assets/nb1_head_source.pkl`. Imprime loss ao longo do tempo. | `optax.adam` (first notebook where optax is acceptable since NB0 already showed raw SGD), checkpoint load |
| 10 | Code | 🔵 | **ATO 2.4 — Figura-assinatura: Mapa de Decisão** — `plot_decision_map(enc_params, head_params, X_overlay, y_overlay, title)`: malha 2D de pontos classificados, regiões pintadas com cores de classe (matplotlib `pcolormesh`), pontos sobrepostos. Dois painéis: (a) Source apenas → lindo; (b) Target sobreposto → catástrofe visível. Se `PRETRAINED`: exibe `assets/nb1_fig_decision_source.png` e `assets/nb1_fig_decision_target.png`. | `plot_decision_map()`, `plt.pcolormesh`, static PNG fallbacks |
| 11 | Code | 🔵 | **ATO 2.5 — Matrizes de confusão e histograma de confiança** — Confusion matrix source vs target (usando `sklearn.metrics.confusion_matrix`). Histograma de confiança (max softmax probability) para predições corretas vs incorretas no target → "errado E confiante." Plots com títulos em pt-BR. | `sklearn.metrics.confusion_matrix`, `jax.nn.softmax`, confidence histogram |
| 12 | Markdown | 🟢 | **ATO 3 — Diagnosticar sem rótulos do alvo** — Explica a ideia: treinar um classificador binário Source-vs-Target. Se ele consegue separar, há shift. Nenhum rótulo de classe alvo necessário. | — |
| 13 | Code | 🔵 | **ATO 3.1 — Classificador de domínio** — `train_domain_classifier(X_source, X_target, key)` → MLP binário simples [2→16→1] com sigmoid. Plota AUC ROC. Imprime: "AUC = X.XX — se > 0.7, o shift é detectável." | `train_domain_classifier()`, `sklearn.metrics.roc_auc_score` |
| 14 | Code | 🔵 | **ATO 3.2 — Scatter latente Source vs Target** — `plot_latent_scatter(enc_params, X_source, y_source, X_target, y_target)` → scatter do espaço 2D latente colorido por domínio. Nuvens separadas = shift visível no latente. Se `PRETRAINED`: exibe `assets/nb1_fig_latent_shift.png`. | `plot_latent_scatter()`, static PNG fallback |
| 15 | Markdown | 🟢 | **ATO 4 — Os três regimes** — Explica os três cenários: (A) Zero-shot (modelo source aplicado direto); (B) Target-only com K rótulos, do zero; (C) SSDA: carregar pré-treinado, congelar a cabeça, adaptar apenas o encoder com K rótulos. Introduz o parâmetro `K`. | — |
| 16 | Code | 🔵 | **ATO 4.1 — Regime (A) Zero-shot** — Já está done (é o modelo treinado em ATO 2). Exibe mapas de decisão e matriz de confusão do modelo source aplicado ao target. | Reuse of `plot_decision_map()` |
| 17 | Code | 🔵 | **ATO 4.2 — Regime (B) Target-only com K rótulos** — Config: `K = 50`. `sample_k_labels(X_target, y_target, K, key)` → K exemplos de rótulos alvo (estratificados). Treina encoder+head do zero nos K exemplos. Se `PRETRAINED`: carrega `assets/nb1_encoder_targetonly.pkl`. Plota mapa de decisão + confusão. | `sample_k_labels()`, target-only training loop or checkpoint |
| 18 | Code | 🔵 | **ATO 4.3 — Regime (C) SSDA** — Carrega o encoder source pré-treinado. Congela a cabeça (head params não recebem gradiente — implementado filtrando o pytree de `optax`). Adapta apenas o encoder nos mesmos K rótulos alvo. Se `PRETRAINED`: carrega `assets/nb1_encoder_ssda.pkl`. Plota mapa de decisão + confusão. | Head-frozen fine-tuning, `jax.lax.stop_gradient` on head params, or separate optimizer scope |
| 19 | Code | 🔵 | **ATO 4.4 — Comparação: mapas de decisão dos 3 regimes** — 3 mapas de decisão lado a lado (A, B, C) com o mesmo target overlay. Título em pt-BR. Se `PRETRAINED`: exibe `assets/nb1_fig_comparison.png`. | 3-panel figure, static PNG fallback |
| 20 | Code | 🔵 | **ATO 4.5 — Figura-síntese: métrica × K** — Varre K ∈ {10, 25, 50, 100, 200} e plota macro-F1 (ou accuracy) para os 3 regimes. As curvas se cruzam: B supera C para K grande (mas C é melhor para K pequeno). Se `PRETRAINED`: exibe `assets/nb1_fig_k_sweep.png` (a varredura é pré-computada). | K-sweep loop or precomputed, `plt.plot` for 3 curves |
| 21 | Markdown | 🟡 | **Poll Ato 4** — "Por que congelamos a CABEÇA e não o encoder? (dica: sotaques) — Levante a mão com sua hipótese antes de ler a célula seguinte." | — |
| 22 | Markdown | 🟢 | **Resposta ao poll** — Explica: a cabeça aprendeu conceitos (o que é cada classe no espaço latente); o encoder aprendeu a percepção (como mapear dados para o espaço latente). Adaptar o encoder = ajustar a percepção ao novo domínio sem reaprender os conceitos. Analogia: "adaptar a um sotaque, não aprender o idioma de novo." | — |
| 23 | Code | 🔵 | **ATO 4.6 — Latente pós-SSDA** — Scatter do espaço latente após SSDA: as nuvens do alvo entraram nas regiões fixas da cabeça. Comparação lado a lado: latente antes vs. após SSDA. Se `PRETRAINED`: exibe `assets/nb1_fig_latent_ssda.png`. | Side-by-side latent scatter, static PNG fallback |
| 24 | Markdown | 🟢 | **Takeaway** — "Quebrar é fácil, falhar em silêncio é perigoso, adaptar é barato — se você souber o que congelar." Teaser: "sexta-feira: este experimento exato, com quasares de verdade." | — |
| 25 | Code | 🟣 | **(Opcional) Mapas de probabilidade por classe** — `plot_class_probability_maps(enc_params, head_params)` → 4 mapas de calor (um por classe), cada um mostrando P(classe=k|x) no plano. | `plot_class_probability_maps()` |
| 26 | Code | 🟣 | **(Opcional) Shift de prior** — Gera uma variante dos dados onde as frequências de classe (não as posições) mudam entre domínios. Repetem os 4 atos neste cenário. | Prior-shift data generator |
| 27 | Code | 🟣 | **(Opcional) Espiada no latente com t-SNE** — Aplica t-SNE ao espaço latente (versão 4D, por exemplo). "Gancho para amanhã: t-SNE é uma ferramenta de inspecção de embeddings — amanhã veremos por quê." | `sklearn.manifold.TSNE`, brief preview only |
| 28 | Markdown | 🟡 | **Para casa** — 3 exercícios em pt-BR: (1) Variar a magnitude do shift (o deslocamento das médias do alvo) e observar como muda a curva de K; (2) Varrer K até B vencer C — o que isso diz sobre orçamentos de rótulos? (3) (🟣) Implementar a variante de prior shift. | — |

### Data and Asset Requirements
- `jax-examples/assets/toy_2d_4class.npz` — the 2D Gaussian mixture data for source and target domains (generated once during build with fixed seed=42, committed).
- `jax-examples/assets/nb1_encoder_source.pkl` — encoder weights trained on source domain.
- `jax-examples/assets/nb1_head_source.pkl` — head weights trained on source domain.
- `jax-examples/assets/nb1_encoder_targetonly.pkl` — encoder weights trained target-only (K=50).
- `jax-examples/assets/nb1_head_targetonly.pkl` — head weights trained target-only (K=50).
- `jax-examples/assets/nb1_encoder_ssda.pkl` — encoder weights after SSDA adaptation (K=50).
- Static PNG fallbacks: `nb1_fig_gmm.png`, `nb1_fig_decision_source.png`, `nb1_fig_decision_target.png`, `nb1_fig_latent_shift.png`, `nb1_fig_comparison.png`, `nb1_fig_k_sweep.png`, `nb1_fig_latent_ssda.png`.

### PRETRAINED Checkpoints to Produce During Build
- Source model: encoder + head (trained on source 2D GMM, 4 classes, ~500 epochs).
- Target-only model: K=50 exemplars, from scratch.
- SSDA model: head frozen, encoder adapted with K=50 target exemplars.
- K-sweep: precomputed macro-F1 for K ∈ {10, 25, 50, 100, 200} for all 3 regimes.
- All produced by `build_assets.py`.

### Shared Utils
- `utils/plotting.py`: `plot_decision_map(enc_params, head_params, X, y, title, ax=None)`, `plot_latent_scatter(enc_params, X_source, y_source, X_target, y_target, ax=None)`, `plot_confusion_matrix(cm, class_names, ax=None)`.
- `utils/models.py`: `init_encoder(key, layer_sizes=[2,32,32,2])`, `init_head(key, in_dim=2, n_classes=4)`, `forward_encoder()`, `forward_head()`.
- `utils/data.py`: `generate_gmm_data(...)`, `sample_k_labels(...)`.
- These utils may be reused by NB2 for the encoder definition pattern.

### Explicit DON'Ts
- No `torch`, no external repo imports (no JPAS_Domain_Adaptation repo code — everything inline or in `utils/`).
- No t-SNE in the main flow (only in 🟣 cell 27 with an explicit "gancho para amanhã" label).
- Total training time across ALL cells must stay < 3 min on Colab CPU with `PRETRAINED=True`.
- Weighted CE explanation must be at most one paragraph + one cell — it is a supporting theme.
- Do NOT reveal the J-PAS paper results or the J-PAS architecture — that is Day 4's payoff.
- The K-sweep must be pre-computed (too slow for live execution).

### Acceptance Criteria
1. Runs end-to-end headless in < 3 min (CPU, `PRETRAINED=True`).
2. Decision map figure (cell 10) shows clearly visible catastrophic failure when target points are overlaid.
3. Confidence histogram (cell 11) shows high-confidence wrong predictions on target.
4. K-sweep figure (cell 20) shows curve crossings (C wins at small K, B wins at large K).
5. Latent scatter before/after SSDA (cells 14 and 23) shows visible alignment improvement.
6. All Markdown cells in pt-BR.
7. Take-home exercises present (cell 28).
8. No `import torch`, no external repo imports.
9. `PRETRAINED = True` is the top-of-notebook default.

---

## NOTEBOOK BRIEF: 02_contrastive_embeddings.ipynb

### Purpose and Block Served
NB2 serves L03_B02 ("Mão na massa II: esculpindo um espaço de embeddings") as a **hands-on lab notebook**. It enacts contrastive learning in three acts of increasing abstraction: first as a pure particle system relaxing under pull/push potentials (no network), then with a real encoder on MNIST, and finally harvesting the embedding via clustering and t-SNE/UMAP inspection. The complete embed-then-cluster pipeline of instance segmentation is demonstrated in miniature. Act 1 is the course's signature visual; Act 3 plants the bridge to Day 4's halo paper.

### Complete Cell-by-Cell Outline (from Master Plan §6.3, expanded)

| Cell # | Type | Tag | Content Spec | Key Functions / Plots |
|--------|------|-----|--------------|----------------------|
| 0 | Markdown | 🟢 | **Mapa do Curso** — mapa em Markdown, L03_B02 em negrito. Título. | — |
| 1 | Markdown | 🟢 | **A tese** — "Perda contrastiva = potencial de interação." Em pt-BR: explica que o notebook começa como simulação de dinâmica molecular, depois adiciona uma rede neural, e ao final percebemos que eram a mesma coisa. Apresenta os 3 atos. | — |
| 2 | Code | ⚙️ | **Setup** — Importações. `PRETRAINED = True`. Carrega dados: `if not PRETRAINED: <download>; else: np.load('assets/mnist_4k.npz')`. Sem downloads externos no fluxo padrão. | `np.load`, config cell |
| 3 | Markdown | 🟢 | **ATO 1 — Sandbox de partículas (sem rede neural)** — Explica: 200 pontos 2D com rótulos de instância (5 grupos). Os "pontos" são as partículas; queremos que membros do mesmo grupo se atraiam e grupos diferentes se repilam. | — |
| 4 | Code | 🔵 | **ATO 1.1 — Gerar pontos 2D** — `generate_particles(n_instances=5, n_per_instance=40, key)` → arrays de posições (200×2) e rótulos de instância (200,). Posições iniciais: grupos ainda misturados (começam próximos para que a separação seja visível). Plota o estado inicial com cores. | `generate_particles()`, initial scatter |
| 5 | Code | 🔵 | **ATO 1.2 — Potenciais em JAX** — Define as 3 perdas (em pt-BR nos comentários): `L_pull = mean(max(||μ_c - x_i|| - δ_pull, 0)²)` para pontos fora do raio pull do seu centro de cluster; `L_push = mean(max(δ_push - ||μ_c - μ_c'||, 0)²)` entre pares de centros de cluster distintos; `L_reg = mean(||μ_c||²)` para ancorar os centros perto da origem. Total: `L = L_pull + L_push + L_reg`. Implementa em ~20 linhas JAX puro; centros de cluster = médias por instância. | `compute_cluster_centers(x, labels)`, `weinberger_loss(x, labels, delta_pull, delta_push)` |
| 6 | Code | 🟡 | **Poll Ato 1** — "Antes de rodar: o que acontece com δ_push = 0? Levante a mão." Instrução: "A próxima célula vai revelar." | — |
| 7 | Code | 🔵 | **ATO 1.3 — Colapso trivial (δ_push = 0)** — Roda a relaxação com `δ_push = 0`: todos os pontos colapsam num único ponto. Plota o resultado. Comentário: "o mínimo trivial — sem repulsão, tudo colapsa." | Relaxation with δ_push=0, collapse visualization |
| 8 | Code | 🔵 | **ATO 1.4 — Relaxação animada (δ_push > 0)** — Roda a relaxação por N_steps iterações com `x ← x − η·jax.grad(L)(x)`. Se `PRETRAINED`: exibe `assets/nb2_fig_sandbox_final.png` e um GIF pré-renderizado `assets/nb2_sandbox_animation.gif` (ou sequência de frames). Config: `δ_pull = 0.5`, `δ_push = 1.5`, `η = 0.01`, `N_steps = 500`. Plota estado inicial, intermediário, e final. | `jax.grad(weinberger_loss)`, gradient descent on positions, animation or filmstrip |
| 9 | Code | 🟣 | **(Opcional) Versão softmax/temperatura** — Implementa a versão InfoNCE: pesos de Boltzmann sobre negativos (softmax com temperatura T sobre distâncias a todos os outros centros). `L_infoNCE = -log(exp(-d_pos/T) / sum(exp(-d_neg/T)))`. Mostra que para T→0 recupera o comportamento hard-negative do push. | `infoNCE_loss()`, temperature parameter |
| 10 | Markdown | 🟢 | **ATO 2 — Encoder real, dados públicos** — Explica a transição: os "pontos" do Ato 1 eram posições 2D otimizadas diretamente; agora, um encoder MLP mapeia dados reais (imagens MNIST 28×28 = 784-D) para um espaço 2D, e aplicamos a MESMA perda. Os rótulos de classe são os pares positivos. | — |
| 11 | Code | 🔵 | **ATO 2.1 — Carregar subset MNIST** — `load_mnist_subset('assets/mnist_4k.npz')` → `(X_train, y_train, X_test, y_test)` com ~4000 treino + 1000 teste. X: float32 [0,1], shape (N, 784). Plota uma grade 5×10 de exemplos. | `np.load`, grid visualization, 10 classes |
| 12 | Code | 🟣 | **(Opcional) Variante Galaxy10** — `load_galaxy10_subset('assets/galaxy10_1k.npz')` → mesma interface. "Se quiser rodar com dados astronômicos: a física é a mesma, as imagens são galáxias." | `np.load`, galaxy grid visualization |
| 13 | Code | 🔵 | **ATO 2.2 — Encoder MLP → embedding 2D** — `init_encoder_mnist(key)` → MLP [784→256→64→2] com ReLU (mais adequado para dados de imagem do que tanh). `forward_encoder_mnist(params, x)`. Usa a mesma estrutura de parâmetros (lista de (W,b)) de NB0 e NB1. | `init_encoder_mnist()`, `forward_encoder_mnist()` |
| 14 | Code | 🔵 | **ATO 2.3 — MESMA perda pull/push, rótulos como geradores de pares** — `discriminative_loss(enc_params, X_batch, y_batch, delta_pull, delta_push)`. Equivalente à perda do Ato 1, mas agora os pontos x_i = encoder(imagem_i). Comentário: "a perda é idêntica à do Ato 1 — a única diferença é que x_i agora é o output do encoder." | `discriminative_loss()` (same functional form as Ato 1) |
| 15 | Code | 🔵 | **ATO 2.4 — Filme da evolução do embedding** — Plota scatter do embedding em 3 momentos: época 0 (caos), época intermediária (início de agrupamento), época final (classes em nuvens separadas). Se `PRETRAINED`: carrega checkpoints `assets/nb2_encoder_epoch0.pkl`, `assets/nb2_encoder_early.pkl`, `assets/nb2_encoder_late.pkl` e exibe filmstrip `assets/nb2_fig_evolution.png` como fallback. | Scatter plots at 3 epochs, pretrained checkpoints, static PNG fallback |
| 16 | Code | 🟡 | **Poll Ato 2** — "Que classes se misturam? Faz sentido visualmente? (dica: pense nos dígitos 4/9, 3/8)" — Levante a mão. | — |
| 17 | Markdown | 🟢 | **Resposta ao poll** — "Os dígitos parecidos visualmente (4 e 9, 3 e 8) formam nuvens adjacentes no espaço latente — o encoder descobriu que eles compartilham estrutura. São as 'degenerescências físicas' da escrita." Gancho: "amanhã veremos que partículas do universo com história de formação similar fazem o mesmo." | — |
| 18 | Markdown | 🟢 | **ATO 3 — Colher o espaço: cluster + projeções** — Explica: o embedding 2D já está organizado; agora usamos clustering para recuperar as classes sem usar os rótulos na inferência. Depois, inspecionamos uma versão de alta dimensão (16-D) com t-SNE e UMAP. | — |
| 19 | Code | 🔵 | **ATO 3.1 — Clustering no embedding** — Aplica k-means (N=10) ao embedding 2D do teste. Calcula ARI (Adjusted Rand Index) entre clusters previstos e rótulos verdadeiros. Plota: (a) clusters coloridos pelo label verdadeiro; (b) clusters coloridos pelo label previsto pelo k-means. Comentário: "isto É segmentação de instâncias: embed, depois cluster — sem usar os rótulos na inferência." | `sklearn.cluster.KMeans`, `sklearn.metrics.adjusted_rand_score`, side-by-side scatter |
| 20 | Code | 🔵 | **ATO 3.2 — Encoder 16-D** — Re-treina (ou carrega `assets/nb2_encoder_16d_late.pkl`) uma versão do encoder com output 16-D em vez de 2-D. Necessário para t-SNE/UMAP (projetar 16-D → 2-D é mais informativo que projetar 2-D → 2-D). | `init_encoder_mnist_16d()`, `assets/nb2_encoder_16d_late.pkl` |
| 21 | Code | 🔵 | **ATO 3.3 — t-SNE com múltiplas perplexidades** — Aplica `sklearn.manifold.TSNE` com perplexidades [5, 30, 100] ao embedding 16-D (subsample para ≤ 1000 pontos). 3 figuras lado a lado. O MAPA MUDA. Comentário: "distâncias entre grupos não têm significado em t-SNE — aviso das constelações." | `sklearn.manifold.TSNE`, subsampling, 3-panel figure, static PNG fallback `nb2_fig_tsne.png` |
| 22 | Code | 🔵 | **ATO 3.4 — UMAP com múltiplos n_neighbors** — Aplica `umap.UMAP` com n_neighbors ∈ {5, 15, 50} ao mesmo embedding 16-D. 3 figuras. Compara com t-SNE. Se `PRETRAINED`: exibe `assets/nb2_fig_umap.png`. | `umap.UMAP`, 3-panel figure, static PNG fallback |
| 23 | Markdown | 🟢 | **Resumo: embed-then-cluster** — "Um bom espaço de embeddings transforma um problema sem função de perda num problema de clustering." Ponte explícita para o Dia 4: "amanhã: esta máquina exata prevendo onde nascem os halos de matéria escura." | — |
| 24 | Code | 🟣 | **(Opcional) mini-SimCLR: pares por augmentação + InfoNCE** — Demonstra a versão auto-supervisionada: pares positivos criados por augmentação de dados (ruído gaussiano + translação aleatória), sem usar rótulos. Perda InfoNCE. "Cada augmentação é uma declaração de invariância." Inclui instruções para rodar em casa. | `augment_mnist(x, key)`, `infoNCE_loss()` |
| 25 | Markdown | 🟡 | **Para casa** — 3 exercícios em pt-BR: (1) Variar δ_push e observar a geometria final do embedding no Ato 1; (2) Trocar k-means por DBSCAN no Ato 3 — o que muda?; (3) Rodar a variante Galaxy10 (se tiver acesso a GPU). | — |

### Data and Asset Requirements
- `jax-examples/assets/mnist_4k.npz` — MNIST subset (4000 train + 1000 test, float32 [0,1], labels int). ~2 MB. Generated once from `torchvision` or `tensorflow_datasets` during build, then committed. **No downloads at notebook runtime.**
- `jax-examples/assets/galaxy10_1k.npz` — Galaxy10-DECaLS subset (1000 samples, 10 classes). Optional, ~5 MB. If not producible in build, skip and mark 🟣 cell as "dados não disponíveis no ambiente padrão."
- `jax-examples/assets/nb2_encoder_epoch0.pkl`, `nb2_encoder_early.pkl`, `nb2_encoder_late.pkl` — 2-D encoder checkpoints at 3 training stages.
- `jax-examples/assets/nb2_encoder_16d_late.pkl` — 16-D encoder checkpoint.
- Static PNG fallbacks: `nb2_fig_sandbox_final.png`, `nb2_fig_evolution.png`, `nb2_fig_tsne.png`, `nb2_fig_umap.png`.
- Optional: `nb2_sandbox_animation.gif` (pre-rendered animation of particle relaxation).

### PRETRAINED Checkpoints to Produce During Build
- Sandbox relaxation: final particle positions for δ_push > 0 and δ_push = 0.
- MNIST encoder (2-D): checkpoints at epoch 0, epoch ~20 (early), epoch ~100 (late).
- MNIST encoder (16-D): checkpoint at epoch ~100 (late).
- All produced by `build_assets.py`.

### Shared Utils
- `utils/contrastive.py`: `weinberger_loss(x, labels, delta_pull, delta_push)`, `compute_cluster_centers(x, labels)`, `infoNCE_loss(z, labels, temperature)`.
- Reuse `utils/models.py` from NB1 for encoder pattern (add `init_encoder_mnist()` and `init_encoder_mnist_16d()`).
- `utils/data.py`: `load_mnist_subset(path)`, `load_galaxy10_subset(path)`, `generate_particles(n_instances, n_per_instance, key)`.

### Explicit DON'Ts
- No live dataset downloads at notebook runtime — cached .npz only.
- t-SNE and UMAP only in Act 3 (cells 21–22), never earlier. Do not preview t-SNE before the concept has been introduced in L03_B01.
- The mini-SimCLR (augmentation-based CL) belongs only in 🟣 cell 24 — the main flow uses supervised pairs (labels as pair-generators).
- Do NOT name-drop the Weinberger paper or the halo paper during this notebook — the recognition moment belongs to L04_B01.
- t-SNE must subsample to ≤ 1000 points to avoid timeout.
- UMAP requires `pip install umap-learn`; add this to the Setup cell with a version pin.

### Acceptance Criteria
1. Runs end-to-end headless in < 3 min (CPU, `PRETRAINED=True`).
2. Act 1 sandbox visualization (cell 8) shows clear cluster separation after relaxation.
3. Collapse visualization (cell 7) shows all points collapsing to one location.
4. Evolution filmstrip (cell 15) shows progression from chaos to organized clusters.
5. K-means ARI in cell 19 is > 0.7 (good cluster recovery from embedding).
6. t-SNE figures (cell 21) visibly change with perplexity.
7. All Markdown cells in pt-BR.
8. Take-home exercises present (cell 25).
9. No external downloads at runtime.
10. No t-SNE before cell 21.

---

## BLOCK BRIEF: L01_B01

**Block ID**: L01_B01
**pt-BR Title**: «Aprendizado de máquina e Física: o mapa do território»
**Type**: Theory / panorama (slides, 2 polls, zero code)
**Date**: Tuesday, July 21, 2026

### Core Intuition (EN)
Situate ML within the current paradigm of physics and society, define the course's central object (learned representations), present the 4-day map, and hand students a curated self-study path so the course is a launchpad, not a closed box.

### One-Line Takeaway (pt-BR)
«Aprender é escolher coordenadas — e este curso ensina a escolhê-las, adaptá-las e desconfiar delas.»

### Full Chronograph (verbatim from Master Plan §2)

| Time | Segment | What & how |
|------|---------|------------|
| 0–5' | Cold open: why now? | One striking slide-pair: (a) data volume of modern surveys (J-PAS/LSST-scale: more data per night than a human can inspect in a lifetime) vs. (b) 2024 Nobel Prize in Physics (Hopfield & Hinton) — "a física emprestou ferramentas ao ML; agora o ML devolve o favor." Poll #1: "quem já usou alguma ferramenta de ML (incluindo ChatGPT)?" — expect ~100%; punchline: "usar todos usam; este curso é sobre _entender_." |
| 5–12' | ML in the scientific method | Where ML sits in physics today: emulating expensive simulations, pattern discovery in data deluges, inverse problems, experiment control. Honest counterpoint slide: what ML does _not_ do (replace understanding; extrapolate reliably; certify its own errors) — frames the course's critical stance. Brief societal contextualization (medicine, climate, language models) kept to one slide. |
| 12–19' | The central concept: representations | The course's thesis in one visual: raw data are points in huge spaces with misleading distances; networks learn _coordinates where distance = meaning_. Physics anchor: choosing good variables (center-of-mass, normal modes) — "vocês já fazem isso; redes automatizam." Vocabulary planted (encoder, espaço latente, embedding) at picture level only — each returns with depth on Days 2–3. |
| 19–26' | The course map | The 4-day arc walked slowly, one slide per day, each with its guiding question and its "payoff figure" teaser (Day 2: a decision boundary failing on shifted data; Day 3: an embedding self-organizing; Day 4: real halo shapes + real quasar confusion matrices). Logistics: repo QR, notebook policy (Day 1 demo-only; Days 2–3 bring laptop _if you can_, not required), how the Markdown study guides work. |
| 26–33' | The self-study map (resources) | Curated, opinionated resource walkthrough (see §7 below): "se você só tiver 3 horas" → 3Blue1Brown NN series; "se você tiver um mês" → Andrew Ng; "se você quiser o livro" → Nielsen / Prince; "se você quiser física+ML" → Carleo et al. review. Each with one sentence on _what it's uniquely good for_. Poll #2: "quantos já assistiram algum vídeo do 3Blue1Brown?" (calibrates the room). |
| 33–36' | Takeaway + bridge | «Se você lembrar de uma coisa: aprender é escolher coordenadas — e este curso ensina a escolhê-las, adaptá-las e desconfiar delas.» Bridge: "depois do intervalo: as ferramentas concretas que vamos usar a semana inteira." |
| 36–40' | Buffer | Questions. |

### Content Inventory

**Concepts owned by this block:**
- The data-deluge framing (J-PAS/LSST scale)
- 2024 Nobel Prize context (Hopfield & Hinton: physics → ML, now ML → physics)
- ML in the scientific method: emulation, pattern discovery, inverse problems, control
- Honest counterpoint: what ML does NOT do
- The central thesis: learned representations = learned coordinates
- Physics anchor: normal modes, center-of-mass as analogies for good coordinates
- Vocabulary introduced (at picture level only): encoder, espaço latente, embedding
- The 4-day course map with guiding questions
- Logistics: repo, QR, notebook policy, study guides
- Curated resource list (§7 of Master Plan)

**Polls (show of hands):**
- Poll 1: "Quem já usou alguma ferramenta de ML (incluindo ChatGPT)?" — expected ~100%; punchline about understanding vs. using.
- Poll 2: "Quantos já assistiram algum vídeo do 3Blue1Brown?" — calibrates room's starting point.

**Figures / Visuals:**
- Slide 1a: data volume of modern surveys (one dramatic number: LSST ~20TB/night)
- Slide 1b: 2024 Nobel Prize announcement image
- Slide "ML no método científico": 4-cell diagram (simulation emulation / pattern discovery / inverse problems / experiment control)
- Slide "O que ML não faz": 3 bullets (honesty slide — maintains credibility)
- Slide "O conceito central": scatter in raw space (misleading) vs. scatter in learned space (organized) — one 2-panel visual
- Slide "O mapa do 4 dias": one slide per day with guiding question + teaser figure; these teasers are the payoff figures committed to assets/

**Resource list** (per Master Plan §7 — all in pt-BR blurbs; lives in this block's .md file):
- 3Blue1Brown "Neural Networks" series
- StatQuest (Josh Starmer)
- Welch Labs / Artem Kirsanov (🟣)
- Andrew Ng ML Specialization (Coursera, audit grátis)
- Andrew Ng DL Specialization (🟣)
- fast.ai Practical Deep Learning
- Nielsen — Neural Networks and Deep Learning (gratuito)
- Prince — Understanding Deep Learning 2023 (PDF grátis)
- Goodfellow, Bengio & Courville — Deep Learning (🟣)
- Carleo et al. 2019 Rev. Mod. Phys.
- Mehta et al. 2019 Phys. Rep.
- Lilian Weng blog — Contrastive Representation Learning (🟣)
- JAX 101 tutorial
- Google Colab FAQ
- Both case study repos: `daniellopezcano/instance_halos` and `daniellopezcano/JPAS_Domain_Adaptation`

### Instructor Callout Content Spec
- **Prep notes**: Prepare the cold-open slide pair before the first class. Verify Nobel Prize date (Oct 2024). Book-end the resource walkthrough slide with the actual markdown file URL/QR so students can scan immediately.
- **Timing risks**: The societal contextualization slide can expand to 10 min if unchecked — timebox to 2 min (1 slide). The resource walkthrough can balloon — it's a tour, not a reading; each resource gets one spoken sentence.
- **Common pitfalls**: Do NOT teach NN mechanics (the separate intro lecture owns that); vocabulary is planted at picture level only in this block — definitions return with mathematical depth in Days 2–3; the honest counterpoint slide MUST appear to frame the course's critical stance.

### Cross-References to Other Blocks
- The "payoff figure teasers" for Days 2–4 must match the actual figures that appear in L02_B02, L03_B02, and L04_B01/B02 — coordination with notebook-builder agents required.
- The resource list lives in this block's .md file and is QR-linked from the slide deck.
- The vocabulary planted here (encoder, espaço latente, embedding) must be defined consistently across L02_B01, L03_B01, and L04 blocks.
- Bridge to L01_B02: "depois do intervalo: as ferramentas concretas."

---

## BLOCK BRIEF: L01_B02

**Block ID**: L01_B02
**pt-BR Title**: «A caixa de ferramentas: Python, Colab e JAX na prática»
**Type**: Guided demo (instructor executes NB0; students watch)
**Date**: Tuesday, July 21, 2026
**Anchor notebook**: `00_caixa_de_ferramentas.ipynb`

### Core Intuition (EN)
Demystify the full working environment (Colab, GitHub, hardware, JAX) and consolidate L01_B01's NN concepts by building a from-scratch FCNN in JAX and fitting a noisy 1D function with gradient descent — the minimal "hello world" of everything that follows.

### One-Line Takeaway (pt-BR)
«Treinar uma rede = descer o gradiente de uma função de perda; todo o resto é engenharia em volta disso.»

### Full Chronograph (verbatim from Master Plan §2)

| Time | Segment | What & how |
|------|---------|------------|
| 0–6' | The environment tour | Live: open Colab from the repo badge; what a notebook _is_ (cells, kernel, state); where this lives (GitHub — 60-second "what is a repo" for non-coders); Runtime menu: CPU vs GPU vs TPU with a one-slide mental model ("CPU = poucos doutores; GPU = milhares de estagiários fazendo a mesma conta") and a live timing cell (matrix multiply on CPU vs GPU). |
| 6–12' | The scientific Python stack in 5 cells | `numpy` arrays as the lingua franca; `matplotlib` in one cell; then **JAX**: "NumPy que sabe derivar e que corre em GPU." The three superpowers each in one line: `jax.numpy` (drop-in), `jax.grad` (autodiff), `jit/vmap` (speed). Demo: `jax.grad` of x² and of a hand-written potential — "derivadas exatas de código arbitrário; é isto que torna o treino possível." |
| 12–30' | The core exercise: FCNN from scratch fits a noisy function | The heart of the block, built live cell by cell (all pre-tested; PRETRAINED fallback exists): (1) sample y = f(x) + ruído (f = damped sine or similar physics-flavored curve); (2) define an MLP _from scratch_ — params as a list of (W, b), forward pass in ~10 lines with `tanh`; (3) MSE loss; (4) the training loop: `grads = jax.grad(loss)(params)` + explicit SGD update — **no optimizer library**, so students see backprop is "just" the chain rule handled by autodiff plus a descent step; (5) the money plot: animated/looped fit snapshots at epochs 0, 10, 100, 1000 over the noisy data. Polls embedded: "o que acontece se a rede for grande demais e treinarmos para sempre?" → run the overfit cell → the wiggly interpolation of noise → first encounter with **generalization**, the seed concept for Day 2. |
| 30–36' | Vocabulary consolidation map | One slide mapping what was just done to the standard jargon (modelo/parâmetros/perda/gradiente/época/treino vs. validação) and to what the intro lecture taught. Explicit pointers: "quarta-feira este mesmo loop vira um classificador; quinta, uma perda contrastiva." Takeaway: «treinar uma rede = descer o gradiente de uma função de perda; todo o resto é engenharia em volta disso.» |
| 36–40' | Buffer | Questions; announce that NB0 (with extended comments + exercises) is now in the repo. |

### Content Inventory

**Concepts owned by this block:**
- Colab environment: cells, kernel, state, GitHub 60-sec intro
- CPU vs GPU mental model ("doutores vs estagiários")
- NumPy as lingua franca (arrays, slicing, broadcasting)
- matplotlib basics
- JAX: the three superpowers (jnp drop-in, `jax.grad`, `jit`/`vmap`)
- Autodiff as "chain rule automated by code"
- MLP architecture: params as list of (W,b), forward in ~10 lines, tanh activation
- MSE loss
- Training loop: explicit SGD, no optimizer library
- The trophy figure: fit at epochs 0/10/100/1000
- Overfitting as memorization of noise
- Generalization: the seed concept for Day 2
- Vocabulary map: modelo, parâmetros, perda, gradiente, época, treino vs. validação

**For hands-on/demo blocks: notebook it wraps**
- Wraps `00_caixa_de_ferramentas.ipynb`. Block-writer must read the FINAL executed notebook and reference its actual cells (cell numbers, figure outputs, exact pt-BR text).
- The block-writer should reference specifically: the hardware timing cell, the jax.grad demo cell, the trophy figure cell, the overfit cell, and the vocabulary map cell.

**Polls (embedded in the notebook walkthrough):**
- "O que acontece se a rede for grande demais e treinarmos para sempre?" — embedded as a 🟡 cell before the overfit demo.

### Instructor Callout Content Spec
- **Prep notes**: Pre-run the GPU timing cell BEFORE class (Colab GPU allocation lags). Have static screenshots of EVERY cell output as backup against wifi failure. Test the repo badge link the day before.
- **Timing risks**: The environment tour can silently consume 15 min — timebox to 6 min hard. The vocabulary map at the end is a slide, not more code.
- **Common pitfalls**: Do NOT introduce `optax`/`flax` (raw pytrees keep the magic visible for NB0). Do NOT open the t-SNE cells. Do NOT start Day 2 material.

### Cross-References to Other Blocks
- Bridge FROM L01_B01: vocabulary (encoder, espaço latente) is placed here in working context.
- The from-scratch training loop style (params as pytrees, explicit SGD) is the exact style reused in NB1 (for the encoder) and NB2 (for the particle relaxation).
- The overfitting cell plants the seed that germinates in L02_B01.
- "Quarta-feira este mesmo loop vira um classificador; quinta, uma perda contrastiva" — these forward-pointers must be echoed in L02_B02 and L03_B02 respectively.

---

## BLOCK BRIEF: L02_B01

**Block ID**: L02_B01
**pt-BR Title**: «Mudança de domínio: quando o treino não é a prova»
**Type**: Theory (slides, 2 polls, zero code)
**Date**: Wednesday, July 22, 2026

### Core Intuition (EN)
Standard ML silently assumes train ≈ test; reality (and especially sim-trained science) breaks this constantly, and models fail confidently. Give students the taxonomy, the detection habits, and the map of mitigation strategies — the vocabulary and mindset to recognize when these tools apply in their careers.

### One-Line Takeaway (pt-BR)
«Desconfie do softmax: confiança não é competência — e adaptar é barato se você souber o que congelar.»

### Full Chronograph (verbatim from Master Plan §3)

| Time | Segment | What & how |
|------|---------|------------|
| 0–4' | Mapa + recap | Highlight Day 2; 90-sec recap of NB0's overfitting cell: "generalizar _dentro_ da mesma distribuição já era difícil; hoje: e quando a distribuição muda?" |
| 4–11' | The phenomenon, everywhere | The i.i.d. assumption exposed with universal examples first: speech models meeting the Paulistano accent; a hospital's diagnostic model failing on another hospital's scanner; sunny-simulator self-driving meeting fog. Then science: simulation-trained models meeting real instruments. The cultural anchor analogy: **«estudar pelo simulado, fazer a prova de verdade»** (ENEM/vestibular). Poll: "quem já treinou/testou algo em dados de origens diferentes?" |
| 11–18' | Taxonomy with cartoons | Three named shifts, each ONE 2D scatter cartoon: **covariate shift** (p(x) moves, rule intact), **prior/label shift** (class frequencies change), **concept shift** (the rule itself changes). Vocabulary stays on-screen all block. Sim-to-real framed as the scientist's chronic case: simulations = free labels + imperfect physics; observations = real physics + scarce labels. |
| 18–24' | Silent failure & calibration | The scariest slide: a confidently wrong softmax under shift. **«Errado e confiante é o modo de falha mais perigoso da ciência com ML.»** Analogy: **termômetro descalibrado** (precise ≠ accurate). Reliability diagram introduced visually; calibration named as a property distinct from accuracy. |
| 24–32' | The mitigation map | The strategy panorama, organized by _what you have_: (i) nothing but source → data augmentation & domain randomization; (ii) unlabeled target → distribution alignment (adversarial alignment at cartoon level) and importance reweighting; (iii) **a few target labels → transfer learning / fine-tuning / SSDA**: pretrain-then-adapt, what to freeze vs. retrain, early-layers-generic vs. late-layers-specific. The **encoder + head** decomposition introduced HERE: analogy — **sotaques**: adapting to a new accent recalibrates perception (encoder) without relearning vocabulary (head). |
| 32–36' | Detection habits + takeaway | Practical checklist: compare feature histograms; train a source-vs-target _domain classifier_; monitor embedding drift. Takeaway: «desconfie do softmax: confiança não é competência — e adaptar é barato se você souber o que congelar.» Teaser: "no próximo bloco vamos quebrar um classificador de propósito e consertá-lo com exatamente estas ideias." |
| 36–40' | Buffer | Questions. |

### Content Inventory

**Concepts owned by this block:**
- i.i.d. assumption and why it is violated
- Universal domain-shift examples (speech, medicine, self-driving, sim-to-real)
- Cultural analogy: "estudar pelo simulado, fazer a prova de verdade" (ENEM/vestibular)
- Three-way shift taxonomy: covariate / prior / concept shift (each with 2D cartoon)
- Silent failure: confidently wrong softmax
- "Errado e confiante é o modo de falha mais perigoso da ciência com ML"
- Calibration: precise ≠ accurate; reliability diagram; termômetro descalibrado analogy
- Mitigation map organized by what you have (source-only / unlabeled-target / few-target-labels)
- Encoder + head decomposition (introduced here for the first time)
- Sotaques analogy for adapting encoder not head
- Practical detection checklist

**Polls (show of hands):**
- "Quem já treinou/testou algo em dados de origens diferentes?"

**Figures / Visuals:**
- Slide: data volume vs. Nobel Prize (reused from L01_B01 as mapa do curso frame)
- Slide: i.i.d. assumption → 3 universal non-physics examples → 1 science example
- 3 cartoon slides: one 2D scatter per shift type (covariate / prior / concept)
- Slide: confidently wrong softmax — bar chart of probabilities on a misclassified example
- Slide: reliability diagram (calibration visualization)
- Slide: mitigation map — table/flowchart organized by available supervision
- Slide: encoder + head decomposition diagram (the first appearance of this architecture)

### Instructor Callout Content Spec
- **Prep notes**: Prepare the 3 cartoon scatter plots (each must be clean and minimal — one concept per slide). The encoder+head diagram must be exactly the same visual used in NB1 (coordination required with notebook-builder).
- **Timing risks**: The mitigation map can expand to 15 min if presented as detail — it must remain a MAP (one sentence per strategy, drill-down happens in NB1 and Day 4).
- **Common pitfalls**: No equations beyond p_source(x,y) ≠ p_target(x,y). Resist telling the J-PAS story now (Day 4's payoff depends on it staying fresh). The encoder+head analogy with sotaques is introduced here but only deepened in NB1.

### Cross-References to Other Blocks
- Picks up from L01_B02's overfitting seed: "generalizar dentro da mesma distribuição já era difícil."
- The encoder+head decomposition introduced here is the SAME architecture used in NB1 (L02_B02) and referenced in L04_B02.
- The "sotaques" analogy introduced here is echoed in NB1's poll answer (cell 22) and L04_B02.
- The confidently-wrong-softmax visual must match the histograma de confiança in NB1 (cell 11).
- Teaser to L02_B02: "no próximo bloco vamos quebrar um classificador de propósito."

---

## BLOCK BRIEF: L02_B02

**Block ID**: L02_B02
**pt-BR Title**: «Mão na massa I: quebrar e consertar um classificador»
**Type**: Hands-on (instructor executes NB1; students with laptops follow via badge)
**Date**: Wednesday, July 22, 2026
**Anchor notebook**: `01_domain_shift_toy.ipynb`

### Core Intuition (EN)
The full domain-shift lifecycle in 2D where everything is visible: train on source, watch confident failure on target, then compare the three regimes (zero-shot / target-only / freeze-head-adapt-encoder) — the exact experimental design of Friday's J-PAS paper, in a toy.

### One-Line Takeaway (pt-BR)
«Quebrar é fácil, falhar em silêncio é perigoso, adaptar é barato — se você souber o que congelar.»

### Full Chronograph (verbatim from Master Plan §3)

| Time | Segment | What & how |
|------|---------|------------|
| 0–3' | Setup + framing | Run-all immediately; frame: "hoje o notebook é um laboratório de patologia: causar a doença, diagnosticar, tratar." Traffic-light cell convention reminder (🟢 conceito / 🔵 código / 🟡 pergunta / 🟣 opcional). |
| 3–10' | Act 1 — the toy universe | Generate the 2D Gaussian-mixture data: 6 classes, **imbalanced** proportions [NOTE: Master Plan says 6 classes in the chronograph here but §6.2 specifies 4 classes; per DECISION D3, use 4 classes — the chronograph text is from an earlier draft], and two domains — _source_ and _target_ — where some class clouds moved (visualize side by side; students literally SEE covariate shift, and the imbalance motivates the weighted cross-entropy, shown in one cell). 🟡 "olhe as duas figuras: quais classes vão sofrer mais?" |
| 10–17' | Act 2 — train on source, break on target | Encoder (MLP → 2D latent) + head (→ 4 classes [D3]), reusing NB0's from-scratch style. Train on source (fast); show the **decision-map** figure with source points → beautiful; overlay target points → visible catastrophe. Confusion matrix source vs. target side by side. Confidence histogram: **still confident while wrong** (yesterday's warning, now on screen). |
| 17–23' | Act 3 — diagnose | Train the 2-line domain classifier (source vs target): high AUC = shift detected without any target labels. Latent-space scatter of source vs target: the clouds don't overlap — "o shift é visível no espaço latente." |
| 23–32' | Act 4 — the three-regime experiment | **(A) zero-shot** (source model applied raw); **(B) target-only** trained from scratch on K target labels; **(C) SSDA**: load the pretrained model, **freeze the head, adapt only the encoder** on the same K labels. Show the decision maps + confusion matrices of the three; the summary figure: accuracy/macro-F1 vs. K for the three regimes (curves cross!). 🟡 "por que congelar a CABEÇA e não o encoder? (dica: sotaques)". Latent scatter after SSDA: target clouds moved into the head's fixed class regions. |
| 32–36' | Takeaway | «Quebrar é fácil, falhar em silêncio é perigoso, adaptar é barato — se você souber o que congelar.» Teaser: "sexta-feira: este experimento exato, com quasares de verdade." |
| 36–40' | Buffer | Questions. |

### Content Inventory

**For hands-on/demo blocks — notebook this wraps:**
This block wraps `01_domain_shift_toy.ipynb`. Block-writer must read the FINAL executed notebook and reference its actual cells. Key cells to reference: cell 4 (GMM visualization), cell 10 (decision map catastrophe), cell 11 (confidence histogram), cell 13 (domain classifier AUC), cell 14 (latent scatter), cells 16–18 (three regimes), cell 20 (K-sweep), cell 23 (latent post-SSDA).

**Concepts owned by this block:**
- Traffic-light cell convention explained to students
- Covariate shift made visible in 2D
- Imbalanced classes + weighted cross-entropy (brief — supporting theme only)
- Decision map as the signature visualization
- Silent confident failure (histograma de confiança)
- Domain classifier as shift detection without labels
- The three-regime experiment (zero-shot / target-only / SSDA)
- Freeze-head adapt-encoder: sotaques analogy confirmed live

**Polls (embedded in notebook):**
- 🟡 "Quais classes vão sofrer mais com o shift?" (cell 5)
- 🟡 "Por que congelar a CABEÇA e não o encoder?" (cell 21)

### Instructor Callout Content Spec
- **Prep notes**: Run-all must complete in < 3 min before the session. Verify that PRETRAINED=True is set. Have the repo badge open in a browser tab before class. Ensure the K-sweep figure is pre-rendered.
- **Timing risks**: Total training time across all cells must stay < 3 min on Colab CPU (pretrained fallbacks mandatory). Act 4 (three-regime) is the centerpiece — budget at least 9 min for it.
- **Common pitfalls**: Do not open the t-SNE cells (🟣 cell 27) — t-SNE is Day 3's material. Keep the weighted-CE explanation to one sentence + one cell. Do NOT reveal J-PAS results.

### Cross-References to Other Blocks
- Picks up from L02_B01's encoder+head diagram and sotaques analogy.
- The three-regime experiment design directly mirrors L04_B02's J-PAS paper — say this explicitly: "sexta-feira: este experimento exato, com quasares de verdade."
- The 🟣 t-SNE cell is explicitly flagged as "gancho para amanhã" (L03_B01/L03_B02).
- Latent scatter visualization previews the embedding concept deepened in L03_B01.

---

## BLOCK BRIEF: L03_B01

**Block ID**: L03_B01
**pt-BR Title**: «Aprendizagem contrastiva: geometria da similaridade»
**Type**: Theory (slides, 2 polls, zero code)
**Date**: Thursday, July 23, 2026

### Core Intuition (EN)
One coherent narrative from "distances in raw data lie" to "define similarity, geometrize it": embeddings → contrastive losses as interaction potentials → where positive pairs come from → clustering the result → inspecting it with t-SNE/UMAP → and the payoff application: instance segmentation.

### One-Line Takeaway (pt-BR)
«Similaridade não se descobre — se define; a rede só geometriza a sua definição.»

### Full Chronograph (verbatim from Master Plan §4)

| Time | Segment | What & how |
|------|---------|------------|
| 0–4' | Mapa + the puzzle | Recap Day 2's latent scatters ("o espaço latente era o palco; hoje ele é o protagonista"). The opening puzzle: image of a digit, the same digit shifted 3 px, and a different digit — poll: "qual par está mais perto em distância de pixels?" → reveal: the shifted copy is FARTHER. «Distâncias cruas mentem; precisamos aprender a métrica.» |
| 4–10' | Embeddings & the label economy | Encoder as cartographer: map data into a space where distance = similarity of meaning. The economics: labels cost telescope time/expert hours; raw data is nearly free → the supervised/semi/self-supervised taxonomy as **budget strategies**. Self-supervision in one slide: the data supervises itself → "é assim que os grandes modelos de linguagem são treinados." |
| 10–18' | The contrastive principle + the physics of it ⭐ | **The peak of the course.** One-sentence definition: **escolha quais pares devem ficar perto (positivos) e quais devem ficar longe (negativos); a rede geometriza a sua escolha.** Contrastive losses as **interaction potentials**: positives attract like springs (with slack δ_pull), cluster centers repel like charges (with cutoff δ_push); training = overdamped relaxation of a particle system. Write the pull/push terms in physics notation. The degenerate-minimum exercise: "e se só houver atração?" → total collapse → why negatives exist. InfoNCE/SimCLR mentioned as softmax-temperature version in one slide (D4: one slide only). |
| 18–24' | Where do pairs come from? | Two answers: (i) **labels** → supervised metric learning (face-ID analogy); (ii) **augmentations** → two distorted views of the same object. Deep slide: **cada augmentação é uma declaração de invariância** — Noether-flavored. "Escolher augmentações é declarar as simetrias do seu problema" (PSF, sky orientation, calibration in astronomy). |
| 24–30' | Harvesting the embedding: clustering & projections | Clustering (k-means/density) finds groups — with the caveat that cluster quality = representation quality. **t-SNE & UMAP** as inspection tools: what they do, how they mislead (distances between clusters ≈ meaningless; perplexity changes the picture), the constelações warning. Optional unification (one slide): t-SNE and UMAP are themselves contrastive methods. |
| 30–36' | The payoff problem: instance segmentation | Definition ladder: semantic vs. instance vs. panoptic. Why instance is awkward: variable number + permutation-invariant labels → no direct differentiable loss → workaround: **embed-then-cluster**. "Vocês agora possuem todas as peças desta máquina." Takeaway. Teaser: "no próximo bloco, vamos VER um espaço se organizar em tempo real." |
| 36–40' | Buffer | Questions. |

### Content Inventory

**Concepts owned by this block:**
- The "raw distances lie" puzzle (pixels example)
- Encoder as cartographer
- Label economy taxonomy: supervised / semi-supervised / self-supervised as budget strategies
- Contrastive principle: choose positives and negatives; network geometrizes the choice
- Contrastive losses = interaction potentials (springs + repulsive charges, physics notation)
- Degenerate trivial minimum (collapse if no push)
- InfoNCE/SimCLR: softmax-temperature version (one slide, D4 decision)
- Where pairs come from: labels vs. augmentations
- Augmentations as invariance declarations (Noether-flavored framing)
- t-SNE and UMAP: what they do + the constelações warning
- t-SNE/UMAP as contrastive methods (optional depth, one slide)
- Instance segmentation definition ladder: semantic / instance / panoptic
- Embed-then-cluster as the workaround for the permutation-invariant label problem

**CRITICAL NOTE for block-writer**: The interaction-potential slide (pull/push in physics notation) is the course's peak moment — rehearse it. The EXACT same pull/push loss appears in NB2 (Act 1) and in the A&A halo paper. Do NOT name-drop the Weinberger paper or the halo application yet — Day 4's recognition moment depends on the audience seeing it as "familiar" when the paper's equations appear.

**Polls (show of hands):**
- Poll 1: "Qual par está mais perto em distância de pixels?" (pixel distance puzzle, before reveal)
- Poll 2: "O que acontece se não houver negativos (ou repulsão)?" (before the collapse explanation)

**Figures / Visuals:**
- Slide: 3-panel pixel puzzle (same digit, shifted digit, different digit) + distance values revealed
- Slide: encoder as cartographer (raw space → learned space, scatter comparison)
- Slide: label economy taxonomy (budget strategies tree)
- Slide: interaction potentials diagram (spring pull + charge push) — this is the signature visual
- Slide: pull/push loss in physics-style notation (L_pull, L_push, L_reg terms)
- Slide: collapse demo (what happens with only attraction)
- Slide: InfoNCE = Boltzmann weights over negatives (one slide)
- Slide: augmentations as invariance declarations
- Slide: t-SNE examples with different perplexities (constelações warning)
- Slide: semantic vs. instance vs. panoptic segmentation definition ladder

### Instructor Callout Content Spec
- **Prep notes**: Rehearse the interaction-potential slide (10'–18' segment) — this is the course's conceptual climax. The physics notation must appear consistent with NB2's cell 5 equations. Do NOT reveal the Weinberger name or the halo application.
- **Timing risks**: The taxonomy slide can balloon — one sentence per category. The t-SNE/UMAP section can expand — keep mechanics at picture level (hands-on shows the knobs).
- **Common pitfalls**: Do not name-drop Weinberger/halo application. Keep InfoNCE at one slide (D4). The taxonomy (supervised/semi/self-supervised) must not expand beyond one slide.

### Cross-References to Other Blocks
- Picks up from L02_B02's latent scatter: "o espaço latente era o palco; hoje ele é o protagonista."
- The pull/push loss in physics notation here is EXACTLY the loss in NB2 (cells 5–8) — students will have already seen it running on particles. This is a forward-reference that becomes a recognition moment in NB2.
- The augmentation-as-invariance framing is previewed here and realized in NB2's 🟣 mini-SimCLR cell.
- The embed-then-cluster paradigm introduced here is the exact method in L04_B01 (halo paper).
- The constelações warning here is demonstrated live in NB2 (cells 21–22).

---

## BLOCK BRIEF: L03_B02

**Block ID**: L03_B02
**pt-BR Title**: «Mão na massa II: esculpindo um espaço de embeddings»
**Type**: Hands-on (instructor executes NB2; students with laptops follow via badge)
**Date**: Thursday, July 23, 2026
**Anchor notebook**: `02_contrastive_embeddings.ipynb`

### Core Intuition (EN)
Watch contrastive learning happen: first as a pure particle system relaxing under pull/push potentials (no network!), then with a real encoder on MNIST, ending with clustering + t-SNE/UMAP inspection — the complete embed-then-cluster pipeline of instance segmentation, in miniature.

### One-Line Takeaway (pt-BR)
«Um bom espaço de embeddings transforma um problema sem função de perda num problema de clustering.»

### Full Chronograph (verbatim from Master Plan §4)

| Time | Segment | What & how |
|------|---------|------------|
| 0–3' | Setup + framing | Run-all; frame: "primeiro dinâmica molecular; depois deep learning; no fim vocês percebem que era a mesma coisa." |
| 3–12' | Act 1 — the particle sandbox (no network) | 2D points with instance labels; pull/push/reg potentials in JAX (~15 lines); relaxation via `jax.grad` + descent, **animated live**. Sliders/params for δ_pull, δ_push. 🟡 before running: "o que acontece com c_push = 0?" → run → total collapse (the trivial minimum, predicted by the audience). This cell is the course's signature visual. |
| 12–24' | Act 2 — a real encoder on public data | MNIST subset: encoder MLP → 2D embedding trained with discriminative (Weinberger-style) loss. Live evolution: scatter at epochs 0/early/late — classes visibly condensing (pretrained checkpoints; filmstrip fallback). 🟡 "que classes se misturam? faz sentido visual?" |
| 24–31' | Act 3 — harvest: cluster, then project | Run k-means on embedding → recover classes without labels → "isto É segmentação de instâncias." t-SNE and UMAP on 16-D embedding with 2–3 perplexity/n_neighbors values → picture changes → constelações warning made concrete. |
| 31–36' | Takeaway | «Um bom espaço de embeddings transforma um problema sem função de perda num problema de clustering.» Teaser: "amanhã: esta máquina exata prevendo onde nascem os halos de matéria escura." |
| 36–40' | Buffer | Questions. |

### Content Inventory

**For hands-on/demo blocks — notebook this wraps:**
This block wraps `02_contrastive_embeddings.ipynb`. Block-writer must read the FINAL executed notebook and reference its actual cells. Key cells: cell 4 (initial particle scatter), cell 7 (collapse demo), cell 8 (animated relaxation / filmstrip), cell 15 (MNIST evolution filmstrip), cell 16 (poll — which classes mix?), cell 19 (k-means on embedding), cells 21–22 (t-SNE / UMAP with parameter sweep).

**Concepts owned by this block:**
- Particle sandbox as "dynamics molecular physics"
- Pull/push/reg potentials in JAX (same as the physics notation from L03_B01)
- Trivial minimum (collapse) — confirmed live by the audience's prediction
- MLP encoder: training with discriminative loss on real data (MNIST)
- Embedding evolution as a filmstrip (chaos → clouds)
- "Degenerescências físicas" of handwriting (4/9, 3/8)
- K-means harvest = instance segmentation
- t-SNE and UMAP: parameter sweeps making the picture change
- Constelações warning made concrete and experiential

**Polls:**
- 🟡 "O que acontece com δ_push = 0?" (cell 6, before collapse demo)
- 🟡 "Que classes se misturam? Faz sentido?" (cell 16, after MNIST embedding)

### Instructor Callout Content Spec
- **Prep notes**: Act 1 (particle sandbox) must be pre-run to ensure animation or filmstrip works. MNIST .npz must be in assets/ and accessible without internet. Ensure UMAP is installed (`pip install umap-learn` in Setup cell).
- **Timing risks**: Act 1 is delightful and can easily exceed 12 min — timebox hard. t-SNE on > 5k points is slow — ensure subsampling is in the cell.
- **Common pitfalls**: MNIST download can be slow on venue wifi → ship cached .npz. The mini-SimCLR belongs in 🟣 only — augmentation-based CL is enrichment.

### Cross-References to Other Blocks
- Directly realizes L03_B01's interaction-potential narrative: "primeiro dinâmica molecular; depois deep learning; no fim vocês percebem que era a mesma coisa."
- The embed-then-cluster demo in Act 3 is the miniature version of L04_B01's halo paper.
- Teaser to L04_B01: "amanhã: esta máquina exata prevendo onde nascem os halos de matéria escura."
- The 🟣 mini-SimCLR previews the augmentation-based pair generation from L03_B01's theory.

---

## BLOCK BRIEF: L04_B01

**Block ID**: L04_B01
**pt-BR Title**: «Estudo de caso I: prevendo a formação de halos com segmentação de instâncias»
**Type**: Case study (slides built from paper figures)
**Date**: Friday, July 24, 2026
**Primary resource**: López-Cano et al., A&A 685, A37 (2024). Repo: `daniellopezcano/instance_halos`.

### Core Intuition (EN)
Yesterday's embed-then-cluster machine, deployed on the universe: predict which particles of the initial conditions end in which dark-matter halo. Students should feel recognition ("eu treinei essa perda ontem"), plus a deep physics coda on the irreducible limits of prediction.

### One-Line Takeaway (pt-BR)
«Segmentação de instâncias = esculpir um espaço onde objetos viram clusters — e o universo é segmentável.»

### Full Chronograph (verbatim from Master Plan §5)

| Time | Segment | What & how |
|------|---------|------------|
| 0–4' | Mapa + reframe | "Três dias de ingredientes; hoje, a cozinha de verdade. Regra do dia: vocês vão RECONHECER, não aprender do zero." |
| 4–11' | The physics problem | Structure formation in 3 slides: initial near-homogeneous field → gravity amplifies → haloes as the scaffolding of galaxies. The Lagrangian question: _which initial particles end up in which halo?_ (proto-halo regions; why it matters: mass, spin, formation history). Analogy: **meteorologia do universo** — given today's map, predict where the storms form. |
| 11–17' | Why it's hard as ML — and the recognition moment ⭐ | Variable number of haloes + permutation-invariant labels = yesterday's "no direct loss" problem. Reveal the paper's solution: map every particle into a **pseudo-space** where haloes are clusters — show the Weinberger loss equations NEXT TO a screenshot of yesterday's sandbox: pull, push, reg, term by term identical. «Vocês treinaram esta perda ontem, com bolinhas coloridas; aqui, as bolinhas são partículas do universo.» |
| 17–26' | What the model achieves | Results tour via paper figures: semantic mask (which particles collapse at all); predicted vs. true Lagrangian halo shapes (visually stunning; disconnected regions that watershed cannot represent); halo mass function reproduced; ~7 minutes per simulation on GPU inference. |
| 26–33' | The physics coda: limits of prediction | Twin-simulation experiment: perturb only unresolved small scales in the ICs → the two universes disagree on some membership assignments → part of the task is **aleatorically undetermined** → the model performs close to that ceiling. Lesson slide: «antes de julgar um modelo, pergunte qual é a nota máxima possível.» (Butterfly-effect resonance.) |
| 33–36' | Takeaway | «Segmentação de instâncias = esculpir um espaço onde objetos viram clusters — e o universo é segmentável.» Repo pointer for the curious. |
| 36–40' | Buffer | Questions. |

### Content Inventory

**Paper: A&A 685, A37 (2024) — key figures and numbers to feature:**
- Fig. 1: slice of initial density field (top) vs. halo membership labels (bottom) — the prediction problem visualized. This is the opening figure of the paper and perfectly illustrates the problem to students.
- Fig. 2: schematic of Weinberger loss in 2D pseudo-space with cluster centers (crosses) and pull/push arrows — this is the RECOGNITION MOMENT slide.
- Halo mass function: model reproduces the HMF (paper Fig. 3 or equivalent).
- Lagrangian halo shapes: predicted vs. true side by side — visually compelling, including disconnected regions.
- Twin-simulation experiment: indetermination analysis (Section 2.5 of the paper) — the optimality ceiling.
- Inference speed: ~7 minutes per GPU for a 256³ simulation.
- The BaCE loss for semantic segmentation: β = 0.58152, semantic threshold = 0.589 (mention briefly as the "first network's job").

**The Recognition Moment (must be staged explicitly):**
Show the Weinberger loss pull/push equations from the paper (Section 2.3) SIDE BY SIDE with a screenshot of NB2's Act 1 cell 5 (the sandbox potentials). Point term by term: L_pull, L_push, L_reg. Say: «Vocês treinaram esta perda ontem, com bolinhas coloridas; aqui, as bolinhas são partículas do universo.»

**Concepts owned by this block:**
- Structure formation: near-homogeneous initial field → gravitational collapse → haloes
- Lagrangian vs. Eulerian perspective (brief vocabulary)
- The Lagrangian question: which initial particles → which halo?
- Why mass, spin, formation history all depend on answering this question
- Meteorologia do universo analogy
- Variable-number + permutation-invariant instance segmentation problem
- The Weinberger (embed-then-cluster) solution: pseudo-space + pull/push loss
- Semantic segmentation network (BaCE loss) as the first stage
- Instance segmentation network (Weinberger loss) as the second stage
- Results: Lagrangian halo shapes, mass function, disconnected regions
- Why watershed fails for disconnected regions (contrast with the paper's approach)
- Inference speed: ~7 min / GPU / simulation (vs. N-body: hours/days)
- Twin-simulation experiment as aleatoric undetermination
- Optimality ceiling: "antes de julgar um modelo, pergunte qual é a nota máxima possível"

**Figures/slides to prepare:**
- 3 slides on structure formation (initial field → amplification → halo scaffolding)
- 1 slide: the Lagrangian question + meteorologia do universo analogy
- 1 slide: why instance segmentation is hard (variable number + permutation invariance) — this slide reuses the concept from L03_B01
- **THE RECOGNITION SLIDE**: Weinberger loss equations NEXT TO NB2 cell 5 screenshot (L_pull, L_push, L_reg, term by term)
- 2–3 slides: results (semantic mask, Lagrangian shapes including disconnected, HMF)
- 1 slide: why watershed fails (disconnected regions) → why embed-then-cluster wins
- 1 slide: inference speed comparison (ML vs. N-body)
- 2 slides: twin-simulation / indetermination / optimality ceiling
- 1 slide: repo pointer + IC entry points

### Instructor Callout Content Spec
- **Prep notes**: Prepare the recognition slide before anything else — it is the emotional climax of the course. Take the pull/push equations directly from paper Section 2.3 and the NB2 cell 5 code. The two should be visually adjacent on one slide. Verify paper figures are available (from the A&A published version or arXiv preprint). Check the `instance_halos` repo for figure files.
- **Timing risks**: The physics problem setup (structure formation) can expand — 3 slides, 7 min max. The results tour is rich — select only the most visually compelling figures (shapes + HMF + disconnected regions). Do not show architecture details or hyperparameter tables.
- **Common pitfalls**: Every equation shown must have appeared in toy form in Days 1–3. No new NN mechanics. Do NOT go into CNN architecture details. The twin-simulation coda is important — do not skip it (it teaches honest ML evaluation culture).

### Cross-References to Other Blocks
- Directly picks up from L03_B02 (NB2 particle sandbox) — the recognition moment connects NB2 Act 1 to the paper's Section 2.3.
- The embed-then-cluster pipeline introduced in L03_B01 and demonstrated in L03_B02 is the same pipeline deployed here.
- The "optimality ceiling" lesson echoes the honest-ML-evaluation theme from L02_B01 ("confiança não é competência").
- Bridge to L04_B02: "agora vemos o mesmo princípio (encoder+head, três regimes) aplicado a quasares reais."

---

## BLOCK BRIEF: L04_B02

**Block ID**: L04_B02
**pt-BR Title**: «Estudo de caso II: do mock ao céu — adaptação de domínio no J-PAS» + encerramento
**Type**: Case study + course closing
**Date**: Friday, July 24, 2026
**Primary resource**: López-Cano et al., arXiv:2602.13902. Repo: `daniellopezcano/JPAS_Domain_Adaptation`.

### Core Intuition (EN)
Wednesday's three-regime toy experiment, at survey scale: a mocks-trained star/galaxy/quasar classifier degrades on the real sky; freezing the head and adapting the encoder with a small labeled set repairs it; calibrated probabilities become telescope economics. Close the course by lighting up the whole map and pointing at the frontier.

### One-Line Takeaway (pt-BR)
«Representações são coordenadas; coordenadas se esculpem (contrastiva); esculturas quebram quando o mundo muda (shift); e adaptá-las é barato — se você souber o que congelar.»

### Full Chronograph (verbatim from Master Plan §5)

| Time | Segment | What & how |
|------|---------|------------|
| 0–7' | The data & the stakes | J-PAS in 3 slides: 54 narrow-band filters → «fotometria fingindo ser espectroscopia» (show real J-spectra of the 4 classes: star, galaxy, QSO low-z, QSO high-z). Why classify: candidate lists for WEAVE spectroscopic follow-up — false positives = wasted fibers = money and photons. **Calibrated probabilities**, not just labels. One slide on the z≈2.1 class boundary: a continuous variable split by a threshold → unavoidable mixing (an "a física é difícil" error, to contrast with shift errors below). |
| 7–13' | The setup — recognized | Mocks: DESI spectra projected through J-PAS filters = free labels; a small cross-matched real-label set; the _same encoder+head architecture and the same three regimes as NB1_, shown as a literal side-by-side with Wednesday's notebook figures. «Vocês rodaram este experimento na quarta-feira, em 2D.» |
| 13–21' | The payoff | Confusion-matrix reveal sequence: in-domain mocks (tight) → zero-shot on real sky (STAR→QSO_high confusion ~1.2% → ~9.2%) → after SSDA (~0.9%). Headline metrics: macro-F1 0.73 (mocks-only/zero-shot) → 0.79 (target-only) → 0.82 (SSDA); rare high-z quasars: F1 0.37 → 0.55 → 0.66. Error decomposition: «a física é difícil» (z≈2.1 boundary, host-galaxy dilution, line aliases) vs. «meu treino estava enviesado» (domain shift). |
| 21–26' | From probabilities to operations | Calibration results (reliability diagram, ECE≈0.05) and how calibrated per-class probabilities → purity/completeness trade-offs → fiber-budget forecasts for WEAVE-QSO. "É aqui que uma curva ROC vira decisão de telescópio." |
| 26–33' | Panorama & how to get in | Frontier in 3 slides: contrastive/self-supervised pretraining for cross-survey embeddings; foundation models in astronomy; photo-z and simulation-based inference. Concrete entry points: both public repos as thesis/IC starting material; FAPESP IC/master opportunities; department groups. Relight the full course map, all 8 blocks. |
| 33–36' | Final takeaway | «Representações são coordenadas; coordenadas se esculpem (contrastiva); esculturas quebram quando o mundo muda (shift); e adaptá-las é barato — se você souber o que congelar.» Feedback QR. |
| 36–40' | Buffer | Final questions. |

### Content Inventory

**Paper: arXiv:2602.13902 — key figures and numbers to feature:**
- Figure 1 from paper: representative J-spectra (real J-PAS solid, DESI→J-PAS mock dashed) for all 4 classes — shows the shift visually.
- Data scale: ~1.5 × 10⁶ mock sources (DESI→J-PAS) vs. 52,020 real J-PAS observations.
- The four classes: star, galaxy, QSO low-z, QSO high-z (match NB1's 4-class structure).
- Class imbalance: QSO high ≲ 2% in real J-PAS data (matches the NB1 rare-class lesson).
- Three regimes = three baselines: mocks-only (zero-shot), J-PAS-only (target-only, same label budget), SSDA.
- Headline metrics: macro-F1 0.73 → 0.79 → 0.82; overall TPR 0.87 → 0.85 → 0.89.
- QSO high F1: 0.37 → 0.55 → 0.66 (the biggest gain; most consequential for WEAVE targeting).
- Confusion jump: STAR→QSO_high ~1.2% (mocks) → ~9.2% (zero-shot) → ~0.9% (SSDA).
- Calibration: ECE≈0.05 after SSDA (reliability diagram figure from paper).
- Pipeline: pretrain on mocks (abundant labels) → adapt on J-PAS×DESI cross-match (scarce labels); same encoder+head decomposition as NB1.
- WEAVE-QSO fiber targeting: calibrated probabilities → purity/completeness trade-offs → fiber budget.

**The Two Recognition Moments (must be staged explicitly):**
1. Recognition moment 1 (0–7'): The 4 J-PAS classes (star, galaxy, QSO low, QSO high) match NB1's 4 classes. Point this out.
2. Recognition moment 2 (7–13'): Show the three-regime experimental design side by side with NB1's notebook figure (decision maps or K-sweep). «Vocês rodaram este experimento na quarta-feira, em 2D.»

**Concepts owned by this block:**
- J-PAS: 54 narrow-band filters, "fotometria fingindo ser espectroscopia"
- J-spectra: the data vector per source (55 bands after masking)
- Why accurate classification matters: WEAVE fiber budget, purity vs. completeness
- Calibrated probabilities vs. hard labels (bring home the termômetro lesson from L02_B01)
- The z≈2.1 class boundary: "a física é difícil" vs. "meu treino estava enviesado"
- Mocks from DESI→J-PAS projection (~1.5M sources, free labels)
- Three regimes: mocks-only (zero-shot), target-only, SSDA
- Headline metrics: macro-F1 and per-class F1 (QSO high as the key beneficiary)
- Calibration: reliability diagram, ECE≈0.05
- Fiber-budget decision: ROC curve → telescope operations
- Frontier: cross-survey embeddings, foundation models, photo-z, SBI
- Entry points: repos, FAPESP, department groups
- Final synthesis: the full 4-day narrative compressed to one sentence

**Error decomposition lesson (critical for scientific maturity):**
- "a física é difícil" errors: z≈2.1 QSO boundary (continuous-to-discrete split), host-galaxy dilution, line aliases → irreducible, not a model failure
- "meu treino estava enviesado" errors: domain shift from mocks to real sky → addressable with SSDA
- Teaching students to distinguish these two types is "a habilidade mais transferível deste curso."

**Figures/slides to prepare:**
- 3 slides: J-PAS survey + 54-filter system + representative J-spectra (Fig. 1 from paper)
- 1 slide: z≈2.1 boundary (QSO classification boundary in redshift space)
- 1 slide: why calibrated probabilities matter (WEAVE fiber budget framing)
- **THE RECOGNITION SLIDE**: three-regime experimental design from J-PAS paper NEXT TO NB1 figures
- Confusion matrix sequence: mocks tight → zero-shot degraded → SSDA recovered
- Headline metric table: macro-F1 and QSO high F1 for all three regimes
- Error decomposition slide: two buckets ("física difícil" vs. "shift")
- Reliability diagram + ECE≈0.05
- ROC → telescope decisions
- 3 frontier slides
- Relit course map (all 8 blocks visible, narrative arc complete)
- Feedback QR code

### Instructor Callout Content Spec
- **Prep notes**: The temptation to go technical is maximal here — every equation shown must have appeared in toy form. Rehearse the recognition slide (three-regime side-by-side with NB1). The course-map relight must be deliberate and slow (30 seconds per day).
- **Timing risks**: The data & stakes segment (0–7') can expand if the J-spectra slide is too rich — keep it to 3 slides. The frontier segment (26–33') must not become a paper-list dump — 3 slides maximum, always grounded in student agency.
- **Common pitfalls**: Do not show paper-internal details (hyperparameter sweeps, architecture tables, appendices). Do not skip the error decomposition — it is the most scientifically mature lesson. The feedback QR must be working before class.

### Cross-References to Other Blocks
- Directly echoes L02_B01 (domain shift theory) and L02_B02 (NB1 three-regime experiment) — the recognition slides connect them.
- The "termômetro descalibrado" analogy from L02_B01 comes home in the calibration results.
- The encoder+head decomposition (sotaques) from L02_B01 is the same architecture in the paper.
- The z≈2.1 "física é difícil" error complements the twin-simulation aleatoric undetermination from L04_B01 — both teach optimality ceilings.
- The final takeaway sentence is the course's closing synthesis of all 4 days.

---

## INDEX BRIEF: 00_INDEX.md

### Audience
Students of the I Escola de Inverno do IFUSP (final-year physics undergraduates, ~130 people). This is the entry-point file for the Obsidian vault. All content in pt-BR.

### Purpose
The index is the student's compass for the entire vault. It must answer: "What is this course?", "Where do I find the material for each day?", "How do I use these files?", and "What resources can I study on my own?"

### Structure Specification

The file must contain the following sections (in pt-BR):

1. **Header with course identity**: Title, dates (July 21–24, 2026), venue (IFUSP), instructor name, and a one-line course tagline.

2. **Mapa do curso** (course map table): A table with all 8 blocks, each linking to its .md file. Columns: Dia | Bloco | Título | Tipo | Notebook.

3. **Guia por dia** (per-day paragraph guide): One paragraph per lecture day (L1–L4) explaining in accessible pt-BR what the day covers, what students will do, and what they will take away. These paragraphs should be written for a student who missed the lecture and is using the vault to catch up.
   - L1: panorama + ferramentas
   - L2: domain shift + hands-on NB1
   - L3: aprendizagem contrastiva + hands-on NB2
   - L4: dois estudos de caso reais

4. **Links for the 8 blocks**: Obsidian-style `[[block_file_name]]` links for each of the 8 blocks.

5. **Links for the 3 notebooks**: Links to the jax-examples/ notebooks with one-line descriptions.

6. **Recursos para estudo autônomo**: Brief intro paragraph followed by the curated resource list (from Master Plan §7, in pt-BR with one-line blurbs). This list lives primarily in L01_B01.md — the index either embeds it or links to it.

7. **Como usar este vault**: A short guide (3–5 bullet points) explaining how the Obsidian vault is organized: dual-purpose files (student sections + instructor callouts), how to navigate between blocks, how to open notebooks in Colab, where to find the resource list.

### Writing Conventions
- pt-BR throughout.
- Obsidian-style wikilinks for cross-references to block files.
- No instructor callouts in this file (it is student-only).
- Friendly, accessible tone — as if written by the instructor to the students after Day 1.
- Include the repo URL (`daniellopezcano/I-Escola-de-Inverno-do-IFUSP`) and Colab badge template.

---

## Manifest Philosophy

Each brief above is SELF-CONTAINED: a writer reading only its brief plus the listed inputs can produce the asset without consulting the Master Plan.

Master Plan tables are copied verbatim (chronographs) rather than summarized.

Where the Master Plan gives a spec in Portuguese (notebook cell outlines), the original Portuguese is preserved.

The DECISIONS section at the top of this manifest records every choice made by the course-architect that was not explicitly resolved in the Master Plan. Every downstream agent must treat these decisions as authoritative unless overridden by the human user.

---

*Manifest written: 3 notebook briefs, 8 block briefs, 8 decisions flagged. Path: .dev/agents/work/course_manifest.md*
