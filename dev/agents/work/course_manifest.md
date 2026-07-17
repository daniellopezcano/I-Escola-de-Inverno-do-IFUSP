# Course Manifest — I Escola de Inverno do IFUSP
# «Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos»
# Refreshed: 2026-07-17 | Authoritative for all downstream agents

---

## GLOBAL CONVENTIONS

### Authoritative brief
`dev/agents/work/my_feedback_v2.md` overrides the Master Plan and all earlier documents.
Rule #1: **L1B1 is the gold standard. Make everything else match its structure and philosophy.**

### Language
- Student-facing content (block markdowns, notebook cells, README sections): **pt-BR**.
- Instructor/meta/pipeline content (this manifest, build logs, coherence report): English.

### Block naming
`LxBy` — e.g., L1B1, L2B1, L4B2. NOT `L01_B01`, `L01B01`, or `LXX_BYY`.  
Block files live at `course-materials/LxBy.md`.

### Block types
| Type | Blocks | Role |
|------|--------|------|
| THEORY | L1B1, L2B1, L3B1, L4B1, L4B2 | **Narrative markdown** — the primary artifact from which the instructor builds Google Slides (direction: markdown → slides, not slides → markdown). Compact, pt-BR, with inline original-paper references. Structure so slides can be lifted directly from the markdown's sections. |
| NOTEBOOK | L1B2, L2B2, L3B2 | **Minimal markdown** — Colab link + one line of framing only. Paired with a JAX notebook in `jax-examples/notebooks/`. |

### Gold-standard exemplars (read before writing any block; do NOT modify them)
- Theory: `course-materials/L1B1.md` — arc: apresentação → contexto/motivação → usos e limites honestos → estrutura/materiais; compact, narrative, original-paper refs inline.
- Notebook: `course-materials/L1B2.md` — just the Colab badge + one line.

### Paths
| Asset | Path |
|-------|------|
| Block markdowns | `course-materials/LxBy.md` |
| Notebooks | `jax-examples/notebooks/NN_slug.ipynb` |
| Notebook py-percent sources | `jax-examples/src_NN_slug.py` |
| Asset generators | `jax-examples/utils/make_assets_NN_slug.py` |
| Input datasets (gitignored) | `jax-examples/assets/` |
| References PDFs | `references/` |
| Build logs | `dev/agents/work/build_logs/<name>.log` |
| Coherence report | `dev/agents/work/coherence_report.md` |
| README hub | `README.md` (single navigational hub; no standalone index file) |
| Python env | `/home/dlopez/miniconda3/envs/WinterSchool/bin/` (absolute paths; never `conda activate`) |

### §5 Artifact policy (from my_feedback_v2.md §5 — authoritative)
- Do **not** commit notebook-generated figures (`*.png`), model checkpoints (`*.pkl`), or produced data (`*.npz` outputs).
- `jax-examples/assets/` is **gitignored** (only `.gitkeep` committed). It holds only genuine input datasets needed at runtime; if none are needed, it can be empty.
- Notebooks must **self-generate or download at runtime** everything they load. No loading from committed artifacts.
- `.gitignore` must exclude `*.png`, `*.pkl`, and generated `*.npz` files. (Currently enforced: `jax-examples/assets/` is gitignored.)
- Target state: every notebook runs top-to-bottom in clean Colab with no committed artifacts required.

### Navigation hub
`README.md` is the single navigational hub. The old standalone `00_INDEX.md` was removed.

### Lecture map
| Block | Day | Title (pt-BR) |
|-------|-----|---------------|
| L1B1 | 1 | Aprendizado de máquina e Física: o mapa do território |
| L1B2 | 1 | A caixa de ferramentas: Python, Colab e JAX na prática → `00_caixa_de_ferramentas.ipynb` |
| L2B1 | 2 | Mudança de domínio: quando o treino não é a prova |
| L2B2 | 2 | Mão na massa I: quebrar e consertar um classificador → `01_domain_shift_toy.ipynb` |
| L3B1 | 3 | Aprendizagem contrastiva: geometria da similaridade |
| L3B2 | 3 | Mão na massa II: esculpindo um espaço de embeddings → `02_contrastive_embeddings.ipynb` |
| L4B1 | 4 | Estudo de caso I: prevendo a formação de halos com segmentação de instâncias |
| L4B2 | 4 | Estudo de caso II: do mock ao céu — adaptação de domínio no J-PAS |

### References
| File | Paper | Anchor block |
|------|-------|--------------|
| `references/2602.13902v1.pdf` | López-Cano et al. arXiv:2602.13902 (J-PAS SSDA) | L4B2 |
| `references/2311.12110v3.pdf` | López-Cano et al. A&A 685 A37 (halo instance segmentation) | L4B1 |

### Definition of done
- **Theory block:** reachable from README; clear, compact, self-sufficient narrative in pt-BR; structured so slides can be built directly from it; references are original papers; no orphan/broken links.
- **Notebook block:** reachable from README; markdown is just Colab link + framing; notebook runs clean in Colab and commits no artifacts.

---

## DECISIONS (flagged for human review; downstream agents treat as authoritative)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | NB2 primary dataset | MNIST subset (~4k samples, generated at runtime) | Master Plan §6.3 names MNIST as primary; the particle sandbox provides physics flavor without data risk. |
| D2 | L4 block order | L4B1 = halos (A&A 685 A37), L4B2 = J-PAS SSDA (arXiv:2602.13902) + closing | L3→L4B1 (instance segmentation recognition) and L2→L4B2 (DA recognition); J-PAS outlook as natural course finale. |
| D3 | NB1 class count | 4 classes | Reduces visual clutter while preserving the imbalance lesson (one rare class). |
| D4 | InfoNCE depth in L3B1 | One slide + one optional section in NB2 | Discriminative pull/push loss is the load-bearing concept; InfoNCE is enrichment. |
| D5 | Poll mechanism | Show of hands | No tech dependency; works in any venue; instructor reads the room instantly. |
| D6 | NB1 shift magnitude | Cluster centers displaced ~2–3 std of source clusters | Zero-shot failure visually obvious but not total. |
| D7 | NB0 1D function | Damped sinusoid: y = A·sin(2πx/λ)·exp(-x/τ) + ε | Physics-flavored; explicitly specified in Master Plan §6.1. |
| D8 | Notebook runtime budget | < 3 min wall-clock end-to-end on free Colab CPU | Shared convention from Master Plan §6. |

---

## BLOCK BRIEFS

---

### L1B1 — teoria (GOLD STANDARD)
**Title:** «Aprendizado de máquina e Física: o mapa do território»  
**Type:** THEORY markdown  
**Slides:** https://docs.google.com/presentation/d/1urJoVZ1Oeko21DEa6jq737MJcpetG1whUMFMDD05oq0/edit?usp=drive_link  
**Status:** COMPLETE. Gold standard — do NOT modify. All other theory blocks must match its structure and spirit.

Narrative arc (all other theory blocks emulate this):
1. Apresentação e introdução
2. Contexto e motivação — data deluge in cosmology (2dFGRS → Rubin) + parallel ML rise (Deep Blue 1997 → ChatGPT 2022)
3. Usos e aplicações — what ML does (emulation, patterns, SBI, anomaly detection, real-time control) + 5 honest limits (no reliable extrapolation; no self-certified error; inherits bias; no causal modeling; hard to interpret); course topics (latent space, domain adaptation, contrastive learning)
4. Estrutura do curso e materiais — pointer to README/GitHub

---

### L1B2 — notebook (GOLD STANDARD)
**Title:** «A caixa de ferramentas: Python, Colab e JAX na prática»  
**Type:** NOTEBOOK markdown  
**Colab:** https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/00_caixa_de_ferramentas.ipynb  
**Notebook:** `jax-examples/notebooks/00_caixa_de_ferramentas.ipynb`  
**Status:** COMPLETE. Gold standard notebook markdown — do NOT modify. All other notebook markdowns must match its minimal format.

---

### L2B1 — teoria
**Title:** «Mudança de domínio: quando o treino não é a prova»  
**Type:** THEORY markdown  
**Slides:** https://docs.google.com/presentation/d/1pIMOeHfmTVYm2h_TUT8vcqtHDXz3jW1oxVN8rdWgm9s/edit?usp=drive_link  
**Status:** NEEDS REWRITE to L1B1 standard. Current file is a slide-schematic draft; block-writer must replace with compact narrative markdown.

**Narrative arc** (follow L1B1 spirit — motivation → concepts → honest limits → materials pointer):
1. Apresentação — brief; contextualizes Day 2 in the course (echo of NB0's overfitting seed: "generalizar dentro da mesma distribuição já era difícil; e quando a distribuição muda?").
2. Contexto/motivação — the i.i.d. assumption and when it breaks: 3 universal examples (speech accents, hospital scanners, self-driving fog) + 1 scientific (sim-trained model meets real instrument); analogia «estudar pelo simulado, fazer a prova de verdade».
3. Conceitos — taxonomy: covariate shift / prior shift / concept shift (one 2D scatter description per type); silent failure and calibration («errado e confiante é o modo de falha mais perigoso da ciência com ML»; reliability diagram; termômetro descalibrado analogy).
4. Usos e limites honestos — mitigation map by available supervision: source-only (augmentation, domain randomization) / unlabeled target (distribution alignment, importance reweighting) / few target labels (transfer learning, SSDA: freeze head, adapt encoder); encoder+head decomposition introduced here for the first time; sotaques analogy; practical detection checklist (domain classifier, feature histograms, embedding drift).
5. Materiais — pointer to README and to NB1.

**Key concepts:** i.i.d. assumption, 3 shift types, silent confident failure, calibration ≠ accuracy, reliability diagram, encoder+head decomposition, sotaques analogy, domain classifier as detection tool.  
**Equations:** p_S(x,y) ≠ p_T(x,y) — no dense derivations.  
**Inline references:** Ganin et al. 2016 JMLR (arXiv:1505.07818); Tzeng et al. 2017 CVPR (arXiv:1702.05464).  
**Do NOT reveal J-PAS results** — that is Day 4's payoff.

---

### L2B2 — notebook
**Title:** «Mão na massa I: quebrar e consertar um classificador»  
**Type:** NOTEBOOK markdown  
**Colab:** https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/01_domain_shift_toy.ipynb  
**Notebook:** `jax-examples/notebooks/01_domain_shift_toy.ipynb`  
**Status:** BUILD GREEN. Markdown needs refresh to L1B2 minimal standard (just Colab link + one framing line).

**Framing line:** Quatro atos num universo gaussiano 2D: causa covariate shift visual (4 classes, desbalanceadas), treina na fonte e observa falha catastrófica no alvo, diagnostica o shift sem rótulos do alvo (AUC ≈ 0,75 (teste)), e compara zero-shot / target-only / SSDA numa varredura de K — o mesmo experimento que o J-PAS replica com quasares no Dia 4.

---

### L3B1 — teoria
**Title:** «Aprendizagem contrastiva: geometria da similaridade»  
**Type:** THEORY markdown  
**Slides:** https://docs.google.com/presentation/d/17ssxMhezRtTREFM1FZc32VMsYP1cQ5eFazUUM1QdQQs/edit?usp=drive_link  
**Status:** NEEDS REWRITE to L1B1 standard.

**Narrative arc:**
1. Apresentação — contextualizes Day 3; the latent space moves from stage to protagonist.
2. Contexto/motivação — the raw-distances puzzle (same digit shifted 3px is FARTHER in pixel space than a different digit; poll: «qual par está mais perto?»); encoder as cartographer; label economy: supervised / semi-supervised / self-supervised as budget strategies.
3. Conceitos — the contrastive principle: «escolha quais pares devem ficar perto (positivos) e quais devem ficar longe (negativos); a rede geometriza a sua escolha»; losses as interaction potentials: L_pull (mola) pulls points toward cluster center, L_push (carga) repels centers from each other, L_reg anchors; degenerate trivial minimum (all collapse when L_push = 0); InfoNCE = Boltzmann weights over negatives (one paragraph, not a full section — per D4); where positive pairs come from: labels vs. augmentations; augmentations as invariance declarations («cada augmentação é uma declaração de invariância», Noether-flavored); t-SNE and UMAP as inspection tools with the constelações warning (distances between clusters ≈ meaningless; picture changes with perplexity).
4. Usos e limites — embed-then-cluster workaround for instance segmentation (variable number + permutation-invariant labels → no direct loss → solution); limitation: cluster quality = representation quality; t-SNE/UMAP do not preserve inter-cluster distances.
5. Materiais — pointer to NB2 and README.

**Key concepts:** raw distances lie / learned distances reveal, L_pull / L_push / L_reg (physics notation), trivial minimum, InfoNCE, augmentations as invariance, t-SNE/UMAP warning, embed-then-cluster.  
**Equations:** L_pull = mean(max(‖μ_c − x_i‖ − δ_pull, 0)²), L_push analogous; each symbol defined on first use.  
**Inline references:** Chen et al. 2020 SimCLR (arXiv:2002.05709); Huertas-Company & Lanusse 2023 PASA (doi:10.1017/pasa.2022.55).  
**CRITICAL:** Do NOT name Weinberger or the halo paper — the recognition moment belongs to L4B1.

---

### L3B2 — notebook
**Title:** «Mão na massa II: esculpindo um espaço de embeddings»  
**Type:** NOTEBOOK markdown  
**Colab:** https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/02_contrastive_embeddings.ipynb  
**Notebook:** `jax-examples/notebooks/02_contrastive_embeddings.ipynb`  
**Status:** BUILD GREEN. Markdown needs refresh to L1B2 minimal standard.

**Framing line:** Três atos — sandbox de partículas 2D relaxando sob potencial pull/push (com demonstração do colapso trivial); encoder MLP em MNIST com a mesma perda; colheita do espaço latente com k-means (ARI = 0,743) e t-SNE em três perplexidades.

---

### L4B1 — teoria
**Title:** «Estudo de caso I: prevendo a formação de halos com segmentação de instâncias»  
**Type:** THEORY markdown  
**Slides:** https://docs.google.com/presentation/d/1ZVmImbVYYQAWHdR6NNlSLlCw8jtiLWMYDwlg4315dhk/edit?usp=drive_link  
**Status:** LOW PRIORITY rewrite (per my_feedback_v2.md §8). Anchor: `references/2311.12110v3.pdf`.

**Narrative arc:**
1. Apresentação — «três dias de ingredientes; hoje, a cozinha de verdade; vocês vão RECONHECER, não aprender do zero».
2. Contexto/motivação — structure formation: near-homogeneous initial field → gravitational collapse → halos as galaxy scaffolding; the Lagrangian question (which initial particles end in which halo?); meteorologia do universo analogy.
3. Conceitos + RECOGNITION MOMENT — instance segmentation problem: variable count + permutation-invariant labels → no direct loss → embed-then-cluster workaround; show Weinberger loss equations from paper Section 2.3 side-by-side with NB2 Act 1 cell: L_pull, L_push, L_reg, term-by-term identical; «Vocês treinaram esta perda ontem, com bolinhas coloridas; aqui, as bolinhas são partículas do universo»; results (Lagrangian halo shapes including disconnected regions, HMF, ~7 min/GPU inference).
4. Limites honestos — twin-simulation experiment: perturb only unresolved scales → two universes disagree on some assignments → part of the task is aleatorically undetermined; the model performs near that ceiling; «antes de julgar um modelo, pergunte qual é a nota máxima possível».
5. Materiais — pointer to `instance_halos` repo and README.

**Key concepts:** structure formation, Lagrangian question, embed-then-cluster, the recognition moment (same loss as NB2), aleatoric undetermination, optimality ceiling.  
**Inline references:** `references/2311.12110v3.pdf` (López-Cano et al. A&A 685 A37); cite Section 2.3 for the loss equations.

---

### L4B2 — teoria
**Title:** «Estudo de caso II: do mock ao céu — adaptação de domínio no J-PAS» + encerramento  
**Type:** THEORY markdown  
**Slides:** https://docs.google.com/presentation/d/1E4n9hgIszUmmZiGFGFF2BJMCBhqDiU1iSYfgl3rX6HE/edit?usp=drive_link  
**Status:** LOW PRIORITY rewrite (per my_feedback_v2.md §8). Anchor: `references/2602.13902v1.pdf`.

**Narrative arc:**
1. Apresentação — J-PAS in brief: 54 narrow-band filters, «fotometria fingindo ser espectroscopia»; 4 classes: star, galaxy, QSO low-z, QSO high-z; stakes: WEAVE fiber budget, calibrated probabilities.
2. Contexto/motivação — the z≈2.1 class boundary: «a física é difícil» (irreducible error from continuous-to-discrete split); mock data from DESI→J-PAS projection (~1.5M sources, free labels); the domain gap: mock J-spectra vs. real sky.
3. Conceitos + RECOGNITION MOMENT — same encoder+head architecture and three regimes as NB1, shown side-by-side with NB1 figures; «Vocês rodaram este experimento na quarta-feira, em 2D»; headline metrics (macro-F1: 0.73 → 0.79 → 0.82; QSO high F1: 0.37 → 0.55 → 0.66; confusion jump STAR→QSO_high: 1.2% → 9.2% → 0.9%); error decomposition: «a física é difícil» vs. «meu treino estava enviesado»; calibration ECE ≈ 0.05; ROC → telescope operations.
4. Limites honestos e fronteira — frontier: cross-survey embeddings, foundation models in astronomy, photo-z, SBI; entry points: both public repos as IC/master starting material; department groups, FAPESP.
5. Encerramento — relight full course map (all 8 blocks); synthesis sentence: «Representações são coordenadas; coordenadas se esculpem; esculturas quebram quando o mundo muda; e adaptá-las é barato — se você souber o que congelar.»

**Key concepts:** J-PAS filter system, calibrated probabilities, z≈2.1 boundary, three-regime experiment (recognition), error decomposition (física difícil vs. shift), calibration, ROC → fiber budget, frontier.  
**Inline references:** `references/2602.13902v1.pdf` (López-Cano et al. arXiv:2602.13902); cite figures 1 (J-spectra), confusion matrices, calibration diagram.

---

## NOTEBOOK STATUS AND §5 CLEANUP TASKS

All three notebooks are **BUILD GREEN** (verified). The `.gitignore` already excludes `jax-examples/assets/`, `*.png`, `*.pkl`. No committed artifacts tracked in git.

### Current status
| Notebook | BUILD GREEN | §5 compliant |
|----------|-------------|--------------|
| 00_caixa_de_ferramentas.ipynb | yes (13s) | verify clean Colab run |
| 01_domain_shift_toy.ipynb | yes (63s clean/17s cached) | §5 DONE: data inline, checkpoints generate-if-absent, AUC bug fixed (0.785→0.749 test-set) |
| 02_contrastive_embeddings.ipynb | yes (25s) | verify clean Colab run |

### §5 cleanup checklist (notebook-builder task)
For each notebook, verify:
1. All figure saves (`plt.savefig`) write to gitignored path (e.g. `jax-examples/assets/`) or `/tmp` — NOT to any committed path.
2. All checkpoint saves write to gitignored path — NOT committed.
3. For NB2 MNIST: data is downloaded at runtime (via `torchvision`, `tensorflow_datasets`, or URL) — no committed `.npz`.
4. For NB1 2D toy data: generated at runtime with fixed seed=42 — no committed `.npz`.
5. For NB0: all assets generated at runtime.
6. Notebook runs top-to-bottom in a clean Colab environment where `jax-examples/assets/` does NOT pre-exist.
7. Re-verify BUILD GREEN after cleanup.

NB1 AUC bug RESOLVED (2026-07-17): domain classifier now evaluates on 20% held-out test split; AUC = 0.749.

---

*Manifest refreshed: 8 block briefs, §5 policy updated, LxBy naming throughout, 00_INDEX.md removed, README as single hub.*
