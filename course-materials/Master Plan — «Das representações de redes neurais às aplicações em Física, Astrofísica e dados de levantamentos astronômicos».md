> **Planning document (v3, consolidated).** Internal working language: English. All student-facing material derived from this plan MUST be produced in Portuguese (pt-BR). **Event**: I Escola de Inverno do IFUSP · July 21–24, 2026 · 4 lectures × 90 min (2 × 40-min blocks + 10-min break) · ~130 final-year physics undergraduates. **Repo**: `daniellopezcano/I-Escola-de-Inverno-do-IFUSP` (`course-materials/` = dual-purpose Obsidian vault; `jax-examples/` = Colab notebooks).

---

## 0. Consolidated design decisions

1. **Day 1 = self-contained global introduction.** No hands-on block. B1 gives the big picture, motivation, course map, and a curated resource list. B2 is an **instructor-driven guided notebook walkthrough** (NB0) presenting the tooling: Python/Colab/GitHub, CPU vs GPU, JAX, and a from-scratch FCNN fitting a noisy 1D function via gradient descent. Students watch; the notebook is shared afterwards.
2. **Day 2 = domain shift & domain adaptation** (theory B1 + hands-on B2). The hands-on notebook (NB1) is a simplified **JAX** re-implementation of the instructor's `06_training_tools.ipynb` toy pipeline (from the `JPAS_Domain_Adaptation` repo).
3. **Day 3 = contrastive learning, representations & instance segmentation** (theory B1 + hands-on B2). The hands-on notebook (NB2) uses a CL loss (Weinberger-style discriminative loss as primary, InfoNCE as optional variant) on simple public data to _watch structures group together_ in an embedding, then inspects it with t-SNE/UMAP.
4. **Day 4 = the two research case studies**, presented broadly (concepts over technical details), explicitly harvesting everything built in Days 1–3. B1 = instance segmentation of proto-haloes (A&A 685, A37); B2 = J-PAS sim-to-obs SSDA (arXiv:2602.13902) + panorama and closing.
5. **Only two true hands-on blocks** (L2_B2, L3_B2). NB0 is a guided demo, not an exercise session.
6. **Audience heterogeneity strategy**: assume physics background, not computational background. Every notebook has clearly marked 🟣 _"para quem quer mais"_ optional cells so advanced students self-serve without derailing pacing; basics are never skipped on their account.
7. **Recurring scaffolding**: every block opens with the _mapa do curso_ slide (current block highlighted), includes 1–2 show-of-hands polls, and closes with a one-line takeaway («se você lembrar de uma coisa…»).
8. Each 40-min block budgets **~36 min content + 4 min buffer**.

### Narrative arc (one line per day)

|Day|Theme (pt-BR tagline)|Question answered|
|---|---|---|
|L1|«O mapa e as ferramentas»|What is this field, why does it matter for physics, and what machinery will we use?|
|L2|«Quando os dados mudam: o gap simulação→observação»|Why do trained models fail on new data, and how do we adapt them?|
|L3|«Esculpindo representações: aprendizagem contrastiva»|How do networks learn geometric representations, and how do we group/inspect them?|
|L4|«Da teoria à fronteira: dois estudos de caso reais»|How do these exact ideas power current research at the department?|

**Structural note**: the two theory days and the two case studies form a chiasmus (L2 ↔ L4_B2, L3 ↔ L4_B1), so Day 4 opens with the topic freshest in memory (instance segmentation/CL from Day 3) and closes with the survey-data story (domain adaptation from Day 2), whose outlook (WEAVE, future surveys, careers) is the natural course finale.

---

## 1. Course at a glance — the 8 blocks

|Block|Title (pt-BR)|Type|Anchor asset|
|---|---|---|---|
|L01_B01|«Aprendizado de máquina e Física: o mapa do território»|Theory / panorama|Slides + resource list|
|L01_B02|«A caixa de ferramentas: Python, Colab e JAX na prática»|Guided demo|**NB0** `00_caixa_de_ferramentas.ipynb`|
|L02_B01|«Mudança de domínio: quando o treino não é a prova»|Theory|Slides|
|L02_B02|«Mão na massa I: quebrar e consertar um classificador»|Hands-on|**NB1** `01_domain_shift_toy.ipynb` (from `06_training_tools.ipynb`)|
|L03_B01|«Aprendizagem contrastiva: geometria da similaridade»|Theory|Slides|
|L03_B02|«Mão na massa II: esculpindo um espaço de embeddings»|Hands-on|**NB2** `02_contrastive_embeddings.ipynb`|
|L04_B01|«Estudo de caso I: prevendo a formação de halos com segmentação de instâncias»|Case study|A&A 685, A37 (2024)|
|L04_B02|«Estudo de caso II: do mock ao céu — adaptação de domínio no J-PAS» + encerramento|Case study + closing|arXiv:2602.13902|

---

## 2. LECTURE 1 — «O mapa e as ferramentas» (Tue Jul 21)

### L01_B01 — «Aprendizado de máquina e Física: o mapa do território»

- **Type**: Theory / panorama (slides, 2 polls, zero code).
- **Core intuition (EN)**: Situate ML within the current paradigm of physics and society, define the course's central object (learned representations), present the 4-day map, and hand students a curated self-study path so the course is a _launchpad_, not a closed box.
- **Deliverables to develop**: slide deck; `course-materials/L01_B01.md`; printable/QR resource sheet.

**Chronograph (40')**

|Time|Segment|What & how|
|---|---|---|
|0–5'|Cold open: why now?|One striking slide-pair: (a) data volume of modern surveys (J-PAS/LSST-scale: more data per night than a human can inspect in a lifetime) vs. (b) 2024 Nobel Prize in Physics (Hopfield & Hinton) — "a física emprestou ferramentas ao ML; agora o ML devolve o favor." Poll #1: "quem já usou alguma ferramenta de ML (incluindo ChatGPT)?" — expect ~100%; punchline: "usar todos usam; este curso é sobre _entender_."|
|5–12'|ML in the scientific method|Where ML sits in physics today: emulating expensive simulations, pattern discovery in data deluges, inverse problems, experiment control. Honest counterpoint slide: what ML does _not_ do (replace understanding; extrapolate reliably; certify its own errors) — frames the course's critical stance. Brief societal contextualization (medicine, climate, language models) kept to one slide.|
|12–19'|The central concept: representations|The course's thesis in one visual: raw data are points in huge spaces with misleading distances; networks learn _coordinates where distance = meaning_. Physics anchor: choosing good variables (center-of-mass, normal modes) — "vocês já fazem isso; redes automatizam." Vocabulary planted (encoder, espaço latente, embedding) at picture level only — each returns with depth on Days 2–3.|
|19–26'|The course map|The 4-day arc walked slowly, one slide per day, each with its guiding question and its "payoff figure" teaser (Day 2: a decision boundary failing on shifted data; Day 3: an embedding self-organizing; Day 4: real halo shapes + real quasar confusion matrices). Logistics: repo QR, notebook policy (Day 1 demo-only; Days 2–3 bring laptop _if you can_, not required), how the Markdown study guides work.|
|26–33'|The self-study map (resources)|Curated, opinionated resource walkthrough (see §6): "se você só tiver 3 horas" → 3Blue1Brown NN series; "se você tiver um mês" → Andrew Ng; "se você quiser o livro" → Nielsen / Prince; "se você quiser física+ML" → Carleo et al. review. Each with one sentence on _what it's uniquely good for_. Poll #2: "quantos já assistiram algum vídeo do 3Blue1Brown?" (calibrates the room).|
|33–36'|Takeaway + bridge|«Se você lembrar de uma coisa: aprender é escolher coordenadas — e este curso ensina a escolhê-las, adaptá-las e desconfiar delas.» Bridge: "depois do intervalo: as ferramentas concretas que vamos usar a semana inteira."|
|36–40'|Buffer|Questions.|

**Instructor pitfalls**: resist teaching NN mechanics (the separate intro lecture owns that); keep the societal slide tight (it can eat 10 min); the resource segment must be a _tour_, not a reading of URLs — the Markdown file carries the links.

### L01_B02 — «A caixa de ferramentas: Python, Colab e JAX na prática» (guided demo, NB0)

- **Type**: Instructor-driven notebook walkthrough. **Students do not need machines.** Notebook shared afterwards with extended comments.
- **Core intuition (EN)**: Demystify the full working environment (Colab, GitHub, hardware, JAX) and consolidate the intro lecture's NN concepts by _building_ a from-scratch FCNN in JAX and fitting a noisy 1D function with gradient descent — the visual, minimal "hello world" of everything that follows.
- **Deliverables to develop**: **NB0** (spec in §5.1); `course-materials/L01_B02.md`.

**Chronograph (40')**

|Time|Segment|What & how|
|---|---|---|
|0–6'|The environment tour|Live: open Colab from the repo badge; what a notebook _is_ (cells, kernel, state); where this lives (GitHub — 60-second "what is a repo" for non-coders); Runtime menu: CPU vs GPU vs TPU with a one-slide mental model ("CPU = poucos doutores; GPU = milhares de estagiários fazendo a mesma conta") and a live timing cell (matrix multiply on CPU vs GPU).|
|6–12'|The scientific Python stack in 5 cells|`numpy` arrays as the lingua franca; `matplotlib` in one cell; then **JAX**: "NumPy que sabe derivar e que corre em GPU." The three superpowers each in one line: `jax.numpy` (drop-in), `jax.grad` (autodiff), `jit/vmap` (speed). Demo: `jax.grad` of x² and of a hand-written potential — "derivadas exatas de código arbitrário; é isto que torna o treino possível."|
|12–30'|The core exercise: FCNN from scratch fits a noisy function|The heart of the block, built live cell by cell (all pre-tested; PRETRAINED fallback exists): (1) sample y = f(x) + ruído (f = damped sine or similar physics-flavored curve); (2) define an MLP _from scratch_ — params as a list of (W, b), forward pass in ~10 lines with `tanh`; (3) MSE loss; (4) the training loop: `grads = jax.grad(loss)(params)` + explicit SGD update — **no optimizer library**, so students see backprop is "just" the chain rule handled by autodiff plus a descent step; (5) the money plot: animated/looped fit snapshots at epochs 0, 10, 100, 1000 over the noisy data. Polls embedded: "o que acontece se a rede for grande demais e treinarmos para sempre?" → run the overfit cell → the wiggly interpolation of noise → first encounter with **generalization**, the seed concept for Day 2.|
|30–36'|Vocabulary consolidation map|One slide mapping what was just done to the standard jargon (modelo/parâmetros/perda/gradiente/época/treino vs. validação) and to what the intro lecture taught. Explicit pointers: "quarta-feira este mesmo loop vira um classificador; quinta, uma perda contrastiva." Takeaway: «treinar uma rede = descer o gradiente de uma função de perda; todo o resto é engenharia em volta disso.»|
|36–40'|Buffer|Questions; announce that NB0 (with extended comments + exercises) is now in the repo.|

**Instructor pitfalls**: the environment tour can silently consume 15 min — timebox hard; do NOT introduce `optax`/`flax` (raw pytrees keep the magic visible); pre-run the GPU timing cell (Colab GPU allocation can lag); have static screenshots of every output as backup against wifi failure.

---

## 3. LECTURE 2 — «Quando os dados mudam» (Wed Jul 22)

### L02_B01 — «Mudança de domínio: quando o treino não é a prova»

- **Type**: Theory.
- **Core intuition (EN)**: Standard ML silently assumes train ≈ test; reality (and especially sim-trained science) breaks this constantly, and models fail _confidently_. Give students the taxonomy, the detection habits, and the map of mitigation strategies — the vocabulary and mindset to recognize when these tools apply in their careers.
- **Deliverables**: slide deck; `course-materials/L02_B01.md`.

**Chronograph (40')**

|Time|Segment|What & how|
|---|---|---|
|0–4'|Mapa + recap|Highlight Day 2; 90-sec recap of NB0's overfitting cell: "generalizar _dentro_ da mesma distribuição já era difícil; hoje: e quando a distribuição muda?"|
|4–11'|The phenomenon, everywhere|The i.i.d. assumption exposed with universal examples first: speech models meeting the Paulistano accent; a hospital's diagnostic model failing on another hospital's scanner; sunny-simulator self-driving meeting fog. Then science: simulation-trained models meeting real instruments. The cultural anchor analogy: **«estudar pelo simulado, fazer a prova de verdade»** (ENEM/vestibular). Poll: "quem já treinou/testou algo em dados de origens diferentes?"|
|11–18'|Taxonomy with cartoons|Three named shifts, each ONE 2D scatter cartoon: **covariate shift** (p(x) moves, rule intact), **prior/label shift** (class frequencies change), **concept shift** (the rule itself changes). Vocabulary stays on-screen all block. Sim-to-real framed as the scientist's chronic case: simulations = free labels + imperfect physics; observations = real physics + scarce labels.|
|18–24'|Silent failure & calibration|The scariest slide: a confidently wrong softmax under shift. **«Errado e confiante é o modo de falha mais perigoso da ciência com ML.»** Analogy: **termômetro descalibrado** (precise ≠ accurate). Reliability diagram introduced visually; calibration named as a property distinct from accuracy.|
|24–32'|The mitigation map|The strategy panorama, organized by _what you have_: (i) nothing but source → data augmentation & domain randomization ("treine no caos para ser robusto ao real"); (ii) unlabeled target → distribution alignment (a critic that tries to tell domains apart — adversarial alignment at cartoon level) and importance reweighting; (iii) **a few target labels → transfer learning / fine-tuning / SSDA**: pretrain-then-adapt, what to freeze vs. retrain, early-layers-generic vs. late-layers-specific. The **encoder + head** decomposition introduced HERE (it is tomorrow's and Day 4's load-bearing concept): analogy — **sotaques**: adapting to a new accent recalibrates perception (encoder) without relearning vocabulary (head).|
|32–36'|Detection habits + takeaway|Practical checklist: compare feature histograms; train a source-vs-target _domain classifier_ ("se dá para distinguir os domínios, há shift"); monitor embedding drift. Takeaway: «desconfie do softmax: confiança não é competência — e adaptar é barato se você souber o que congelar.» Teaser: "no próximo bloco vamos quebrar um classificador de propósito e consertá-lo com exatamente estas ideias."|
|36–40'|Buffer|Questions.|

**Instructor pitfalls**: no equations beyond p_source(x,y) ≠ p_target(x,y); resist telling the J-PAS story now (Day 4's payoff depends on it staying fresh); the mitigation map must remain a _map_ — depth arrives in NB1 and Day 4.

### L02_B02 — «Mão na massa I: quebrar e consertar um classificador» (hands-on, NB1)

- **Type**: Hands-on (instructor executes and narrates; students with laptops follow via badge; nothing depends on their machines).
- **Core intuition (EN)**: The full domain-shift lifecycle in 2D where _everything is visible_: train on source, watch confident failure on target, then compare the three regimes (zero-shot / target-only / freeze-head-adapt-encoder) — the exact experimental design of Friday's J-PAS paper, in a toy.
- **Primary source**: instructor's `06_training_tools.ipynb`, simplified & ported to JAX (spec §5.2).
- **Deliverables**: **NB1**; `course-materials/L02_B02.md`.

**Chronograph (40')**

|Time|Segment|What & how|
|---|---|---|
|0–3'|Setup + framing|Run-all immediately; frame: "hoje o notebook é um laboratório de patologia: causar a doença, diagnosticar, tratar." Traffic-light cell convention reminder (🟢 conceito / 🔵 código / 🟡 pergunta / 🟣 opcional).|
|3–10'|Act 1 — the toy universe|Generate the 2D Gaussian-mixture data: 6 classes, **imbalanced** proportions, and two domains — _source_ and _target_ — where some class clouds moved (visualize side by side; students literally SEE covariate shift, and the imbalance motivates the weighted cross-entropy, shown in one cell). 🟡 "olhe as duas figuras: quais classes vão sofrer mais?"|
|10–17'|Act 2 — train on source, break on target|Encoder (MLP → 2D latent) + head (→ 6 classes), reusing NB0's from-scratch style. Train on source (fast); show the **decision-map** figure (predicted-class regions painted over the plane, the notebook's signature visual) with source points → beautiful; overlay target points → visible catastrophe. Confusion matrix source vs. target side by side. Confidence histogram: **still confident while wrong** (yesterday's warning, now on screen).|
|17–23'|Act 3 — diagnose|Train the 2-line domain classifier (source vs target): high AUC = shift detected without any target labels. Latent-space scatter of source vs target: the clouds don't overlap — "o shift é visível no espaço latente."|
|23–32'|Act 4 — the three-regime experiment|The block's centerpiece, mirroring the paper's design: **(A) zero-shot** (source model applied raw); **(B) target-only** trained from scratch on K target labels (K small — it struggles, especially on rare classes); **(C) SSDA**: load the pretrained model, **freeze the head, adapt only the encoder** on the same K labels. Show the decision maps + confusion matrices of the three; the summary figure: accuracy/macro-F1 vs. K for the three regimes (curves cross!). 🟡 "por que congelar a CABEÇA e não o encoder? (dica: sotaques)". Latent scatter after SSDA: target clouds moved into the head's fixed class regions — "adaptamos a percepção, não os conceitos."|
|32–36'|Takeaway|«Quebrar é fácil, falhar em silêncio é perigoso, adaptar é barato — se você souber o que congelar.» Teaser: "sexta-feira: este experimento exato, com quasares de verdade." Take-home exercises flagged (change shift magnitude; prior shift variant; sweep K).|
|36–40'|Buffer|Questions.|

**Instructor pitfalls**: total training time across all cells must stay < 3 min on Colab CPU (pretrained fallbacks mandatory); do not open the t-SNE cells here (🟣 optional — t-SNE is Day 3's material and must not be spoiled beyond a glimpse); keep the weighted-CE explanation to one sentence + one cell (imbalance is a supporting theme, not the theme).

---

## 4. LECTURE 3 — «Esculpindo representações» (Thu Jul 23)

### L03_B01 — «Aprendizagem contrastiva: geometria da similaridade»

- **Type**: Theory.
- **Core intuition (EN)**: One coherent narrative from "distances in raw data lie" to "define similarity, geometrize it": embeddings → contrastive losses as interaction potentials → where positive pairs come from (labels, augmentations = invariance declarations) → clustering the result → inspecting it with t-SNE/UMAP (which are secretly contrastive too) → and the payoff application: instance segmentation.
- **Deliverables**: slide deck; `course-materials/L03_B01.md`.

**Chronograph (40')**

|Time|Segment|What & how|
|---|---|---|
|0–4'|Mapa + the puzzle|Recap Day 2's latent scatters ("o espaço latente era o palco; hoje ele é o protagonista"). The opening puzzle: image of a digit, the same digit shifted 3 px, and a different digit — poll: "qual par está mais perto em distância de pixels?" → reveal: the shifted copy is FARTHER. «Distâncias cruas mentem; precisamos aprender a métrica.»|
|4–10'|Embeddings & the label economy|Encoder as cartographer: map data into a space where distance = similarity _of meaning_. Then the economics: labels cost telescope time/expert hours; raw data is nearly free → the supervised/semi/self-supervised taxonomy as **budget strategies**. Self-supervision in one slide: the data supervises itself (fill the gap, predict the next word) — "é assim que os grandes modelos de linguagem são treinados."|
|10–18'|The contrastive principle + the physics of it ⭐|The one-sentence definition: **escolha quais pares devem ficar perto (positivos) e quais devem ficar longe (negativos); a rede geometriza a sua escolha.** Then the block's signature move — contrastive losses as **interaction potentials**: positives attract like springs (with slack δ_pull), cluster centers repel like charges (with cutoff δ_push); training = overdamped relaxation of a particle system. Write the pull/push terms in physics notation (this IS the Weinberger loss, though the name arrives tomorrow). The degenerate-minimum exercise: "e se só houver atração?" → total collapse → why negatives (or tricks) exist. Mention InfoNCE/SimCLR as the softmax-temperature version (Boltzmann weights over negatives) in one slide.|
|18–24'|Where do pairs come from?|Two answers: (i) **labels** → supervised metric learning (face-ID in their pockets: unlocking a phone = a distance test in an embedding); (ii) **augmentations** → two distorted views of the same object declared "the same". The deep slide: **cada augmentação é uma declaração de invariância** — rotation says orientation is meaningless; noise says this noise level is instrumental, not physical. "Escolher augmentações é declarar as simetrias do seu problema" — Noether-flavored, and exactly what astronomers do (PSF, sky orientation, calibration).|
|24–30'|Harvesting the embedding: clustering & projections|Once the space is sculpted: **clustering** (k-means/density-based in 2 slides) finds the groups — with the caveat that clustering quality = representation quality ("lixo nas coordenadas, lixo nos clusters"). **t-SNE & UMAP** as inspection tools for high-D embeddings: what they do (preserve neighborhoods), how they mislead (distances between clusters ≈ meaningless; perplexity changes the picture), the astronomy-native warning: **constelações** — apparent groupings that are projection artifacts. The elegant unification (one slide, optional depth): t-SNE and UMAP are themselves contrastive methods — attraction between neighbors, repulsion between non-neighbors (Damrich et al. 2022) — "a mesma física de molas e cargas, três nomes diferentes."|
|30–36'|The payoff problem: instance segmentation|Definition ladder in one visual: **semantic** ("há floresta aqui?") vs. **instance** ("qual pixel pertence a qual árvore?") vs. panoptic. Why instance is mathematically awkward: variable number of objects + permutation-invariant labels → no direct differentiable loss → the workaround: **embed-then-cluster** — map every pixel/particle into a learned space where instances become clusters, then harvest. "Vocês agora possuem todas as peças desta máquina." Takeaway: «similaridade não se descobre — se define; a rede só geometriza a sua definição.» Teaser: "no próximo bloco, vamos VER um espaço se organizar em tempo real."|
|36–40'|Buffer|Questions.|

**Instructor pitfalls**: the potentials slide is the course's peak — rehearse it; do NOT name-drop the Weinberger/halo application yet (Day 4's recognition moment depends on it); keep t-SNE/UMAP mechanics at picture level (the hands-on shows the knobs); the taxonomy slide must not balloon.

### L03_B02 — «Mão na massa II: esculpindo um espaço de embeddings» (hands-on, NB2)

- **Type**: Hands-on (same execution model as NB1).
- **Core intuition (EN)**: Watch contrastive learning happen: first as a pure particle system relaxing under pull/push potentials (no network!), then with a real encoder on simple public data, ending with clustering + t-SNE/UMAP inspection — the complete embed-then-cluster pipeline of instance segmentation, in miniature.
- **Deliverables**: **NB2** (spec §5.3); `course-materials/L03_B02.md`.

**Chronograph (40')**

|Time|Segment|What & how|
|---|---|---|
|0–3'|Setup + framing|Run-all; frame: "primeiro dinâmica molecular; depois deep learning; no fim vocês percebem que era a mesma coisa."|
|3–12'|Act 1 — the particle sandbox (no network)|2D points with instance labels; pull/push/reg potentials in JAX (~15 lines); relaxation via `jax.grad` + descent, **animated live**. Sliders/params for δ_pull, δ_push. 🟡 before running: "o que acontece com c_push = 0?" → run → total collapse (the trivial minimum, predicted by the audience). This cell is the course's signature visual.|
|12–24'|Act 2 — a real encoder on public data|Small public dataset (MNIST subset primary; 🟣 optional Galaxy10-DECals variant for astro flavor): encoder MLP → 2D embedding trained with the same discriminative (Weinberger-style) loss, labels as pair-generators. Live evolution: scatter of the embedding at epochs 0 / early / late — classes visibly condensing into clouds (pretrained checkpoints for each stage; the "evolution filmstrip" is pre-rendered as fallback). 🟡 "que classes se misturam? faz sentido visual?" (e.g., 4/9, 3/8 — 'degenerescências físicas' of handwriting).|
|24–31'|Act 3 — harvest: cluster, then project|Run a simple clustering algorithm on the embedding → recover the classes _as if we had never had labels at test time_ → "isto É segmentação de instâncias: embed, depois cluster." Then inspection tools: t-SNE and UMAP on a higher-D (e.g., 16-D) version of the embedding, with 2–3 perplexity/n_neighbors values → the picture _changes_ → the constelações warning made concrete.|
|31–36'|Takeaway|«Um bom espaço de embeddings transforma um problema sem função de perda num problema de clustering.» Teaser: "amanhã: esta máquina exata prevendo onde nascem os halos de matéria escura — e o notebook de ontem virando quasares reais." Take-home flagged (swap loss to InfoNCE-with-augmentations = mini-SimCLR, provided as 🟣 section; change δ values; cluster with different algorithms).|
|36–40'|Buffer|Questions.|

**Instructor pitfalls**: Act 1 must not exceed 12 min (it is delightful and dangerous); MNIST download can be slow on venue wifi → ship a cached `.npz` subset in the repo; t-SNE on >5k points is slow → subsample aggressively; the mini-SimCLR belongs in 🟣 (augmentation-based CL was theory; hands-on time only covers the supervised-pairs version).

---

## 5. LECTURE 4 — «Da teoria à fronteira» (Fri Jul 24)

### L04_B01 — «Estudo de caso I: prevendo a formação de halos com segmentação de instâncias»

- **Type**: Case study (slides built from paper figures; broad conceptual level, technical details only where they carry a lesson).
- **Primary resource**: López-Cano et al., A&A 685, A37 (2024) + `instance_halos` repo figures.
- **Core intuition (EN)**: Yesterday's embed-then-cluster machine, deployed on the universe: predict which particles of the initial conditions end in which dark-matter halo. Students should feel _recognition_ ("eu treinei essa perda ontem"), plus a deep physics coda on the irreducible limits of prediction.

**Chronograph (40')**

|Time|Segment|What & how|
|---|---|---|
|0–4'|Mapa + reframe|"Três dias de ingredientes; hoje, a cozinha de verdade. Regra do dia: vocês vão RECONHECER, não aprender do zero."|
|4–11'|The physics problem|Structure formation in 3 slides: initial near-homogeneous field → gravity amplifies → haloes as the scaffolding of galaxies. The Lagrangian question: _which initial particles end up in which halo?_ (proto-halo regions; why it matters: mass, spin, formation history; brief Lagrangian-vs-Eulerian vocabulary). Analogy: **meteorologia do universo** — given today's map, predict where the storms form.|
|11–17'|Why it's hard as ML — and the recognition moment ⭐|Variable number of haloes + permutation-invariant labels = yesterday's "no direct loss" problem. Reveal the paper's solution: map every particle into a **pseudo-space** where haloes are clusters — show the Weinberger loss equations NEXT TO a screenshot of yesterday's sandbox: pull, push, reg, term by term identical. «Vocês treinaram esta perda ontem, com bolinhas coloridas; aqui, as bolinhas são partículas do universo.»|
|17–26'|What the model achieves|Broad results tour via figures: semantic mask (which particles collapse at all); predicted vs. true Lagrangian halo shapes side by side (visually stunning; including disconnected regions that border-based methods like watershed cannot represent — one slide on that design choice as an honest "how researchers pick methods" lesson); halo mass function reproduced; ~minutes-per-simulation inference speed.|
|26–33'|The physics coda: limits of prediction|The twin-simulation experiment: perturb only unresolved small scales in the ICs → the two universes disagree on some membership assignments → part of the task is **aleatorically undetermined** → the model performs close to that ceiling. Lesson slide: «antes de julgar um modelo, pergunte qual é a nota máxima possível.» (Butterfly-effect resonance; also foreshadows honest ML evaluation culture.)|
|33–36'|Takeaway|«Segmentação de instâncias = esculpir um espaço onde objetos viram clusters — e o universo é segmentável.» Repo pointer for the curious.|
|36–40'|Buffer|Questions.|

### L04_B02 — «Estudo de caso II: do mock ao céu — adaptação de domínio no J-PAS» + encerramento

- **Type**: Case study + course closing.
- **Primary resource**: López-Cano et al., arXiv:2602.13902 + `JPAS_Domain_Adaptation` repo.
- **Core intuition (EN)**: Wednesday's three-regime toy experiment, at survey scale: a mocks-trained star/galaxy/quasar classifier degrades on the real sky; freezing the head and adapting the encoder with a small labeled set repairs it; calibrated probabilities become telescope economics. Close the course by lighting up the whole map and pointing at the frontier.

**Chronograph (40')**

|Time|Segment|What & how|
|---|---|---|
|0–7'|The data & the stakes|J-PAS in 3 slides: 54 narrow-band filters → «fotometria fingindo ser espectroscopia» (show real J-spectra of the 4 classes: star, galaxy, QSO low-z, QSO high-z). Why classify: candidate lists for spectroscopic follow-up (WEAVE) — false positives = wasted fibers = money and photons; hence **calibrated probabilities**, not just labels (Wednesday's thermometer comes home). One slide on the z≈2.1 class boundary: a continuous variable split by a threshold → unavoidable mixing (an "a física é difícil" error, to contrast with shift errors below).|
|7–13'|The setup — recognized|Mocks: DESI spectra projected through J-PAS filters = free labels ("o simulado"); a small leak-safe cross-matched real-label set ("as provas antigas"); the _same encoder+head architecture and the same three regimes as NB1_, shown as a literal side-by-side with Wednesday's notebook figures. «Vocês rodaram este experimento na quarta-feira, em 2D.»|
|13–21'|The payoff|The confusion-matrix reveal sequence: in-domain mocks (tight) → zero-shot on real sky (STAR→QSO_high confusion jumps ~1.2% → 9.2%) → after SSDA (back to ≈0.9%, near in-domain). Headline metrics broadly: macro-F1 0.73 (zero-shot) → 0.79 (target-only) → 0.82 (SSDA); rare high-z quasars gain most (F1 0.37 → 0.55 → 0.66). The error-decomposition lesson: separate «a física é difícil» (z≈2.1 boundary, host-galaxy dilution, line aliases) from «meu treino estava enviesado» (domain shift) — "a habilidade mais transferível deste curso."|
|21–26'|From probabilities to operations|Calibration results (reliability, ECE≈0.05) and how calibrated per-class probabilities → purity/completeness trade-offs → fiber-budget forecasts for WEAVE-QSO. "É aqui que uma curva ROC vira decisão de telescópio."|
|26–33'|Panorama & how to get in|The frontier in 3 slides: contrastive/self-supervised pretraining for cross-survey embeddings; foundation models in astronomy; photo-z and simulation-based inference. Concrete entry points for THEM: both public repos as thesis/IC starting material; FAPESP IC/master opportunities; the department's groups. Relight the full course map, all 8 blocks.|
|33–36'|Final takeaway|«Representações são coordenadas; coordenadas se esculpem (contrastiva); esculturas quebram quando o mundo muda (shift); e adaptá-las é barato — se você souber o que congelar.» Feedback QR.|
|36–40'|Buffer|Final questions.|

**Instructor pitfalls (both blocks)**: the temptation to go technical is maximal here — every equation shown must have appeared (in toy form) on Days 1–3; keep paper-internal details (hyperparameter sweeps, architecture tables, appendices) out; rehearse the two "recognition" slides (Weinberger↔sandbox; three-regimes↔NB1) — they are the course's emotional payoff.

---

## 6. Notebook specifications (`jax-examples/`)

**Shared conventions**: Colab badge at top; pinned versions in a Setup cell; `PRETRAINED = True` flags loading cached weights/figures from the repo so **Run-all completes in < 3 min on free Colab CPU**; cell taxonomy 🟢 conceito / 🔵 código / 🟡 pergunta-relâmpago / 🟣 para quem quer mais; all student-facing text in **pt-BR**; 2–3 take-home exercises at the end; static PNG fallbacks of every key figure committed to the repo (wifi insurance).

### 6.1 NB0 — `00_caixa_de_ferramentas.ipynb` (L01_B02, guided demo)

```
🟢 0. Bem-vindo(a): o que é este ambiente (Colab, células, kernel, GitHub)
🔵 1. Hardware: célula de timing — multiplicação de matrizes em CPU vs GPU
🔵 2. NumPy essencial em 3 células (arrays, slicing, broadcasting) + matplotlib em 1
🔵 3. JAX = "NumPy que sabe derivar":
      3.1 jax.numpy drop-in;  3.2 jax.grad de x² e de um potencial qualquer
      3.3 (🟣) jit e vmap com micro-benchmark
🔵 4. O EXERCÍCIO CENTRAL — regressão com uma FCNN do zero:
      4.1 Dados: y = f(x) + ruído  (f = senoide amortecida, sabor físico)
      4.2 MLP do zero: params = [(W1,b1),(W2,b2),...], forward ~10 linhas (tanh)
      4.3 Perda MSE;  4.4 Loop de treino: grads = jax.grad(perda)(params);
          atualização SGD explícita (SEM biblioteca de otimizador)
      4.5 Figura-troféu: ajuste nas épocas 0 / 10 / 100 / 1000 sobre os dados
      🟡 O que acontece com uma rede grande treinada "para sempre"? → célula
          de overfitting → interpolação do ruído → GENERALIZAÇÃO (semente do Dia 2)
🟢 5. Mapa de vocabulário: o que fizemos ↔ jargão padrão ↔ aula introdutória
🟣 6. Para quem quer mais: trocar tanh/ReLU; largura×profundidade; learning rate
🟡 7. Para casa: ajustar outra f(x); dividir treino/validação e plotar as duas perdas
```

### 6.2 NB1 — `01_domain_shift_toy.ipynb` (L02_B02) — JAX port of `06_training_tools.ipynb`

Simplifications vs. the research notebook: pure JAX (no torch, no external repo modules — all helper functions defined inline or in one small `utils.py`); 6→**4 classes** (less visual clutter, keeps one rare class); single flat config cell replacing the multi-regime comment-toggling (regimes become three explicit, named sections); dropout/batchnorm removed; keep: imbalance + weighted CE, encoder(→2D latent)+head split, decision-map plots, per-class probability maps (🟣), confusion matrices, latent scatters.

```
🟢 0. O ciclo de vida do domain shift em 4 atos
⚙️ 1. Setup (+ PRETRAINED flags)
🔵 2. ATO 1 — O universo de brinquedo:
      2.1 Misturas gaussianas 2D, 4 classes DESBALANCEADAS; domínios
          Source e Target com nuvens deslocadas (specs lado a lado)
      2.2 Visualização Source vs Target (KDE + scatter)
      🟡 Quais classes vão sofrer mais com o shift?
🔵 3. ATO 2 — Treinar na fonte, quebrar no alvo:
      3.1 Encoder MLP → latente 2D → cabeça → 4 classes (estilo NB0)
      3.2 Cross-entropy PONDERADA (1 célula: por que pesos ∝ 1/frequência)
      3.3 Treino no Source [PRETRAINED fallback]
      3.4 Figura-assinatura: MAPA DE DECISÃO no plano + pontos Source → lindo;
          sobrepor Target → catástrofe visível
      3.5 Matrizes de confusão Source vs Target; histograma de confiança
          (errado E confiante)
🔵 4. ATO 3 — Diagnosticar sem rótulos do alvo:
      4.1 Classificador de domínio Source-vs-Target (AUC alto = shift)
      4.2 Scatter latente Source vs Target (nuvens separadas)
🔵 5. ATO 4 — Os três regimes (o desenho experimental do artigo de sexta):
      5.1 (A) Zero-shot;  5.2 (B) Só-alvo com K rótulos, do zero;
      5.3 (C) SSDA: carregar pré-treinado, CONGELAR a cabeça, adaptar o encoder
      5.4 Comparação: mapas de decisão + confusões dos 3 regimes
      5.5 Figura-síntese: métrica × K para A/B/C (as curvas se cruzam)
      🟡 Por que congelar a CABEÇA e não o encoder? (dica: sotaques)
      5.6 Latente pós-SSDA: as nuvens do alvo entram nas regiões fixas da cabeça
🟣 6. Para quem quer mais: mapas de probabilidade por classe; shift de PRIOR;
      espiar o latente com t-SNE (gancho para amanhã)
🟡 7. Para casa: variar magnitude do shift; varrer K até B vencer C — o que
      isso diz sobre orçamentos de rótulos?
```

### 6.3 NB2 — `02_contrastive_embeddings.ipynb` (L03_B02)

Primary data: cached MNIST subset (`.npz` in repo, ~4k samples); 🟣 optional astro-flavored variant: Galaxy10-DECaLS subset (pre-cached small split) — same code path.

```
🟢 0. A tese: "perda contrastiva = potencial de interação"
⚙️ 1. Setup (+ dados em cache no repo, sem downloads externos)
🔵 2. ATO 1 — Sandbox de partículas (SEM rede neural):
      2.1 200 pontos 2D com rótulos de instância (5 grupos)
      2.2 Potenciais em JAX:
          L_pull = ⟨max(‖μ_c − x_i‖ − δ_pull, 0)²⟩
          L_push = ⟨max(δ_push − ‖μ_c − μ_c'‖, 0)²⟩ ;  L_reg = ⟨‖μ_c‖²⟩
      2.3 Relaxação: x ← x − η·jax.grad(L)(x), ANIMADA
      🟡 Antes de rodar: o que acontece com c_push = 0? → colapso (mínimo trivial)
      🟣 Versão softmax/temperatura (pesos de Boltzmann sobre negativos)
🔵 3. ATO 2 — Encoder real, dados públicos:
      3.1 Carregar subset MNIST em cache (🟣 variante Galaxy10)
      3.2 Encoder MLP → embedding 2D; MESMA perda pull/push, rótulos geram pares
      3.3 Filme da evolução: scatter do embedding nas épocas 0/early/late
          [checkpoints PRETRAINED por estágio; filmstrip PNG de fallback]
      🟡 Que classes se misturam? (4/9, 3/8 — "degenerescências" da escrita)
🔵 4. ATO 3 — Colher o espaço: cluster + projeções:
      4.1 Clustering simples no embedding → classes recuperadas SEM usar
          rótulos na inferência → "isto É segmentação de instâncias"
      4.2 Versão 16-D do embedding → t-SNE e UMAP com 2–3 valores de
          perplexity/n_neighbors → a figura MUDA → aviso das constelações
🟢 5. Resumo: embed-then-cluster; ponte para os halos de amanhã
🟣 6. mini-SimCLR: pares por AUGMENTAÇÃO (ruído/shift) + InfoNCE — a versão
      auto-supervisionada, pronta para rodar em casa
🟡 7. Para casa: variar δ_push e ver a geometria final; trocar o algoritmo de
      clustering; rodar a variante Galaxy10
```

---

## 7. Resource list for students (deliver in L01_B01; lives in `course-materials/L01_B01.md`, links + one-line pt-BR blurbs)

**Vídeo — intuição visual**

- **3Blue1Brown — série "Neural Networks"** (YouTube; legendas em pt): a melhor intuição visual de redes, gradiente e backprop que existe. «Se você só tiver 3 horas, gaste-as aqui.»
- **StatQuest (Josh Starmer)**: conceitos de estatística e ML explicados um a um, do zero, com bom humor.
- **Welch Labs / Artem Kirsanov** (🟣): visualizações mais avançadas de aprendizado e representações.

**Cursos estruturados**

- **Andrew Ng — Machine Learning Specialization (Coursera/Stanford, audit grátis)**: o caminho clássico e completo para fundamentos. «Se você tiver um mês.»
- **Andrew Ng — Deep Learning Specialization** (🟣): o passo seguinte, focado em redes profundas.
- **fast.ai — Practical Deep Learning for Coders**: abordagem "código primeiro", excelente para quem aprende fazendo.

**Livros (gratuitos online)**

- **Michael Nielsen — _Neural Networks and Deep Learning_**: curto, interativo, matemática acessível.
- **Simon Prince — _Understanding Deep Learning_ (2023, PDF grátis)**: o livro-texto moderno; figuras excepcionais.
- **Goodfellow, Bengio & Courville — _Deep Learning_** (🟣): a referência enciclopédica.

**Física ∩ ML**

- **Carleo et al. 2019, "Machine Learning and the Physical Sciences" (Rev. Mod. Phys. 91, 045002)**: o panorama da interseção.
- **Mehta et al. 2019, "A high-bias, low-variance introduction to ML for physicists" (Phys. Rep.)**: escrito POR e PARA físicos, com notebooks.
- **Lilian Weng — blog "Contrastive Representation Learning"** (🟣): a melhor revisão em formato blog do tema do Dia 3.

**Ferramentas**

- **Documentação do JAX (jax.readthedocs.io)** — em especial o tutorial "JAX 101".
- **Google Colab FAQ** + atalhos essenciais.
- **Os dois repositórios dos estudos de caso**: `daniellopezcano/instance_halos` e `daniellopezcano/JPAS_Domain_Adaptation`.

_(Verify all links at material-production time; add any IFUSP/local resources the instructor wants, e.g., department ML journal club.)_

---

## 8. Asset production checklist (feeds the next work phases)

|Asset|Block(s)|Status|Notes|
|---|---|---|---|
|Slide deck L1 (panorama + resources)|L01_B01|☐|Nobel/data-deluge cold open; course-map slides reused daily|
|**NB0** toolbox notebook|L01_B02|☐|Spec §6.1; static PNG fallbacks|
|`course-materials/L01_B01.md`, `L01_B02.md`|L1|☐|Dual-purpose format (instructor callout + pt-BR study guide)|
|Slide deck L2 (shift theory)|L02_B01|☐|Taxonomy cartoons; sotaques/simulado/termômetro visuals|
|**NB1** DA toy (JAX port of 06_training_tools)|L02_B02|☐|Spec §6.2; pretrained checkpoints for 3 regimes|
|`course-materials/L02_*.md`|L2|☐||
|Slide deck L3 (CL theory)|L03_B01|☐|Potentials slide = course centerpiece; rehearse|
|**NB2** contrastive embeddings|L03_B02|☐|Spec §6.3; cached MNIST npz (+ 🟣 Galaxy10); filmstrip fallback|
|`course-materials/L03_*.md`|L3|☐||
|Slide deck L4 (two case studies)|L04_B01/B02|☐|Built from paper figures; two "recognition" slides are critical|
|`course-materials/L04_*.md`|L4|☐||
|Resource sheet (QR + md)|L01_B01|☐|§7; verify links|
|Feedback form + QR|L04_B02|☐||

**Open questions to settle in the next iteration**

1. NB2 data: confirm MNIST-primary + Galaxy10-optional, or invert for astro flavor throughout?
2. L04 internal order: current plan = halos (B1) → J-PAS (B2, closes with panorama). Alternative = chronological by paper. Confirm.
3. Whether NB1 keeps 4 classes or mirrors the research notebook's 6.
4. Depth of the InfoNCE/temperature material in L03_B01 (currently one slide + 🟣 notebook section).
5. Poll mechanism for 130 students: show of hands vs. Mentimeter-style tool.