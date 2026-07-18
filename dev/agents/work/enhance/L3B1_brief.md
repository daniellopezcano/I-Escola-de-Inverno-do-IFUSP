# Enhancement brief — L3B1 (teoria: espaços latentes, aprendizagem contrastiva, projeções)

## Role of this block in the course
Lecture 3, block 1 — the theory block whose companion hands-on is L3B2
(02_contrastive_embeddings). It follows L1 (foundations, "raw distances lie, learned
distances reveal") and L2 (domain shift and adaptation). It must feel like the natural
next step: L2 asked "what happens when the data distribution moves?"; L3 asks "how do we
BUILD the space in which distance means something in the first place?"

Read L1B1.md and L2B1.md first and honor their terminology, tone and the unifying
latent-space thread introduced in L1B1. L2B1 deliberately de-emphasized latent spaces
because they belong HERE — this block owns them.

## Time budget
40 minutes of lecture. Scope ruthlessly. Mark anything optional as cuttable so the
instructor can compress live. Prefer three ideas explained well over six sketched.

## Required content

### 1. Espaços latentes: o que são e por que dominam o ML atual
- Recall and deepen L1B1's thread: encoder maps raw data → latent vector; distance in the
  latent space is meant to encode similarity of MEANING, not of raw coordinates.
- Why raw coordinates are pathological: brief, concrete (two images of the same object
  differing by a small shift are far apart in pixel space; high-dimensional geometry is
  counterintuitive). Keep it to the essential intuition, no measure theory.
- THE ROLE OF LATENT SPACES IN MODERN ML — give this real weight, as requested:
  representations as the reusable currency of deep learning; pretrained encoders and
  transfer of representations; embeddings underlying retrieval, similarity search and
  recommendation; the representation layer inside large models generally. The message:
  much of modern ML is "learn a good representation once, reuse it for many tasks."
  Connect back to L2: transfer/fine-tuning works BECAUSE representations are reusable.
- Physics anchor (in the L1B1 spirit): choosing good coordinates — center-of-mass frame,
  normal modes — the encoder automates the change of variables.

### 2. Como se aprende um espaço latente: o princípio contrastivo
- The one-sentence principle: DEFINE which pairs should be close (positives) and which
  far (negatives); train the encoder so the geometry obeys that definition. Similarity is
  not discovered, it is DECLARED — the network only geometrizes the declaration.
- Where positive pairs come from — the two families:
  (a) LABELS → supervised metric learning (same class = positive). Everyday example:
      face recognition / phone unlocking is a distance test in an embedding.
  (b) AUGMENTATIONS → self-supervised. Two distorted views of the same sample are declared
      "the same thing". KEY PHYSICS POINT: choosing augmentations = declaring the
      invariances/symmetries of the problem (rotation says orientation is meaningless;
      added noise says that noise level is instrumental, not physical). This is the
      deepest bridge in the block — give it space.
- The label economy in one or two lines: labels are expensive, raw data is nearly free;
  hence the pull toward self-supervision.

### 3. As perdas contrastivas (a parte matemática — mantida acessível)
Give real mathematical content here, but compact, with every symbol defined.
- The attraction/repulsion structure as INTERACTION POTENTIALS: positives attract
  (springs, with slack given by a margin), negatives repel (with a cutoff distance);
  training is overdamped relaxation of a particle system in the latent space. For a
  physics audience this is the memorable framing — use it.
- Present a margin-based / triplet-style loss with explicit terms, and the
  pull/push/regularization decomposition (this same loss family reappears in L4's
  instance-segmentation case study — say so in one line, do not develop it here).
- Present InfoNCE / the SimCLR-style objective as the softmax-with-temperature version:
  write it, define the temperature, and note the statistical-mechanics reading (Boltzmann
  weights over negatives; low temperature focuses the loss on the hardest negatives).
- THE COLLAPSE PROBLEM: if you only pull positives together, the trivial minimum maps
  everything to one point. Negatives (or architectural tricks such as the BYOL-style
  asymmetry, mentioned in one line) exist to exclude that degenerate solution. Frame it
  as "find the flaw in this objective" — physicists enjoy spotting degenerate minima.
- Briefly name the main method families so students leave with keywords they can search:
  SimCLR, MoCo, BYOL, supervised contrastive, triplet/metric learning. One line each at
  most, with original references.

### 4. Projeções: visualizar espaços latentes de alta dimensão (t-SNE e UMAP)
The requested in-depth part — this is where the block gets its second dose of real math.
- The problem: embeddings live in tens-to-hundreds of dimensions; we can only look at 2D.
- t-SNE, explained properly but accessibly:
  * high-dimensional neighbor relations converted into conditional probabilities via
    Gaussian kernels; the PERPLEXITY parameter controls the effective neighborhood size
  * the low-dimensional map uses a heavy-tailed (Student-t) kernel — explain WHY: it
    mitigates the crowding problem, letting dissimilar points spread out
  * the objective is a KL divergence between the high- and low-dimensional neighbor
    distributions, minimized by gradient descent
  * define each symbol; keep the derivation to its logical skeleton, not full algebra
- UMAP in contrast, briefly: neighborhood-graph based, generally faster, tends to
  preserve more global structure; different assumptions, similar use.
- HONEST LIMITATIONS (essential — matches L1B1's "what ML does not do" spirit):
  distances BETWEEN clusters are largely meaningless; cluster sizes and densities are not
  faithful; the picture changes with perplexity/n_neighbors and with the random seed;
  apparent clusters can be artifacts. Give students a rule: use these maps to INSPECT and
  form hypotheses, never as evidence.
- A memorable astronomy-native warning: constellations — stars that look grouped in
  projection are light-years apart. Use it, briefly.
- One line connecting to the elegant unification: t-SNE and UMAP can themselves be read
  as contrastive methods (attraction between neighbors, repulsion between non-neighbors).
  Mention as an aside, do not develop.

### 5. Fechamento: para onde isto leva
- Clustering the learned space as the natural harvest (what L3B2 will do hands-on):
  a good representation turns a hard problem into an easy clustering problem.
- One or two lines pointing forward to L4: this exact machinery — a learned space where
  objects become clusters — is what the instance-segmentation case study uses; and the
  reusability of representations is what makes the domain-adaptation case study possible.
- Pointer back to the README / materials, in the L1B1 closing style.

## Explicitly OUT of scope (do not include)
- Do not re-teach domain shift/adaptation (L2 owns it); reference it only for continuity.
- Do not develop the instance-segmentation application (L4 owns it) — one line max.
- Do not duplicate the hands-on content of L3B2; the theory should SET UP the notebook
  (the particle-relaxation intuition, the clustering harvest, the t-SNE caveats), and may
  say "veremos isto na prática", but must not walk through the notebook's steps.
- No autoencoders/VAEs beyond, at most, a single sentence if needed for context —
  the block's spine is contrastive learning, not generative modeling.

## Style requirements
- pt-BR, narrative, compact, self-sufficient (the slides will be built FROM this file).
- Clear section headers so the deck's structure can be lifted directly.
- Equations in LaTeX where they earn their place; every symbol defined in ≤1 line.
- Inline references to ORIGINAL papers (arXiv/DOI): t-SNE (van der Maaten & Hinton 2008),
  UMAP (McInnes et al. 2018), SimCLR (Chen et al. 2020), MoCo (He et al. 2019),
  BYOL (Grill et al. 2020), supervised contrastive (Khosla et al. 2020), and the
  margin/discriminative-loss line of work. Closing "Referências" section.
- Frontmatter matching the other blocks, with a `slides:` placeholder.
