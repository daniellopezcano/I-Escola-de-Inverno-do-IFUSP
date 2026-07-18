# Enhancement brief — L3B2 notebook (02_contrastive_embeddings), full rebuild

## Purpose
Rebuild this notebook from scratch so it MIRRORS the theory block course-materials/L3B1.md
section by section, using MNIST as the single running toy dataset. Every major idea in
L3B1 should get a small, visual, self-contained demonstration here.

IMPORTANT: read the CURRENT course-materials/L3B1.md and map to its ACTUAL section
structure and numbering (it was recently renumbered). Do not trust section numbers quoted
from memory; derive the mapping from the file itself. In the notebook, reference the
corresponding L3B1 section by NAME in each part's opening markdown cell, so students can
navigate between theory and practice.

Output: jax-examples/notebooks/02_contrastive_embeddings_v2.ipynb
(leave the current notebook untouched). Resync the mirror from the canonical notebook
FIRST, per the source-of-truth policy in CLAUDE.md.

## Philosophy (inherited from L1B2 and L2B2 — match them)
- Self-contained, minimalistic, VISUAL. Students must not get lost in technicalities.
- Equinox + Optax, same style as the previous notebooks. Helpers defined INLINE where
  first needed — no big utility block at the top.
- Short cells, one idea per cell, pt-BR narrative, plots interleaved throughout.
- Every conceptual point should produce a FIGURE the instructor can lift into slides.
- Runtime modest on CPU (target: a few minutes total). Use a SUBSET of MNIST and small
  networks; prefer fewer epochs over big models.
- Artifact hygiene: download MNIST at runtime, save NO figures, write nothing to a
  committed path. Must run clean on a fresh Colab with an empty assets/.

## Structure — four parts

### Parte 1 — O espaço de pixels cru é difícil de interpretar
Mirrors L3B1's opening sections on latent spaces and why raw coordinates are pathological.
- Load a subset of MNIST at runtime (state the source; keep it small).
- Show several examples of the SAME digit class and make visible how DIFFERENT their raw
  pixel vectors are: e.g. plot the images side by side with their flattened vectors
  underneath, or a heatmap of several same-class vectors stacked.
- Quantify the problem concretely: compute raw pixel-space distances showing that two
  images of the same digit can be FARTHER apart than two images of different digits.
  Make this a small table or annotated figure — it is the notebook's opening punchline.
- A shifted/translated copy of a digit is a particularly effective demonstration: it is
  semantically identical but far away in pixel space.
- One short cell on dimensionality (784 dims) to make the high-dimensional point concrete.
Message: raw coordinates are a complex, high-dimensional, hard-to-interpret vector;
distance there does not mean similarity of meaning.

### Parte 2 — Um espaço latente 2D emerge ao treinar um classificador
Mirrors L3B1's treatment of latent spaces and their role.
- Build a SMALL CNN ENCODER with Equinox that maps the image to a 2-DIMENSIONAL latent
  vector, followed by a small classification head (2D → 10 classes).
  ARCHITECTURE NOTE: this is a bottleneck encoder, NOT a U-Net — do NOT add skip
  connections around the 2D layer; the whole point is that all information must pass
  through the 2-dimensional bottleneck.
- Train it as an ordinary supervised classifier with Optax (cross-entropy), same style as
  L1B2/L2B2. Show train/validation curves.
- THE KEY FIGURE: scatter the 2D latent representations of the validation set, coloured by
  digit class. Structure emerges — classes occupy distinct regions.
- Add a second visualization that makes it interpretable, e.g. overlaying a few actual
  digit thumbnails at their latent positions, and/or showing which classes end up adjacent
  (e.g. 4/9, 3/8) with one line of pt-BR narrative on why that is semantically sensible.
- Brief narrative: this is why latent spaces matter — structure, interpretability,
  reusability. Note that here the structure was obtained WITH labels; Parte 4 will ask
  whether we can get it WITHOUT them.

### Parte 3 — Augmentações e pares positivos/negativos
Mirrors L3B1's section on where positive pairs come from and augmentations as declarations
of invariance. This part is mostly VISUAL — the figures are the deliverable.
- Implement a small set of MNIST-appropriate augmentations (in JAX where practical):
  e.g. small random rotations, translations/shifts, scaling/zoom, elastic-style or mild
  distortions, cropping, added noise. Follow standard practice from the augmentation
  literature for MNIST-like data; keep the implementation simple and readable.
- FIGURE: a grid showing one original digit and several augmented views of it — the
  canonical "these are all the same thing" visual.
- FIGURE: explicitly illustrate a POSITIVE PAIR (two augmented views of the same image)
  versus NEGATIVE PAIRS (views of different images), laid out so the pairing logic is
  visually obvious.
- Short pt-BR narrative connecting to L3B1's key point: choosing augmentations = declaring
  which transformations should NOT change the meaning, i.e. declaring the invariances of
  the problem. Include a cautionary example if it fits cheaply: an augmentation that is
  too aggressive can destroy the label (e.g. large rotation making 6 look like 9) — a
  memorable lesson that invariance choices are problem-dependent.

### Parte 4 — Treinar com InfoNCE e recuperar a estrutura SEM rótulos
The notebook's final objective, mirroring L3B1's InfoNCE centrepiece.
- Reuse the same CNN encoder architecture (2D output, or a slightly higher-dim embedding
  reduced for plotting — prefer 2D for direct visualization and simplicity).
- Implement the InfoNCE loss explicitly and readably, matching L3B1's presentation:
  normalized embeddings, cosine similarity, temperature parameter, softmax over the
  positive against the negatives in the batch. Write it so the code lines map visibly onto
  the equation in L3B1 (this correspondence is a teaching goal).
- Train SELF-SUPERVISED: positives come from augmentations, NO labels used in training.
- THE CLIMAX FIGURE: scatter the learned 2D embedding of the validation set, coloured by
  the true digit labels — which the model never saw. Clustering by digit should be
  visible. Compare it side by side with Parte 2's supervised latent space.
- Evaluate the representation honestly: a simple linear probe (or k-NN) on the frozen
  embedding, reporting accuracy — showing a trivial classifier on top of the learned space
  performs well. Keep it to a couple of lines.
- If cheap, show the effect of the TEMPERATURE parameter with two values (one figure each
  or a two-panel figure) — this connects directly to L3B1's discussion. Mark as optional.
- Honest closing note (in L3B1's spirit): the self-supervised structure is usually less
  cleanly separated than the supervised one, and 2D is a severe constraint — real systems
  use much higher-dimensional embeddings and then project for visualization.

### Fechamento
- Short pt-BR summary tying the four parts back to L3B1's narrative.
- If t-SNE was covered in L3B1 and time/runtime allow, an OPTIONAL final cell: train a
  higher-dimensional embedding (e.g. 16D) and project it with t-SNE, illustrating both the
  tool and its caveats (vary perplexity, note that between-cluster distances are not
  meaningful). Mark clearly as optional/🟣 and keep it cheap.
- "Para casa" exercises at the end, in the style of the previous notebooks.

## Technical requirements
- Equinox + Optax; self-contained; no imports from external repos.
- MNIST downloaded/fetched at runtime; nothing committed; no figures saved to disk.
- Must run top-to-bottom clean in the WinterSchool kernel and on a fresh Colab.
- Keep the total runtime modest: subsample MNIST (e.g. a few thousand images), small CNNs,
  few epochs. Prefer a slightly worse-looking result over a long-running notebook.
- Colab badge at top; pt-BR markdown throughout.
