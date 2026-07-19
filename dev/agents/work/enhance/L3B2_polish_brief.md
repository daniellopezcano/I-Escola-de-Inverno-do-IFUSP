# Polish brief — L3B2 notebook (02_contrastive_embeddings), round 2

## Problem being fixed
In the current notebook the contrastive (InfoNCE) stage does NOT produce visible digit
clusters. Diagnosis — three compounding causes, all must be addressed:
 (1) the augmentations are far too weak, so InfoNCE can match positives via low-level
     shortcuts (total ink, coarse stroke position) without learning semantics;
 (2) the InfoNCE loss is applied DIRECTLY on a 2-dimensional embedding — a brutal
     constraint that standard practice avoids (representation vs. projection head);
 (3) too few negatives (batch too small) and probably too few epochs.
Also: too few points are plotted in the latent scatters to see structure.

Keep the four-part narrative, the pt-BR style, L1B2/L2B2 conventions, inline helpers,
short cells, visual-first philosophy. This is a refinement, not a restructure.

Output: jax-examples/notebooks/02_contrastive_embeddings_v2.ipynb (leave the canonical
notebook untouched). Resync the mirror from the canonical notebook FIRST per CLAUDE.md.

## R1 — MUCH stronger, literature-grounded augmentations (Parte 3)
MNIST is grayscale, so SimCLR's colour-jitter half of the recipe is unavailable (see
Chen et al. 2020; the UvA SimCLR tutorial drops colour distortion when colour matters).
The geometric augmentations must therefore carry the whole load and must be AGGRESSIVE.
Implement a composed pipeline, sampling several transforms per view:

- ELASTIC DEFORMATION (Simard, Steinkraus & Platt 2003, ICDAR — the classic MNIST
  augmentation): generate random displacement fields dx, dy ~ U(-1,1), smooth them with a
  Gaussian of width sigma, scale by alpha, and resample the image with bilinear
  interpolation. Expose alpha and sigma as parameters. This is the single most important
  addition — do not omit it.
- RANDOM RESIZED CROP / scale jitter: crop a random sub-region and resize back to 28x28.
  SimCLR identifies cropping as the most effective single augmentation; make it strong
  (e.g. covering a substantial but variable fraction of the image).
- AFFINE transforms (following Ciresan et al. 2010, arXiv:1003.0358, which combines
  affine + elastic for MNIST): random rotation and/or horizontal shear by an angle beta,
  and independent horizontal/vertical scaling gamma_x, gamma_y.
- TRANSLATION by a few pixels.
- GAUSSIAN BLUR and additive noise.
- CUTOUT / random erasing of a small patch (forces use of global shape, not one stroke).

Calibration requirements:
- Rotations must stay bounded (roughly within ±25-30 degrees) so 6 and 9 are not
  conflated — keep the existing cautionary note about this and make it explicit that
  this bound is a DOMAIN-SPECIFIC invariance decision (the L3B1 point).
- The composed views must be visibly, strongly distorted while a human can still read the
  digit. Include a figure showing a grid of MANY augmented views of the same digit to let
  the instructor (and the audience) judge the strength.
- Implement in JAX where practical; if an operation is far simpler with scipy/numpy at
  the data-preparation stage, that is acceptable — prioritize readability and runtime.
- Add a short pt-BR markdown cell citing Simard et al. 2003 and SimCLR, explaining WHY
  strong composition matters (weak augmentations let the network cheat with shortcuts).

## R2 — Fix the architecture: representation vs. projection head (Parte 4)
Follow standard SimCLR practice instead of forcing InfoNCE through 2 dimensions:
- CNN encoder -> REPRESENTATION h of moderate dimension (e.g. 32-64D)
- small MLP PROJECTION HEAD -> z (e.g. 32-64D, L2-normalized) where InfoNCE is computed
- After training, DISCARD the head; the representation h is what we analyse.
  Explain in one or two pt-BR lines why (the head's output is over-invariant; the base
  representation transfers better — Chen et al. 2020; UvA tutorial).
- VISUALIZE h with t-SNE, which directly exercises L3B1's t-SNE section and makes it
  pedagogically necessary rather than optional. Show a couple of perplexity values and
  keep L3B1's caveats (between-cluster distances are not meaningful).
- HIGH-VALUE TEACHING CONTRAST (include if runtime allows): also train a variant with the
  loss forced onto a 2D embedding and show that it fails to separate — an explicit
  demonstration of why the representation/projection split exists. Mark as optional if
  runtime is tight.

## R3 — Training regime that actually converges
- Increase the batch size substantially: InfoNCE quality scales with the number of
  negatives (each batch supplies 2(N-1) negatives). State this explicitly in a markdown
  cell — it connects to L3B1's discussion of negatives and to why memory banks/momentum
  encoders were invented.
- Train the contrastive stage for meaningfully more epochs than the supervised stage;
  contrastive learning converges more slowly. Plot the InfoNCE loss curve.
- Use a sensible temperature (SimCLR-range, ~0.1-0.5) and keep the temperature-comparison
  figure if cheap.
- Report a linear-probe (or k-NN) accuracy on the frozen representation as the objective
  measure that structure was learned — this must improve clearly versus the current
  version. If it does not, iterate on augmentation strength / epochs / batch size before
  declaring the build green.

## R4 — Plot many more points in the latent scatters (Partes 2 and 4)
- Make the train/validation split EXPLICIT and printed (sizes stated in a cell), so the
  reader knows exactly what is being trained on and plotted.
- Increase the validation subset used for the latent scatters substantially (target at
  least several thousand points, e.g. 5000-10000) so cluster structure is legible.
- Use small marker size, suitable alpha and a categorical colormap; add a legend keyed to
  digit class. The scatter figures are slide deliverables — make them publication-clean.
- Keep the MNIST subset used for TRAINING modest enough to respect the runtime budget,
  even while plotting many validation points (evaluation is cheap; training is not).

## R5 — Modest encoder upgrade (Parte 2 and 4)
Strengthen the CNN encoder a little (an extra conv stage and/or more channels) so it can
support the harder contrastive task, WITHOUT blowing up runtime. Keep the Parte 2
supervised model's 2D bottleneck as it is — that part works and its emergent-structure
figure is good; the architectural change in R2 applies to the contrastive stage.

## Constraints (unchanged)
- Equinox + Optax; self-contained; no external repo imports.
- MNIST fetched at runtime; NO figures saved; nothing written to a committed path;
  must run clean on a fresh Colab with an empty assets/.
- pt-BR narrative, Colab badge, "Para casa" exercises.
- Runtime: aim for a few minutes on CPU; the notebook may be somewhat heavier than before
  (that is expected), but if it exceeds roughly 8-10 minutes, cut the optional 2D-failure
  variant and the temperature comparison first, then reduce the training subset — never
  weaken the augmentations or the batch size, which are the fixes.
- Every part must still yield slide-ready figures.

## Definition of done
The final t-SNE plot of the contrastive representation must show visibly separated digit
groupings, and the linear probe on the frozen representation must report a clearly
non-trivial accuracy. Report both in the completion summary.
