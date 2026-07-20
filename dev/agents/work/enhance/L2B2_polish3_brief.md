# Polish brief — L2B2 notebook (01_domain_shift_toy), round 3 — IN-PLACE EDIT

## SCOPE LOCK (critical)
Modify ONLY these two files, IN PLACE (no _v2 copies, no new versions):
  - jax-examples/notebooks/01_domain_shift_toy.ipynb
  - jax-examples/src_01_domain_shift_toy.py
Do NOT create, modify or delete ANY other file in the repository: no course-materials,
no README, no other notebooks, no requirements.txt, no assets, no agent files, no new
_v2/_v3 variants. You may READ other files for style reference, but write nothing else.

## Working procedure (source-of-truth aware)
1. The .ipynb is authoritative and may contain hand edits, so FIRST resync the mirror:
   $ENV/jupytext --to py:percent jax-examples/notebooks/01_domain_shift_toy.ipynb \
     -o jax-examples/src_01_domain_shift_toy.py
2. Apply all changes below to jax-examples/src_01_domain_shift_toy.py.
3. Regenerate the notebook from it and execute headless to verify:
   $ENV/jupytext --to ipynb jax-examples/src_01_domain_shift_toy.py \
     -o jax-examples/notebooks/01_domain_shift_toy.ipynb
   then nbconvert --execute --inplace with the WinterSchool kernel.
4. Iterate on the .py until the executed notebook is green; leave both files consistent.

## What is wrong with the current notebook
1. The class distributions are essentially regular blocks/blobs — far too simple. The
   trained classifier therefore learns boring, nearly-straight decision contours, which
   defeats the purpose of showing students how rich a neural decision boundary can be.
2. The sample sizes are too small, so scatter plots and probability maps look sparse and
   the metrics are noisy.
3. The domain shift is simplistic — a direct consequence of (1): blobs can only be moved.
Keep the seven-block structure, the pt-BR narrative, the Equinox/Optax style, the inline
helper philosophy and ALL existing evaluation machinery. This is a data-and-polish round,
not a restructure.

## R1 — Rebuild the data generator as a PARAMETRIC SHAPE FAMILY
Keep exactly THREE classes and keep the current imbalance (same proportions as now).
Replace blob sampling with non-trivial, curved, interleaved geometry. Each class must be
defined by a small set of GENERATIVE PARAMETERS — this is what makes R3 possible.

- Shapes must be NON-CONVEX and INTERLEAVED, so the optimal boundary is curved and in
  places disconnected. Suggested composition (adapt freely, keep the spirit):
    * majority class: a spiral arm or wide crescent sweeping across the plane,
      parameterized by centre, radius, angular span, pitch/twist and thickness
    * second class: a second arm offset in phase, or an annulus/ring that partially
      encloses and interleaves with the first — parameterized by radius, thickness,
      angular extent and rotation
    * rare class (the minority): MULTI-MODAL — two or three compact lobes placed inside
      the ambiguous regions between the other two, so it is genuinely hard to recover and
      makes per-class TPR/PPV/F1 interesting. Parameterized by lobe centres, spread,
      elongation.
- Classes must PARTIALLY OVERLAP so Bayes error is clearly non-zero and visibly localized.
  Per-point Gaussian jitter whose scale is itself a parameter.
- The generator must be ONE clean, documented function taking a parameter dict (or
  dataclass) plus a PRNG key, returning points and labels — so the SAME function produces
  source and target by changing parameters only.
- Print the realized class proportions after sampling.
- Calibration: if the rare class ends up with near-zero F1 even on the SOURCE, the lobes
  sit too deep in the overlap — nudge them slightly out rather than abandoning the design.

## R2 — Much larger samples and higher-resolution visualizations
- Training ~20,000 points; validation ~10,000. State split sizes explicitly in a cell.
  (2D data with a small MLP — this stays cheap.)
- Scatters: small markers, sensible alpha, categorical colours, legend by class.
- Probability and decision-region maps: raise the grid to ~300-400 per axis for smooth
  contours; keep the seismic/diverging colormap for probabilities. These are slide
  deliverables — titles, axis labels, colorbars, consistent class colours everywhere.
- Tune classifier capacity (width/depth/epochs) so contours are visibly rich and follow
  the curved supports, but the model is NOT perfect — overlap regions must still err.
  Show train/validation curves.

## R3 — A genuinely non-trivial domain shift
The target must be produced by PERTURBING THE GENERATIVE PARAMETERS, not by translating
point clouds. Make it HETEROGENEOUS across classes so degradation differs per class:
- change the spiral pitch/twist and/or angular span of one class (its shape deforms)
- change radius and thickness of another (it grows / becomes more diffuse)
- move and spread the rare class's lobes, and/or merge them differently
- leave one aspect nearly unchanged, so shift damage is visibly uneven
- include a PRIOR SHIFT: change class proportions between source and target
Keep it subtle enough that the target still reads as "the same problem in a slightly
different world". In pt-BR, name which shift types from L2B1 are present (covariate shift
and prior shift). The closing figure of that block must make the source→target change
unmistakable: side-by-side scatters plus, ideally, an overlay showing how each support
deformed.

## R4 — Visualization polish throughout
- Consistent class colours and consistent axis limits across EVERY figure, so
  source/target/fine-tuned panels are comparable by eye.
- Where three conditions are compared (source→source, source→target, finetuned→target),
  prefer aligned multi-panel figures over separate plots.
- Keep and reuse the existing confusion-matrix plot (counts, row percentages, diagonal
  TPR/PPV/F1); verify it renders well with the new data.
- Keep the existing extras (confidence/calibration under shift, unlabeled shift detection,
  error-localization maps, label-budget sweep, full vs partial fine-tuning) — re-verify
  each works with the new geometry and looks good. Do NOT add new analysis sections.

## Constraints
- Equinox + Optax; self-contained; no external repo imports; no new dependencies.
- Plots inline; SAVE NO FIGURES; write nothing to a committed path.
- Must run top-to-bottom clean in the WinterSchool kernel and on a fresh Colab.
- Runtime reasonable on CPU (target a few minutes). If long, reduce epochs or the K-sweep
  grid BEFORE reducing sample sizes or map resolution — those are the point of this round.
- pt-BR markdown throughout; keep the "Para casa" exercises.

## Definition of done
The source scatter shows visibly complex, interleaved, curved class supports; the learned
probability/decision maps show correspondingly rich contours; the source→target figure
makes a non-trivial, class-dependent deformation obvious; every figure is slide-ready.
