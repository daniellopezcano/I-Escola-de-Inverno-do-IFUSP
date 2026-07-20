# Redesign brief — L2B2 notebook (01_domain_shift_toy), round 4 — IN-PLACE EDIT

## SCOPE LOCK
Modify ONLY, in place (no _v2 copies):
  - jax-examples/notebooks/01_domain_shift_toy.ipynb
  - jax-examples/src_01_domain_shift_toy.py
Write nothing else in the repository. Read other files for style reference only.

## Working procedure
1. Resync first (the .ipynb is authoritative and may contain hand edits):
   $ENV/jupytext --to py:percent jax-examples/notebooks/01_domain_shift_toy.ipynb \
     -o jax-examples/src_01_domain_shift_toy.py
2. Apply changes to the .py; regenerate the .ipynb; execute headless with the
   WinterSchool kernel; iterate until green; leave both files consistent.

## Why this round exists
The current example is a clear improvement but still does not make the two central
lessons land hard enough:
  (a) HOW domain shift degrades a model, in a controlled and quantifiable way;
  (b) WHY fine-tuning a pretrained model is better than training from scratch on the
      target — i.e. that pretraining buys LABEL EFFICIENCY.
This round adopts the canonical domain-adaptation toy benchmark and adds the experiments
that make (b) undeniable. Keep the seven-block structure, pt-BR narrative, Equinox/Optax
style, inline helpers, and ALL existing evaluation machinery (confusion matrices with
TPR/PPV/F1, probability maps, decision regions, ROC/AUC, calibration, error maps).

## R1 — Adopt the rotating-moons benchmark, generalized to three classes
Replace the current generator with the canonical INTER-TWINNING MOONS design, which is
the standard toy in the domain-adaptation literature (Courty et al. 2016; Germain et al.
PAC-Bayes DA; and many follow-ups). Its virtues: strongly non-linearly separable, curved
and interleaved supports, and — crucially — a domain shift controlled by a SINGLE
INTERPRETABLE PARAMETER (the rotation angle).

Design:
- THREE interleaved arc/moon-shaped classes (a three-class generalization of the classic
  two-moon layout): three crescent arcs, mutually offset in phase and radius so they
  interleave and partially overlap. Non-convex supports, curved optimal boundaries.
- Keep the class IMBALANCE at the current proportions. The minority class should be the
  arc most entangled with the others, so per-class metrics stay interesting.
- Gaussian jitter on each point (noise scale as a parameter) producing genuine overlap and
  non-zero Bayes error.
- ONE clean generator function taking a parameter dict (centres, radii, angular spans,
  thickness, noise, class proportions, ROTATION ANGLE theta) plus a PRNG key.
  Source = theta 0; target = theta > 0. Same function, different parameters.
- Sample sizes: ~20,000 train, ~10,000 validation (2D + small MLP stays cheap).
  Print the split sizes and realized class proportions.

## R2 — Make non-linearity an explicit teaching beat (early in the notebook)
Before the main model, add a SHORT demonstration that a LINEAR classifier fails on this
data even in-domain. The literature makes this point explicitly: with interleaving moons a
linear model cannot separate the classes, so it would also perform poorly on the target —
you must first choose a model that works on the source. Two or three cells, one figure
(linear decision regions overlaid on the data), one line of pt-BR narrative. Then proceed
with the MLP.

## R3 — The shift dial: degradation as a function of rotation angle
This replaces the single before/after shift with a controlled study.
- Show the target at a few angles (e.g. 15, 30, 45, 60 degrees) as a small multi-panel
  figure, so students see the shift growing.
- SWEEP the angle and plot a DEGRADATION CURVE: macro-F1 (and/or accuracy) of the
  SOURCE-trained model evaluated on targets at each angle, with the source-validation
  level drawn as a horizontal reference. The literature reports monotone degradation with
  angle for unadapted models — reproduce that shape.
- Choose ONE "working angle" (a moderate value where degradation is clear but the problem
  is still recognizably the same, e.g. ~35-45 degrees) and use it for all subsequent
  blocks. State the choice and why.
- Add mild CLASS-DEPENDENT secondary perturbations at the working angle (e.g. one arc
  slightly thicker/more diffuse, one barely changed) plus a PRIOR SHIFT (class proportions
  differ between source and target), so the shift is not a purely rigid rotation and
  degradation is visibly uneven across classes. Name the shift types present against
  L2B1's taxonomy (covariate shift and prior shift), in pt-BR.

## R4 — The core experiment: fine-tuning buys LABEL EFFICIENCY
This is the block the whole notebook exists for. Make it unmistakable.
- Regime comparison as a function of the number of LABELED TARGET EXAMPLES K
  (e.g. K in {10, 25, 50, 100, 250, 500, 1000, 2500}):
    (A) ZERO-SHOT: source-trained model applied to target, no target labels (a horizontal
        reference line — it does not depend on K)
    (B) FROM SCRATCH: a fresh model trained only on the K target labels
    (C) FINE-TUNED: the pretrained source model, fine-tuned on the same K target labels
- THE KEY FIGURE: metric vs K for the three regimes on the same axes (log x-axis).
  The expected and pedagogically decisive pattern: C dominates B strongly at small K,
  and B catches up only when K becomes large. Narrate this in pt-BR as THE argument for
  transfer learning: pretraining does not just save time, it saves LABELS — and labels are
  the expensive resource (tie back to L2B1).
- Use the SAME architecture, initialization scheme and training budget for B and C so the
  comparison is fair; state this explicitly.
- Show the qualitative counterpart at one small K: decision-region maps for A, B and C
  side by side over the target data. B (from scratch, few labels) should look
  jagged/overfit; C should look like a sensibly re-oriented version of the source model's
  boundaries. This visual is the "pretraining transfers structure" argument.

## R5 — What to fine-tune, and what it costs
Two short additions that make the treatment honest and complete:
- WHAT TO ADAPT: compare fine-tuning ALL parameters vs only the LAST layer(s), at the same
  K. Frame as a design choice (keep general structure, re-fit the decision surface), tying
  to L2B1. One extra curve or a compact table.
- CATASTROPHIC FORGETTING: after fine-tuning on the target, re-evaluate the model back on
  the SOURCE validation set and report the drop. Most toy demos omit this; it teaches that
  adaptation has a cost and motivates why one might keep both domains in play. Keep it to
  a couple of cells plus one small figure or table row.
- Optionally (only if cheap): show that the fine-tuning LEARNING RATE matters — too large
  and the pretrained advantage is destroyed (the model effectively restarts). One extra
  line on the K-curve, marked optional.

## R6 — Visualization polish
- Consistent class colours and axis limits across EVERY figure so all panels are directly
  comparable by eye.
- Probability/decision maps on a grid of ~300-400 per axis for smooth contours; keep the
  seismic/diverging colormap for probabilities; titles, axis labels, colorbars, legends.
- Prefer aligned multi-panel figures for any three-way comparison.
- Every figure must be slide-ready.

## Constraints
- Equinox + Optax; self-contained; no external repo imports; no new dependencies.
- Plots inline; SAVE NO FIGURES; write nothing to a committed path.
- Runs clean top-to-bottom in the WinterSchool kernel and on a fresh Colab.
- Runtime reasonable on CPU. The angle sweep (R3) and K sweep (R4) each retrain several
  small models — keep those models small and the epoch counts modest. If runtime is a
  problem, thin the K grid and the angle grid BEFORE reducing sample sizes or map
  resolution, and drop the optional learning-rate variant first.
- pt-BR markdown throughout; keep the "Para casa" exercises.

## Definition of done
- The source scatter shows three interleaved, curved, overlapping class supports.
- The degradation-vs-angle curve shows clear monotone degradation.
- The K-sweep figure shows fine-tuning (C) clearly above from-scratch (B) at small K, with
  the gap closing at large K.
- The side-by-side decision maps at small K visibly support that story.
- Report in the completion summary: the working angle chosen, the zero-shot vs fine-tuned
  metrics at the smallest and largest K, the forgetting drop on source, and total runtime.
