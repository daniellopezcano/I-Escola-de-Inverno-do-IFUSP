# Brief — L4B2 «Simulações, ML e adaptação de domínio: dois estudos de caso»

## SCOPE LOCK
Modify ONLY course-materials/L4B2.md, in place. Create/modify/delete nothing else.

## Role of this block
Lecture 4, block 2 — the course finale. It takes the cosmological context built in L4B1
and connects it to everything learned in L1–L3: representations, contrastive learning
(L3) and domain shift/adaptation (L2). Two of its four sections are the instructor's own
research; treat those BROADLY and conceptually — the instructor will supply detail live.
Students should feel RECOGNITION, not a firehose of new material.

## Inputs
- slides/L4B2.pdf — the instructor's DRAFT deck. Follow its section structure and figures.
- references/2311.12110v3.pdf — the instance-segmentation / halo-detection paper (§2).
- references/2602.13902v1.pdf — the J-PAS sim-to-obs SSDA paper (§4).
  Extract from both only what §2 and §4 below require: the problem, the method, the
  headline results. No deep technical detail, no hyperparameters, no appendices.
- course-materials/L4B1.md (context just delivered), L2B1.md and L3B1.md (the concepts to
  call back to), L1B1.md (style exemplar).
- dev/agents/work/my_feedback_v2.md — repo standard.

## Structure — four sections (matching the deck)

### 1. Limitações das simulações e o papel do ML
Open from L4B1's closing trade-off. Cover:
- The cost problem: the enormous dynamic range of cosmological simulations forces a
  sacrifice of resolution, of volume, or both; and Stage-IV surveys (DESI, Euclid,
  Rubin/LSST) demand large ensembles of simulations for forward models, covariances and
  training data — which multiplies the cost.
- Then the ML remedies, in four compact families (one short paragraph each):
  (a) EMULATORS OF SUMMARY STATISTICS — the established line beginning with frameworks
      like CosmicEmu for the matter power spectrum, later extended to smaller scales,
      wider redshift and larger parameter spaces, plus emulators for the halo mass
      function, galaxy power spectra and the concentration–mass relation.
  (b) SUPER-RESOLUTION — learn the high-resolution field conditioned on a cheap
      low-resolution one. Concrete numbers worth quoting: AI-assisted SR generating ~512×
      more particles by predicting their displacements from initial positions, trained on
      as few as ~16 LR–HR simulation pairs, with stochastic generation that samples
      small-scale modes conditioned on the large-scale environment; GAN- and U-Net-based
      approaches reaching percent-level power spectra and ~10% halo mass functions down to
      ~10^11 M_sun.
  (c) FAST FIELD EMULATION — e.g. dimensionality-reduction + supervised ML emulators
      reporting ~3 orders of magnitude speed-up over a full N-body run while reproducing
      the power spectrum at the ~1% level.
  (d) ML CORRECTIONS INSIDE CHEAP SIMULATIONS — rather than replacing N-body, add learned
      corrections for unresolved physics, e.g. an extra effective force in particle-mesh
      simulations to capture unresolved inter-particle forces.
  Optionally in one line each: painting baryons onto N-body with generative models; fast
  Sunyaev–Zel'dovich map generation; emulator-based calibration of sub-grid models.
- Close with the honest caveat, in the L1B1 spirit: these models are trained on
  simulations, so they inherit their assumptions and do not extrapolate reliably outside
  the regime they were trained on — which motivates §3 and §4.
- Cite original papers inline (arXiv/DOI) for the specific results quoted.

### 2. Detectando halos com perda de CL
The instructor's instance-segmentation paper (references/2311.12110v3.pdf), presented
BRIEFLY and conceptually. Follow the deck:
- The problem: given the linear/initial density field, predict which particles will end up
  in which dark-matter halo — normally answered only by running the full N-body.
- The framing as PANOPTIC SEGMENTATION, split into two sub-problems (use the deck's
  contrast explicitly):
  * SEMANTIC ("is it a halo?"): a fixed number of classes {0,1}, each label is a class —
    a standard classification problem, trained with cross-entropy.
  * INSTANCE ("which halo?"): an arbitrary number of instances {0,…,N}, labels are
    permutation-invariant — NOT a classification problem, hence no direct differentiable
    loss.
- The architecture in one paragraph: a U-Net-style 3D convolutional network taking the
  linear density field and producing both outputs (the deck shows the down/up-sampling
  blocks and skip connections). Keep it at the level of the figure.
- THE RECOGNITION MOMENT — the pedagogical heart of this section: the instance branch is
  trained with the WEINBERGER loss, a contrastive/metric loss that embeds each particle in
  a learned "pseudo-space" where haloes become clusters, with a PULL term attracting
  particles to their instance centre and a PUSH term repelling different centres, plus a
  regularization term. Connect this EXPLICITLY to L3B1's contrastive principle (define
  which pairs should be close and which far; the network geometrizes that definition) —
  the students have already met this idea; here it is doing physics. Then note the
  properties the deck lists: differentiable (so backpropagation works), handles an
  arbitrary number of instances, deals with permutation-invariant labels, regions need not
  be connected, requires a clustering post-processing step (as do competing techniques),
  and needs no extra parameters for that clustering.
- Results, briefly and visually: predicted halo membership vs. ground truth in Lagrangian
  space, the probability and error maps, and predicted vs. true halo masses. Mention that
  the model reproduces the population statistics, without dwelling on numbers.

### 3. O espaço entre as diferentes observações e com simulações
NOTE: still under development in the deck. Write a SHORT, clear section the instructor can
expand, covering:
- Why two datasets of "the same sky" are not the same data: different telescopes,
  detectors, filters/bandpasses, depth, point-spread function, calibration, selection
  functions and noise properties. Each survey observes the Universe through its own
  instrument, and that instrument leaves its fingerprint on the data.
- Why this makes COMBINING datasets hard, and why a model trained on one survey degrades
  on another — call back explicitly to L2's domain shift taxonomy (covariate shift, prior
  shift) using the vocabulary already taught.
- MODEL MISSPECIFICATION with simulations: we constrain cosmology by comparing data to
  simulations, but simulations do not reproduce observations exactly (sub-grid physics,
  missing baryonic effects, finite resolution). When the forward model is wrong, the
  inference is BIASED — the posterior can be tight and wrong. Tie this to L4B1 §3: SBI
  makes us maximally dependent on the simulator being right.
- Frame the section as the problem statement that §4 answers.

### 4. SSDA aplicado a observações
The instructor's J-PAS paper (references/2602.13902v1.pdf), again BROAD and conceptual.
Follow the deck:
- The setting: J-PAS observes with 54+ narrow-band filters, producing a low-resolution
  "J-spectrum" for every object; the task is classifying objects into GALAXY, low-z QSO,
  high-z QSO and STAR, which feeds spectroscopic follow-up target selection.
- The data construction, and why it is a textbook domain-shift problem: the SOURCE domain
  is MOCKS built by convolving DESI spectra with the J-PAS narrow-band response (labels
  are abundant and free); the TARGET domain is REAL J-PAS observations, whose labels come
  only from the small DESI cross-matched sample (scarce and expensive). Reuse the deck's
  Source/Target framing.
- The three models compared (the deck's central experimental design), stated so students
  recognize L2B2's three regimes:
  * SUPERVISED model — trained only on real J-PAS observations (the scarce-label baseline),
  * no-DA model — trained on mocks and applied directly to real observations (the
    zero-shot / domain-shift case),
  * SSDA model — the no-DA model RETRAINED/fine-tuned on real J-PAS observations with part
    of the network's weights kept fixed (the deck shows the fixed-weights arrow).
- The results, at headline level only: the per-class confusion matrices (with TPR/PPV/F1)
  for the three models, and the radar plot / per-class ROC-AUC comparison. Report the
  qualitative message and only a few illustrative numbers, e.g. how the rare high-z quasar
  class degrades under the no-DA model and recovers with SSDA. Do not tabulate everything.
- The honest residual: the z ≈ 2.1 boundary between low-z and high-z quasars is an
  arbitrary cut on a continuous variable, so objects near it are intrinsically ambiguous
  (the deck's probability-vs-redshift zoom). Distinguish this "the physics is hard" error
  from "the training set was biased" error — the decomposition lesson from L2.
- Close the COURSE: one short paragraph tying the four days together (representations →
  domain shift and adaptation → contrastive learning → these two applications), and
  pointing students to the public repositories and to the possibility of working on these
  topics.

## Style
- pt-BR, narrative, COMPACT, matching L1B1's density; clear section headers.
- For §2 and §4, prioritize CONCEPTUAL clarity and the callbacks to L2/L3 over technical
  completeness — the instructor narrates the detail live.
- Equations only where they earn their place; define every symbol in ≤1 line. The
  Weinberger pull/push terms may be written if compact.
- Inline references to original papers (arXiv/DOI) including the two in references/.
- Frontmatter consistent with the other blocks, `slides:` pointing to the L4B2 Google
  Slides deck, noting slides/L4B2.pdf as the current draft export.
