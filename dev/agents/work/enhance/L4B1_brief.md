# Brief — L4B1 «Cosmologia: contexto físico e o estado da arte»

## SCOPE LOCK
Modify ONLY course-materials/L4B1.md, in place. Create/modify/delete nothing else.

## Role of this block
Lecture 4, block 1. Unlike L1–L3 (which are ML-centric), THIS BLOCK IS PHYSICS-FIRST:
it gives students the cosmological context and state of the art, so that L4B2 can then
connect that context to machine learning. Keep the L1B1 house style — pt-BR, narrative,
compact, self-sufficient, clear section headers the slides map onto — but the content is
cosmology, not ML. Do not teach ML methods here; L4B2 owns that.

## Inputs
- slides/L4B1.pdf — the instructor's DRAFT deck. Follow its section structure and the
  figures it already contains; the markdown is the narrated companion.
- dev/agents/work/enhance/cosmological_probes_comparison.md — the instructor's summary of
  cosmological probes; it is the authoritative content source for section 2.
- course-materials/L1B1.md — style/density exemplar.
- dev/agents/work/my_feedback_v2.md — repo standard.

## Structure — four sections (matching the deck)

### 1. Introdução à Cosmologia
Narrate the arc from General Relativity to today's ΛCDM picture, briefly and without
heavy formalism:
- Einstein's formulation of GR and the field equations (show them; define the symbols in
  one line each, do not derive anything).
- The FLRW model built on homogeneity and isotropy (the cosmological principle); name the
  people behind it (Friedmann, Lemaître, Robertson, Walker) as the deck does; expansion
  and redshift as the observational consequence.
- The second half of the 20th century: the evidence for DARK MATTER (galaxy rotation
  curves — Vera Rubin) and for DARK ENERGY (the supernova Hubble diagram and accelerated
  expansion).
- Close with the modern picture as in the deck's ESA "Universe across space and time"
  figure: Big Bang → recombination/CMB at ~380,000 yr → dark ages → first stars and
  reionisation → structure formation → accelerated expansion today (~13.7 Gyr).
- State the goal that frames the rest of the lecture (as the deck does): characterize the
  Universe's properties, understand its composition, determine the nature of gravity.

### 2. Levantamentos e observações
An overview of the main observational probes used to constrain cosmology. Use
cosmological_probes_comparison.md as the content source. For EACH probe give a compact
paragraph covering: what we look at, what it traces, when in cosmic history (redshift
range), what it tells us, and its main difficulty/limitation. Probes to cover, following
the deck: galaxy clustering (biased tracer of the dark-matter web; BAO as a standard
ruler; galaxy bias as the difficulty), weak gravitational lensing (all intervening mass,
no assumptions; a ~1% distortion hidden in galaxy noise), the CMB (sound waves frozen at
recombination; the anchor calibrating other probes; a single sky, hence cosmic variance),
the Lyman-α forest (quasar sightlines as core samples through the intergalactic medium;
the smallest scales and deepest reach; gas-temperature degeneracy), and gravitational
waves (ripples in spacetime; absolute distance without a calibration ladder; no redshift).
If the probes file also covers CMB lensing and 21 cm intensity mapping (as the deck's
summary table does), include them in the same compact form.
Close the section with the complementarity message: the probes cover different epochs and
scales and have different systematics, so they are combined rather than ranked.

### 3. Analisar os dados: modelos e inferência
The inference logic, kept conceptual:
- The forward picture from the deck: parameters θ → model → predicted data/observations D;
  inference is the arrow back from D to θ.
- Bayesian inference: state Bayes' theorem, name posterior, likelihood, prior and
  evidence, each in one line. The goal is to characterize the posterior.
- MCMC: present Metropolis–Hastings as the deck does — propose a candidate, compute the
  acceptance ratio, accept or reject — building a chain whose empirical distribution
  converges to the posterior. Give the acceptance ratio for a symmetric proposal.
- IMPLICIT/SIMULATION-BASED models: the key twist the deck highlights — for many
  cosmological problems there is NO tractable likelihood formula; we have a prior and a
  SIMULATOR. Explain why this breaks the standard MCMC acceptance ratio (it needs the
  likelihood), and introduce simulation-based inference (SBI, also called
  likelihood-free inference) as the family of methods that learn the
  likelihood/posterior/ratio from simulated data instead. Keep it to the idea and the
  motivation — this is the bridge that makes simulations indispensable, leading into §4.

### 4. A necessidade de simulações
NOTE: this section is still under development in the deck. Write a SHORT, clear
placeholder-quality section (a few compact paragraphs) that the instructor can expand:
- Why simulations are unavoidable: linear/perturbative theory (e.g. CLASS/CAMB) describes
  the early universe and large scales well, but structure formation becomes non-linear at
  late times and small scales, exactly where the data is most constraining. As the deck's
  diagram shows: GR perturbation theory from z~1100, Newtonian perturbation theory back to
  the initial conditions at z_ICs~50, then non-linear N-body evolution to z=0.
- COLLISIONLESS DARK-MATTER (N-body) SIMULATIONS: how they work in essence (sample the
  initial density field with particles, evolve them under gravity alone), what they are
  for (the non-linear matter distribution, halo catalogues, mock catalogues, covariance
  matrices, training data for emulators), and their limits (no baryonic physics; finite
  resolution and box size).
- HYDRODYNAMIC SIMULATIONS: adding gas, cooling, star formation and feedback; what they
  buy (realistic galaxies and observables) and what they cost (far more expensive; results
  depend on sub-grid prescriptions that are calibrated, not derived).
- The trade-off table in words: volume vs. resolution vs. physical completeness — you
  cannot have all three, which sets up L4B2's discussion of what ML can do about it.
- Flag clearly (in the instructor-facing sense, without breaking student tone) that this
  section is to be expanded.

## Style
- pt-BR, narrative, COMPACT, matching L1B1's density; clear section headers.
- Equations only where they earn their place (Einstein field equations, Bayes, MH
  acceptance ratio); define every symbol in ≤1 line.
- Keep the deck's image credits/links where they carry information (ESA, Argonne, etc.).
- Inline references to original sources where natural; closing "Referências" section.
- Frontmatter consistent with the other blocks, with a `slides:` field pointing to the
  L4B1 Google Slides deck and a note that slides/L4B1.pdf is the current draft export.
- Do NOT teach ML here; at most one closing sentence pointing forward to L4B2.
