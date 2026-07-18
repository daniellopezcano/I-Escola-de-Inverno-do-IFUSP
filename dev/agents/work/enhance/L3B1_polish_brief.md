# Polish brief — L3B1, round 2 (focus on InfoNCE and t-SNE)

## Context
The current course-materials/L3B1.md is good but unfocused. This round SHARPENS it:
concentrate on the pipelines actually used today (InfoNCE-style contrastive learning) and
on t-SNE, and cut what does not serve that backbone. Keep the file's pt-BR narrative,
compactness, L1B1 density and the 40-minute budget. This is a polish, not a rewrite:
preserve everything not listed below.

First READ the current course-materials/L3B1.md and locate the sections by their number
AND content (numbering below refers to the current file). After the edits, RENUMBER all
sections/subsections cleanly and fix any internal cross-references.

## CUTS

### C1 — Remove subsection 2.3
Delete it entirely. It does not contribute to the conceptual backbone of this lecture.
Fold anything genuinely load-bearing into neighbouring subsections in one sentence, or
drop it.

### C2 — Remove subsection 4.2 (the Weinberger-style / margin-based pull-push loss)
Delete the margin/pull-push loss treatment. The block should present ONE loss in depth:
InfoNCE. (The pull/push interaction-potential intuition may survive only as at most one
short sentence if it helps motivate attraction/repulsion in general — do not develop it,
do not write its equations.)

### C3 — Remove the UMAP part
Keep the t-SNE treatment; delete the UMAP subsection/comparison. If a single sentence is
useful to acknowledge that other projection methods exist (UMAP among them), that is the
maximum — no mechanism, no comparison table.

## EXPANSIONS

### E1 — InfoNCE, explained in real depth (the block's mathematical centrepiece)
This becomes the single loss the students truly understand. Cover, compactly but
thoroughly, with every symbol defined in ≤1 line:
- the setup: an anchor, its positive, and a set of negatives; what a "view" is
- the similarity function used in practice (cosine similarity on normalized embeddings)
  and why normalization matters
- the loss itself, written explicitly, and read as a CLASSIFICATION problem: identify the
  positive among the negatives — i.e. a softmax over similarities (cross-entropy in
  disguise)
- the TEMPERATURE parameter: what it does geometrically, the statistical-mechanics
  reading (Boltzmann weights over negatives), and the practical effect of low vs high
  temperature (focus on hard negatives vs. spreading attention)
- the role of the NUMBER of negatives / batch size, and why this drove designs such as
  memory banks and momentum encoders
- the collapse problem and how InfoNCE's negatives exclude the trivial solution
- brief practical notes students would actually need: the projection head, why the
  embedding used downstream is usually taken before the head, and typical failure modes
Keep it accessible: logical skeleton and interpretation over algebraic derivation.

### E2 — New subsection 3.4: types of contrastive learning (conceptual taxonomy)
Add a subsection at the end of section 3 mapping the CONCEPTUAL varieties, so students
leave knowing what exists and when each applies. Organize by what defines the positives
and the training signal, e.g.:
- self-supervised instance discrimination (augmented views of the same sample)
- supervised contrastive (labels define positives)
- clustering-based approaches (pseudo-labels/prototypes as the signal)
- negative-free / distillation-style approaches (asymmetry instead of negatives)
- multimodal / cross-modal contrastive (paired data from two modalities, e.g. CLIP-style)
For each: one or two lines on the core idea and on WHEN it is the right choice
(what data situation it suits). This is a conceptual map, not a methods survey.

### E3 — Sharpen 4.5 (other losses: pros and cons)
Keep this subsection and make it genuinely useful as a decision guide. For each notable
alternative loss/method, give: the core idea in one line, its main advantage, its main
drawback, and the situation where it is preferable to InfoNCE. Keep entries terse — the
value is orientation and keywords, not detail. Ensure it does NOT overlap with the new
3.4: 3.4 is a CONCEPTUAL taxonomy (what kind of contrastive learning, defined by the
training signal); 4.5 is about the LOSSES themselves and their practical trade-offs.
Make the division of labour explicit so the two read as complementary.

## Constraints
- pt-BR, narrative, compact, self-sufficient (slides are built FROM this file).
- Respect the 40-minute budget: the cuts should roughly pay for the expansions. Mark
  anything optional/cuttable as such.
- Every equation in LaTeX with all symbols defined; original-paper references inline
  (arXiv/DOI); keep and update the closing "Referências" section.
- Keep the block's continuity with L1B1 and L2B1, and keep it setting up L3B2 without
  duplicating the notebook.
- Renumber sections cleanly after the cuts and fix all internal references.

## Output
course-materials/L3B1_v3.md — leave the current L3B1.md untouched.
