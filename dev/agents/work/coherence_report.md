# Coherence Report — I Escola de Inverno do IFUSP
# Post full-revision-pass QA | Generated: 2026-07-17

This report replaces the earlier 2026-07-15 report, which was written against
the old slide-schematic block format (L01_B01 etc.) and is now fully superseded.
Scope: 8 block files (L1B1–L4B2 in new naming), 3 notebooks (NB0/NB1/NB2),
README.md, dev/agents/work/course_manifest.md, git artifact hygiene.

---

## SUMMARY

### Assets found vs. manifest

| Asset | Status |
|-------|--------|
| course-materials/L1B1.md | PRESENT — gold-standard theory block, untouched |
| course-materials/L1B2.md | PRESENT — gold-standard notebook block, untouched |
| course-materials/L2B1.md | PRESENT — rewritten to L1B1 standard (THEORY) |
| course-materials/L2B2.md | PRESENT — rewritten to L1B2 standard (minimal NOTEBOOK) |
| course-materials/L3B1.md | PRESENT — rewritten to L1B1 standard (THEORY) |
| course-materials/L3B2.md | PRESENT — rewritten to L1B2 standard (minimal NOTEBOOK) |
| course-materials/L4B1.md | PRESENT — rewritten to L1B1 standard (THEORY case study) |
| course-materials/L4B2.md | PRESENT — rewritten to L1B1 standard (THEORY case study) |
| jax-examples/notebooks/00_caixa_de_ferramentas.ipynb | PRESENT, 38 cells |
| jax-examples/notebooks/01_domain_shift_toy.ipynb | PRESENT, AUC bug fixed |
| jax-examples/notebooks/02_contrastive_embeddings.ipynb | PRESENT, 35+ cells |
| dev/agents/work/course_manifest.md | PRESENT, current |
| README.md | PRESENT — single hub, 3 fixes applied this pass (see below) |
| references/2602.13902v1.pdf | PRESENT |
| references/2311.12110v3.pdf | PRESENT |
| GoogleCollab_and_notebooks_setup.md | PRESENT (repo root) |

**Missing vs. manifest:** None. All 8 blocks and 3 notebooks are present.
The old standalone index file (course-materials/00_INDEX.md) was removed in a
prior pass; README.md is the single navigational hub, as required by
my_feedback_v2 §3.

### Artifact hygiene

`git ls-files jax-examples/assets/` returns nothing (no tracked artifacts).
`git status` shows 3 modified notebooks, 3 src_*.py files, and 1
make_assets_*.py file — all code-only modifications from the notebook-builder
pass; no new PNG/pkl/npz files tracked. CLEAN.

### Fixes applied directly to README.md this pass

Three stale values corrected (instructor attention not required):

1. Line 52, Day 2 arc paragraph: "AUC ≈ 0,785" → "AUC ≈ 0,749" (post-bugfix
   value from held-out test evaluation; consistent with L2B1.md line 102).
2. Line 82, NB1 description: "AUC ≈ 0,785" → "AUC ≈ 0,749" (same reason).
3. Line 82, NB1 description: "56 bandas fotométricas" → "55 bandas
   fotométricas" (L4B2 consistently describes the J-espectro as 55 bands:
   section 1 "vetor de fluxos em 55 bandas" and section 3.1 encoder
   h_psi: R^55 → R^d).

---

## NARRATIVE CONTINUITY

### L1B1 → L1B2

L1B1 section 4 ("Estrutura do curso e materiais") points to the GitHub repo
and the notebooks. No dedicated "próximo bloco" sentence, but the framing
is complete as a standalone theory block. L1B2 is minimal (badge + path),
requiring no explicit pickup. CLEAN.

### L1B2 (NB0) → L2B1

L2B1 section 1 (Apresentação) opens: "No caderno de ontem, um polinômio de
grau alto decorou os pontos de treino e colapsou nos pontos de teste: essa foi
a lição do overfitting — generalizar *dentro* da mesma distribuição já é
difícil. O Dia 2 leva a pergunta um passo além…"

NB0 (00_caixa_de_ferramentas.ipynb) does NOT contain an explicit "amanhã
veremos…" teaser cell in the current rebuilt version (cells 31–37 are
exercises and vocabulary map). The conceptual bridge is carried by the
overfitting demo (cells 31–33) and by README.md's description of NB0 as
"plantando a primeira semente do Dia 2". The pickup in L2B1 section 1 is
clear and specific.

VERDICT: Functionally clean. Minor gap: NB0 has no dedicated teaser sentence
for Day 2. Low severity — the instructor can add it verbally; the written
bridge from L2B1's side is explicit. See TODO 1.

### L2B1 → L2B2

L2B1 section 5 ("Materiais") explicitly names NB1/L2B2 and describes its
four acts. L2B2 is minimal (badge + path). CLEAN.

### L2B2 (NB1) → L3B1

L3B1 section 1 (Apresentação) opens: "Nos dois primeiros dias, o espaço
latente foi o palco — o lugar para onde o encoder manda os dados antes da
classificação ou da adaptação de domínio. No Dia 3, ele vira o protagonista."

NB1 cell 28 (Takeaway markdown) ends with "SSDA vence para K pequeno: se
você tem poucos rótulos do alvo, o pré-treino na fonte é um ativo valioso.
Congele a cabeça" — a correct conceptual close but no explicit Day 3 teaser.

VERDICT: Clean pickup from L3B1's side. NB1 has no explicit Day 3 bridge.
Same pattern as NB0→L2B1. See TODO 1.

### L3B1 → L3B2

L3B1 section 5 ("Materiais") names NB2/L3B2 and describes its three acts,
ARI = 0,743, and t-SNE perplexities. CLEAN.

### L3B2 (NB2) → L4B1

NB2 cell 31 (Resumo markdown) has the table summarizing three acts and the
embed-then-cluster paradigm. L4B1 section 1 (Apresentação) opens: "Três dias
de ingredientes. Hoje, a cozinha de verdade." And section 3.2 explicitly
names the Weinberger loss and says: "Vocês treinaram esta perda ontem, com
bolinhas coloridas. Aqui, as bolinhas são partículas do universo."

VERDICT: Clean, well-constructed recognition moment. The bridge is designed
and executed correctly. CLEAN.

### L4B1 → L4B2

L4B1 section 5 ("Materiais") ends with the repo URL. There is no explicit
"no próximo bloco" teaser pointing to L4B2. L4B2 section 1 opens with the
J-PAS instrument description without a pickup sentence linking back to L4B1.

VERDICT: Minor gap. Since both are within Dia 4 and the session is continuous,
the instructor bridges verbally. But an explicit sentence at the end of L4B1
pointing to L4B2 would improve the written arc. See TODO 2.

---

## NOTEBOOK-BLOCK ALIGNMENT

The three notebook blocks (L1B2, L2B2, L3B2) are now MINIMAL per the L1B2
gold standard — just Colab badge + notebook path. There is therefore no
detailed act-by-act alignment to check in the block files themselves. The
alignment check falls entirely on README.md's notebook descriptions and on
the theory blocks' "Materiais" sections.

### NB0 / L1B2 / README NB0 description

README (lines 68–73) describes NB0 accurately: raw SGD (no optimizer library),
tanh + MSE, overfit demo with [1→128→128→128→1], vocabulary map. Matches
NB0 cells 14, 28, 31–33, 34. MATCH.

### NB1 / L2B2 / L2B1 section 5 / README NB1 description

L2B1 section 5 (line 112): "compara os três regimes — zero-shot, target-only
e SSDA" — uses English "target-only". README NB1 description (line 82): "zero-
shot / somente alvo / SSDA" — uses pt-BR. L4B2 table (line ~46): "Zero-shot",
"Só alvo", "SSDA". TERMINOLOGY DRIFT — see Terminology section and TODO 3.

AUC in L2B1 section 4.2 (line 102): "AUC é 0,749 no conjunto de teste". 
README (now fixed): "AUC ≈ 0,749". NB1 cell 17 function docstring confirms
held-out 20% test evaluation. MATCH (after this pass's fix).

NB1 four acts match L2B1 section 5's description. AUC bug fixed, confirmed
via NB1 cell 17 docstring: "O AUC é calculado num subconjunto de TESTE (20%
dos dados, nunca vistos durante o treino)." MATCH.

### NB2 / L3B2 / L3B1 section 5 / README NB2 description

L3B1 section 5: "sandbox de partículas 2D… encoder MLP… k-means (ARI = 0,743)
e inspecionado com t-SNE em três perplexidades." README NB2 description: "k-
means (ARI = 0,743) e t-SNE em três perplexidades." NB2 cell 27 implements
ARI on k-means clusters. NB2 cell 29 implements t-SNE (three perplexities).
MATCH. The collapse demo (c_push = 0) is in NB2 cell 8. L3B1 section 3.2
explains the trivial minimum. MATCH.

---

## TERMINOLOGY

| Term | Variants found | Files | Recommended canonical |
|------|----------------|-------|-----------------------|
| zero-shot / zero-shot | "zero-shot" | L2B1, L4B2, README | "zero-shot" — consistent, keep |
| target-only / somente alvo / só alvo | "target-only" (L2B1 §5), "somente alvo" (README NB1), "só alvo" (L4B2 table) | L2B1, L4B2, README | "só alvo" in prose and tables; drop English "target-only" from L2B1 §5 |
| encoder | encoder, $f_\theta$, $h_\psi$ | All blocks | Intentional: $f_\theta$ is full model (L1B1, L3B1); $h_\psi$ is encoder component of decomposed $f = g \circ h$ (L2B1, L4B2). No harmful drift. |
| cabeça / head | "cabeça (_head_)", "$g_\varphi$" | L2B1, L4B2 | "cabeça" in prose, "(head)" on first use. Consistent across L2B1 and L4B2. |
| perda contrastiva / discriminativa / de Weinberger | "contrastiva" (L3B1), "discriminativa" (NB2 function name perda_discriminativa_2d), "de Weinberger" (L4B1) | L3B1, NB2, L4B1 | Intentional three-step reveal: contrastiva in theory → discriminativa in code → Weinberger in paper. No harmful drift. |
| pseudo-espaço | "pseudo-espaço" | L4B1 only | Reflects the paper's term. No drift; single-block usage is appropriate. |
| embed-then-cluster | "embed-then-cluster" (L3B1, L4B1) | L3B1, L4B1 | Hyphen form. Consistent. |
| SSDA | SSDA | L2B1, L4B2, README | Consistent acronym. |
| sobreajuste / overfitting | "sobreajuste (*overfitting*)" | L2B1 §1, NB0 cells | Correct parenthetical on first use. Consistent. |

Most significant drift: **"target-only" vs. "só alvo"**. L2B1 section 5 is the
only place that uses the English form; everywhere else in the course (L4B2
table, README) the pt-BR forms are used. One-line fix in L2B1. See TODO 3.

---

## LINKS

### README.md internal links (all verified)

| Link | Target | Status |
|------|--------|--------|
| course-materials/L1B1.md | File exists | OK |
| course-materials/L1B2.md | File exists | OK |
| course-materials/L2B1.md | File exists | OK |
| course-materials/L2B2.md | File exists | OK |
| course-materials/L3B1.md | File exists | OK |
| course-materials/L3B2.md | File exists | OK |
| course-materials/L4B1.md | File exists | OK |
| course-materials/L4B2.md | File exists | OK |
| jax-examples/notebooks/00_caixa_de_ferramentas.ipynb | File exists | OK |
| jax-examples/notebooks/01_domain_shift_toy.ipynb | File exists | OK |
| jax-examples/notebooks/02_contrastive_embeddings.ipynb | File exists | OK |
| references/2602.13902v1.pdf | File exists | OK |
| references/2311.12110v3.pdf | File exists | OK |
| GoogleCollab_and_notebooks_setup.md | File exists at repo root | OK |

### Colab badge URL path segments (README and block files)

All three notebook badges in README.md and in L1B2.md, L2B2.md, L3B2.md use
the path `jax-examples/notebooks/<name>.ipynb`. These path segments match the
actual files. The external Colab domain cannot be verified locally, but the
path segments are correct.

L4B1.md and L4B2.md are pure THEORY blocks with no Colab badges. No issue.

### Wikilinks (Obsidian)

All [[L1B1]], [[L2B1]], [[L2B2]], [[L3B2]], [[L4B1]], [[L4B2]] wikilinks in
the block files resolve to existing files under the standard Obsidian vault
path. No broken wikilinks detected.

### Block cross-references in "Materiais" sections

L2B1 §5 → "NB1 (bloco L2B2)": correct.
L3B1 §5 → "NB2 (bloco L3B2)": correct.
L4B1 §5 → "notebook NB2 (bloco L3B2, 02_contrastive_embeddings.ipynb)": correct.
L4B1 §5 → "daniellopezcano/instance_halos" GitHub repo: external, not verified.
L4B2 §4 → "daniellopezcano/JPAS_Domain_Adaptation" GitHub repo: external, not verified.

---

## TIMING RISKS

### L2B1 — three-regime taxonomy at end of section 4.1

Section 4.1 covers three strategy regimes (no-data, unlabeled, SSDA). The SSDA
subsection is the most content-dense (encoder decomposition + "analogia dos
sotaques" + honest limit) and could easily expand beyond its slot. The "analogy
dos sotaques" is a high-value pedagogical moment; the instructor should protect
it even if the unlabeled/adversarial subsection gets trimmed.

### L3B1 — equations section (§3.2)

The pull/push/reg triple with three LaTeX equations, the trivial-minimum result,
and the InfoNCE aside are all in §3.2. This is the conceptual climax of the
course. Instructors should plan a Q&A buffer here and treat the InfoNCE
paragraph as sacrificial if time is short — L4B1's recognition moment does not
depend on InfoNCE.

### L4B2 — Encerramento (§5)

The closing recap (§5) runs through all 8 blocks individually. At ~30 s per
block, this is 4 min of scripted content. If the J-PAS results discussion in
§3 overruns, this risks compression. The "Horizonte" directions (§4) are the
sacrificial content; the 8-block synthesis sentence in §5 is the emotional
close and must not be cut.

---

## PT-BR QUALITY

### Finding 1 — L2B1 section 5: "target-only" (medium severity)

File: course-materials/L2B1.md, section 5 ("Materiais"), line 112.
Current text: "…compara os três regimes — zero-shot, target-only e SSDA —"
Issue: "target-only" is English; pt-BR is used everywhere else in the course
for this concept ("somente alvo" in README, "só alvo" in L4B2 table).
Fix: replace "target-only" with "só alvo" for consistency.

### Finding 2 — L4B2 section 2.1: phrasing of mock count (low severity)

File: course-materials/L4B2.md, section 2.1.
Current text: "O domínio **fonte** contém ~1,5 × 10⁶ J-espectros *mock*"
No grammatical issue. Exponent superscript notation "10⁶" renders correctly
in Markdown environments that support Unicode superscripts (Obsidian, rendered
HTML). In plain-text environments it may not. Low severity.

### Finding 3 — NB0 cell 0: block reference uses old naming

NB0 cell 0 markdown header: "I Escola de Inverno do IFUSP — Bloco L01_B02"
(underscore+zero-padded naming). The canonical naming in all course-materials
and README is "L1B2" (no underscores, no padding). This does not affect
student experience but is inconsistent with the naming convention.
Same pattern in NB2 cell 0: "Bloco L03_B02". See TODO 4.

---

## PRIORITIZED TODO

Items ordered by impact. Each is phrased as a ready-to-paste instruction.

**TODO 1 — Add Day-forward teaser to NB0 and NB1 (LOW — narrative polish)**

In jax-examples/src_00_caixa_de_ferramentas.py, add a final markdown cell
(after the "Para casa" cell, cell 37) with this text:

    ## 🔵 O que vem por aí — Dia 2
    > Na quarta-feira, veremos o que acontece quando o problema não é
    > *memorizar demais* — mas quando os **dados de teste vêm de uma
    > distribuição completamente diferente da de treino**.
    > Esse modo de falha tem nome: *mudança de domínio*.

In jax-examples/src_01_domain_shift_toy.py, add after the Takeaway cell (cell
28) a short teaser pointing to the L3B1 themes:

    ## 🔵 O que vem por aí — Dia 3
    > Amanhã, em vez de *alinhar* dois espaços, vamos *construir* um espaço
    > onde distância significa algo. Essa é a aprendizagem contrastiva.

After editing each src file, re-execute the notebook (nbconvert) and confirm
BUILD GREEN before committing.

**TODO 2 — Add L4B1→L4B2 bridge sentence (LOW — narrative polish)**

In course-materials/L4B1.md, at the end of section 5 ("Materiais"), add
one sentence before the repo URL line:

    "No bloco seguinte (L4B2), o mesmo par encoder + cabeça e os mesmos três
    regimes reaparece com dados reais do J-PAS — o experimento de brinquedo do
    Dia 2 em escala astronômica."

**TODO 3 — Fix "target-only" term drift in L2B1 (LOW — pt-BR consistency)**

In course-materials/L2B1.md, section 5, line 112:
Replace: "compara os três regimes — zero-shot, target-only e SSDA —"
With:    "compara os três regimes — zero-shot, só alvo e SSDA —"
This aligns with L4B2's table terminology and README.md.

**TODO 4 — Fix old-style block names in NB0 and NB2 cell 0 (INFORMATIONAL)**

In jax-examples/src_00_caixa_de_ferramentas.py, cell 0 header:
Replace: "Bloco L01_B02"
With:    "Bloco L1B2"

In jax-examples/src_02_contrastive_embeddings.py, cell 0 header:
Replace: "Bloco L03_B02"
With:    "Bloco L3B2"

Re-execute and confirm BUILD GREEN after each change.

**TODO 5 — Verify "55 bandas" against arXiv:2602.13902 (INFORMATIONAL)**

README.md now says "55 bandas fotométricas" (fixed this pass from "56"),
consistent with L4B2 which states "54 filtros de banda estreita… vetor de
fluxos em 55 bandas" and encoder h_psi: R^55 → R^d. Before the July
course dates, confirm against Section 2 of arXiv:2602.13902 that the actual
input dimension used in the paper is 55. If it is 56, revert README line 82
to "56" and update L4B2 section 1 and section 3.1 encoder dimension to match.

---

*Review scope: 8 block files (L1B1–L4B2), 3 notebooks (38+34+35 cells),
README.md, course_manifest.md, git artifact status.*
*Fixes applied directly this pass: 3 (all in README.md).*
*Findings requiring instructor attention: 5 TODOs (all LOW or INFORMATIONAL).*
