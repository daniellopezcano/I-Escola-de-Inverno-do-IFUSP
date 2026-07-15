# Coherence Report — I Escola de Inverno do IFUSP
# Pass 4 QA Review | Generated: 2026-07-15 | Reviewer: course-reviewer agent
---
## SUMMARY
### Assets found

| Asset type | Expected | Found | Status |
|---|---|---|---|
| Block files (L0X_B0Y.md) | 8 | 8 | OK |
| Notebooks (.ipynb) | 3 | 3 | OK |
| Index (00_INDEX.md) | 1 | 1 | Written this pass |
| Course manifest | 1 | 1 | OK |
| Static PNG fallbacks (NB0) | 3 | 3 | OK (`nb0_fig_trophy.png`, `nb0_fig_overfit.png`, `nb0_fig_senoide_intro.png`) |
| Static PNG fallbacks (NB1) | 7 | 7 | OK |
| Static PNG fallbacks (NB2) | 4 | 4 | OK (`nb2_fig_sandbox_final.png`, `nb2_fig_evolution.png`, `nb2_fig_tsne.png`, `nb2_fig_umap.png`) |
| PKL checkpoints | 15 | 15 | OK |
| NPZ data files | 6 | 6 | OK |
### Missing vs manifest
1. **Manifest-specified PKL names not on disk**: `nb0_epoch10_params.pkl` and `nb0_epoch100_params.pkl` (manifest §NB0 brief). The actual assets are `nb0_epoch200_params.pkl` and `nb0_epoch500_params.pkl`. The notebook and block L01_B02 correctly use the as-built names; the manifest asset-name list is stale. No functional impact.
2. **Optional assets not produced**: `galaxy10_1k.npz`, `nb2_sandbox_animation.gif`. Both were explicitly optional per manifest decisions D1 and NB2 brief. No action needed.
3. **Three static PNGs referenced in L03_B02 block but absent from `jax-examples/assets/`**:
   - `nb2_fig_sandbox_initial.png` — the initial particle scatter (NB2 cell 4 renders live)
   - `nb2_sandbox_collapsed.png` — collapse demo (NB2 cell 8 loads from `nb2_sandbox_collapsed.npz` and renders live; the `.npz` exists but no PNG fallback)
   - `nb2_fig_kmeans.png` — k-means comparison scatter (NB2 cell 27 renders live; no PNG fallback)
   These three wikilinks in L03_B02's "Visualização e Slides" section will appear as broken images in Obsidian.

---
## NARRATIVE CONTINUITY
All eight block transitions are coherent and internally consistent. The recap/teaser chain holds end-to-end.

| Transition | Teaser in block N | Recap in block N+1 | Match? |
|---|---|---|---|
| L01_B01 → L01_B02 | "depois do intervalo: as ferramentas concretas" (line 33–36 timing) | "o laboratório computacional como extensão do laboratório físico" | Yes |
| L01_B02 → L02_B01 | overfitting cell: "Na quarta-feira, veremos o que acontece quando a distribuição de TESTE é diferente da de TREINO" | "Yesterday we saw that a network with too many parameters memorizes noise — it fails within the same distribution. Today: what if the test distribution itself is different?" | Yes |
| L02_B01 → L02_B02 | "no próximo bloco vamos quebrar um classificador de propósito" | "um laboratório de patologia: causar a doença, diagnosticar, tratar" | Yes |
| L02_B02 → L03_B01 | "t-SNE que aparece na célula 🟣 deste notebook é o gancho para essa discussão" | "Day 2's latent scatters — the space was the stage; today it is the protagonist" | Yes |
| L03_B01 → L03_B02 | "no próximo bloco, vamos VER um espaço se organizar em tempo real" | "primeiro dinâmica molecular; depois deep learning; no fim vocês percebem que era a mesma coisa" | Yes |
| L03_B02 → L04_B01 | "amanhã: esta máquina exata prevendo onde nascem os halos de matéria escura" (cell 31 and Ato 2 gancho) | "Três dias de ingredientes; hoje, a cozinha de verdade. Regra do dia: vocês vão RECONHECER, não aprender do zero." | Yes |
| L04_B01 → L04_B02 | "no próximo bloco, vemos o mesmo princípio — encoder + cabeça, três regimes — aplicado a quasares reais no J-PAS" | "O experimento de três regimes que você executou na quarta-feira em 2D, com nuvens gaussianas deslocadas, agora aparece em escala real" | Yes |
**No narrative continuity mismatches found.**

---
## NOTEBOOK ↔ BLOCK ALIGNMENT
### NB0 / L01_B02
All 6 notebook parts match the block description. Cell numbering in the block matches the notebook (trophy at cell 30, overfit at cell 33, vocab map at cell 34, etc.). Polling question in cell 31 matches the 🟡 description in the block. Learning objectives, timing table, and takeaway are consistent.
**ONE DIVERGENCE — trophy epoch checkpoints:**
- L01_B02 (block for NB0) correctly states "épocas 0, 200, 500 e 1000" in the learning objectives and timing plan. Consistent with notebook cell 30: `EPOCAS_TROPHY = [0, 200, 500, 1000]`. ✓
- L01_B01 (day-1 theory block, "Demonstração Prática" section, line ~161): states "acompanhe como os parâmetros θ evoluem ao longo das épocas **(0, 10, 100, 1000)**". This does not match the notebook. The correct values are 0, 200, 500, 1000.
- The manifest NB0 brief (cell 16) also lists "epochs 0, 10, 100, 1000" — this is stale in the manifest.
**Numeric verification (NB0 actual outputs vs. block claims):**

| Claim | Source | Actual notebook output | Match? |
|---|---|---|---|
| Initial loss ≈ 0.28 ≈ Var(y) | L01_B02 | 0.2801 (vs Var=0.2881) | Yes |
| Final loss ≈ 0.035 | L01_B02 | 0.034995 | Yes |
| Overfit loss < σ²=0.0225 | L01_B02 | 0.013914 | Yes |
| CPU timing ~27 ms | L01_B02 | 26.7 ms | Yes |
| GPU < 5 ms | L01_B02 | printed as "<5 ms" (no GPU available; text claim) | Consistent |
### NB1 / L02_B02
**CRITICAL DIVERGENCE — domain classifier AUC = NaN:**
NB1 cell 17 (`treinar_classificador_dominio`) produces:
```
AUC do classificador de domínio: nan
→ AUC ≈ 0,5: domínios indistinguíveis — sem shift relevante.
```
with sklearn warning: "Only one class is present in y_true. ROC AUC score is not defined in that case."
This is a bug in the y_true construction: the binary label array appears to contain only one class when passed to `roc_auc_score`. The printed message ("AUC ≈ 0,5: domínios indistinguíveis") is factually wrong — the domains ARE highly distinguishable (shift of ~2.5σ), as confirmed by the decision-map catastrophe already shown. The AUC nan is purely a code error.
Impact: the pedagogical point ("shift is detectable without class labels") is visually supported by the ROC curve shape (the notebook still plots it), but the quantitative claim fails. The instructor notes in L02_B02 acknowledge this: "Note: the AUC metric has a known y_true construction issue (returns NaN); emphasize the ROC curve shape and the pedagogical point verbally." This workaround is insufficient for independent student use.
Blocks that reference this AUC value:
- L02_B01 "Hábitos de detecção de shift" (line ~138): "o classificador de domínio detecta o covariate shift com **AUC > 0,99**"
- L02_B02 "Ato 3" description: "o classificador atinge AUC próxima de 1,0"
- Both are contradicted by the notebook output.
**Numeric verification (NB1 actual outputs vs. block claims):**

| Claim | Source | Actual notebook output | Match? |
|---|---|---|---|
| Source accuracy 100% | L02_B02 | 1.000 | Yes |
| Target accuracy 67.8% | L02_B02 | 0.678 | Yes |
| Zero-shot Macro-F1 = 0.615 | L02_B02 | 0.615 | Yes |
| Confidence wrong predictions = 0.953 | L02_B02 | 0.953 | Yes |
| Confidence correct predictions = 0.979 | L02_B02 | 0.979 | Yes |
| K=10 SSDA = 0.873, B = 0.793 | L02_B02 | 0.873 / 0.793 | Yes |
| K=50 regime B = 0.853, SSDA = 0.854 | L02_B02 | 0.853 / 0.854 | Yes |
| Domain classifier AUC ≈ 1.0 | L02_B01 and L02_B02 | NaN | **NO** |

**SECONDARY DIVERGENCE — L02_B01 K-sweep table uses rounded values inconsistent with NB1:**

L02_B01 section "A curva Macro-F1 × K" (line ~200) displays:

| K | (A) | (B) | (C) |
|---|---|---|---|
| 10 | 0.61 | 0.79 | 0.87 |
| 25 | 0.61 | 0.91 | 0.92 |
| 50 | 0.61 | 0.94 | 0.92 |
| 100 | 0.61 | 0.94 | 0.94 |

The actual NB1 values (`nb1_ksweep.npz`):

| K | (A) | (B) | (C) |
|---|---|---|---|
| 10 | 0.615 | 0.793 | 0.873 |
| 25 | 0.615 | 0.914 | 0.918 |
| 50 | 0.615 | 0.938 | 0.918 |
| 100 | 0.615 | 0.940 | 0.937 |

The L02_B01 table is rounded and approximated. For K=50, the SSDA value is shown as 0.92 in L02_B01 but the actual value is 0.918. For K=100, SSDA is shown as 0.94 but actual is 0.937. The L02_B02 table (which was built from the executed notebook) is exact. A student comparing both blocks would see inconsistent numbers. The L02_B01 claim "A lição da tabela: para K pequeno, o pré-treino da fonte é um ativo valioso" is directionally correct, but the specific values differ.

### NB2 / L03_B02

All numeric claims match notebook outputs.

**Numeric verification (NB2 actual outputs vs. block claims):**

| Claim | Source | Actual notebook output | Match? |
|---|---|---|---|
| Collapse max inter-center distance = 0.056 | L03_B02 | 0.056 | Yes |
| Post-relaxation min inter-center distance = 0.877 | L03_B02 | Not printed in output (loaded from NPZ, figure shown) | Consistent (NPZ loaded; visual confirmed) |
| K-means ARI 2D = 0.743 | L03_B02 | 0.743 | Yes |
| ARI 16D = 0.893 | NB2 cell 28 | 0.893 | (block does not quote 16D ARI; consistent with omission) |

---

## TERMINOLOGY

| Term | Variants found | Files | Suggested canonical (pt-BR) |
|---|---|---|---|
| encoder | "encoder", "codificador" | "encoder" used throughout all blocks; "codificador" absent | **encoder** (borrowing is standard in Brazilian ML) |
| embedding / incorporação | "embedding", "incorporação", "espaço de embeddings", "espaço latente" | L01_B01 introduces both forms; all other blocks use "embedding" | **embedding** with "(ou incorporação)" on first use per block — already done in L01_B01 |
| espaço latente vs. espaço de embeddings | Both used interchangeably | L01_B01, L02_B01, L02_B02, L03_B01, L03_B02 | Acceptable: "espaço latente" for the abstract concept; "espaço de embeddings" for the trained-output space — used consistently in this way |
| aprendizagem contrastiva vs. aprendizado contrastivo | "aprendizagem contrastiva" in block titles (L03_B01); "aprendizado contrastivo" absent | L03_B01 title, L03_B02 title | **aprendizagem contrastiva** — feminine form, consistent |
| sobreajuste vs. overfitting | "sobreajuste (*overfitting*)" on first use; "sobreajuste" thereafter | L01_B02, L02_B01 | **sobreajuste** with "*overfitting*" on first use — already done correctly |
| cabeça vs. head | "cabeça (*head*)" on first use per block; "cabeça" thereafter | L02_B01, L02_B02, L03_B01, L03_B02, L04_B01, L04_B02 | **cabeça** — consistent |
| mudança de domínio vs. domain shift | Both used; "mudança de domínio (*domain shift*)" pattern followed | L02_B01, L02_B02, L04_B02 | Consistent, no action needed |
| c_reg vs. λ_reg | L03_B01 uses $c_\text{reg}$; L03_B02 uses $\lambda_\text{reg}$; L04_B01 uses $c_\text{reg}$ | L03_B01, L03_B02, L04_B01 | **$c_\text{reg}$** — fix L03_B02 |

**One formula inconsistency (beyond notation):**

The regularization term $\mathcal{L}_\text{reg}$ differs between blocks:
- L03_B01 (theory): $\mathcal{L}_\text{reg} = \frac{1}{C}\sum_c \|\boldsymbol{\mu}_c\|^2$ (squared norm)
- L03_B02 (practice): $\mathcal{L}_\text{reg} = \frac{1}{C}\sum_c \|\boldsymbol{\mu}_c\|^2$ (squared norm — matches L03_B01) ✓
- L04_B01 (paper): $\mathcal{L}_\text{reg} = \frac{1}{C}\sum_c \|\boldsymbol{\mu}_c\|$ (un-squared norm — matches the published paper A&A 685 A37)

This is a genuine difference between the toy implementation (squared) and the paper (un-squared). L04_B01 notes the push-margin parametrization difference with the notebook, but does not note the reg-term formula difference. A student comparing NB2 cell 6 code with L04_B01 equations will notice the inconsistency without explanation.

---

## LINKS

### Wikilinks

All `[[L0X_B0Y]]` wikilinks across all 8 blocks resolve to existing files. No broken wikilinks found.

One minor inconsistency in L04_B02 frontmatter YAML: the footer links include `[[L01_B01]]` (the closing relight is a thematic back-reference), but L01_B01 is absent from the `related_blocks` frontmatter list `[L02_B01, L02_B02, L03_B01, L03_B02, L04_B01]`. No functional impact; Obsidian backlinks will work regardless.

### Colab badge URLs

All Colab badge URLs follow the pattern:
`https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/<notebook>.ipynb`

This is consistent across L01_B02, L02_B01 (preview badge), L02_B02, L03_B01 (preview badge), L03_B02, L04_B01 (link to NB2), L04_B02 (link to NB1), and all three notebook cell 0 badges. No broken badge URLs found.

### Missing static PNG fallback wikilinks in L03_B02

The following `![[assets/...]]` image embeds in L03_B02's "Visualização e Slides" section reference files not present in `jax-examples/assets/`:

| Wikilink | Status | Note |
|---|---|---|
| `![[assets/nb2_fig_sandbox_initial.png]]` | **Missing** | Cell 4 renders live; no PNG committed |
| `![[assets/nb2_sandbox_collapsed.png]]` | **Missing** | Cell 8 loads `nb2_sandbox_collapsed.npz` and renders live; NPZ exists but no PNG |
| `![[assets/nb2_fig_kmeans.png]]` | **Missing** | Cell 27 renders live; no PNG committed |

These will display as broken images in Obsidian. The notebook functionality is unaffected (the cells render matplotlib figures directly from the NPZ data). Fix: either produce the three PNGs via `build_assets.py` and commit them, or remove the three wikilinks from L03_B02.

---

## TIMING RISKS

All eight blocks sum to 40 minutes and include a 4-minute question buffer. The timing tables are detailed and internally consistent.

### Flagged sections

**L03_B02 — Act 1 (particle sandbox): 3–12 min (9 min allocation)**

The instructor notes say "Act 1 animation/filmstrip can seduce the room into parameter tweaking; redirect to the poll and move on." The sandbox visualization is the course's signature visual and historically runs over. The timing plan allocates 9 minutes for Act 1 (which includes framing + initial scatter + two poll+reveal cycles + relaxation filmstrip). This is the tightest segment in the practical blocks. The "hard timebox at minute 12" instruction is correct and should be rehearsed.

**L04_B02 — Course-map relight: 26–33 min (7 min allocation)**

The "second relight" requires approximately 30 seconds per block × 8 blocks = 4 minutes, leaving 3 minutes for the frontier slides. The instructor notes correctly warn: "The frontier segment must not become a paper-list dump — 3 slides maximum." However, the data & stakes segment (0–7 min) with 3 J-PAS slides plus the J-spectra figure is also a known expansion risk. If the room is engaged with J-spectra questions, the recognition moment (7–13 min) is at risk. L04_B02 is the densest block of the course.

**L02_B02 — Act 4 (three-regime experiment): 23–32 min (9 min allocation)**

The K-sweep figure and the poll/answer cycle for "why freeze the head" are both in this segment. The instructor notes say "if running behind, cut Act 3's ROC curve narration and go straight from latent scatter to Act 4." This is good guidance. The 9-minute budget for Act 4 is feasible but tight if the domain classifier AUC NaN issue requires extra verbal explanation.

---

## PT-BR QUALITY

The writing across all 8 blocks is strong, idiomatic, and direct. No passages were found that read as machine-translated. The following minor suggestions are offered for naturalness:

**L04_B02, line ~244 (slides description):**
> "É aqui que uma curva ROC vira decisão de telescópio."
Slightly stilted. Suggested revision: "É nesse ponto que uma curva ROC se traduz em decisão operacional de telescópio."

**L04_B01, line ~73 (Lagrangian question):**
> "A questão Lagrangiana: por que importa"
The heading "A questão Lagrangiana" uses "Lagrangiana" as a proper noun modifier, which is uncommon in Brazilian Portuguese academic writing. Possible revision: "A perspectiva Lagrangiana: por que importa". This is a stylistic suggestion, not an error.

**L02_B01, line ~302 (reference section):**
> "Villar et al. 2024 (e referências internas) — sobre domain shift em levantamentos astronômicos"
This reference entry has no DOI, no journal, and no link — it reads as a placeholder. It should either be completed with a real citation or removed.

---

## PRIORITIZED TODO

### P1 — Critical (fix before first use of NB1)

**TODO 1 — Fix NB1 domain classifier AUC = NaN bug**

The AUC computation in NB1 cell 17 (`treinar_classificador_dominio`) passes a `y_true` array that contains only one class to `roc_auc_score`, causing it to return NaN and print a misleading message ("AUC ≈ 0,5: domínios indistinguíveis"). The root cause is likely that `y_true` is constructed as all-zeros or all-ones. Rebuild block L02_B02 is NOT needed; the notebook cell itself must be fixed.

Rebuild instruction: "In `jax-examples/01_domain_shift_toy.ipynb`, cell 17 (`treinar_classificador_dominio`): inspect how `y_true` is constructed before the `roc_auc_score` call. Ensure it is a concatenation of zeros (source labels) and ones (target labels), not a single-class array. Add a diagnostic print before the AUC call: `print('y_true unique:', np.unique(y_true))`. Then re-run `build_assets.py` to regenerate all NB1 assets and re-execute the notebook. Update block L02_B01 line ~138 from 'AUC > 0,99' to match the actual value once the fix is verified."

---

### P2 — High (fix before student distribution)

**TODO 2 — Fix epoch reference in L01_B01**

L01_B01 "Demonstração Prática" section (approximately line 161) states "épocas (0, 10, 100, 1000)". The notebook uses [0, 200, 500, 1000]. Rebuild instruction: "Rebuild block L01_B01 with this feedback: in the section '💻 Demonstração Prática', item 1, change '(0, 10, 100, 1000)' to '(0, 200, 500, 1000)' to match the actual checkpoint epochs in `nb0_epoch0_params.pkl`, `nb0_epoch200_params.pkl`, `nb0_epoch500_params.pkl`, and `nb0_fcnn_params.pkl`."

**TODO 3 — Standardize reg coefficient notation**

L03_B02 uses $\lambda_\text{reg}$ for the regularization coefficient while L03_B01 and L04_B01 both use $c_\text{reg}$. The NB2 notebook code uses `LAMBDA_REG` as the variable name. Rebuild instruction: "Rebuild block L03_B02 with this feedback: in the section '⚙️ Formulação e Conexão com a Física', replace $\lambda_\text{reg}$ with $c_\text{reg}$ in both the perda total equation and in the parameter list for `perda_discriminativa_2d`. Add a parenthetical note: '(denominado `LAMBDA_REG` no código por clareza)' to explain the name mismatch in the notebook variable."

**TODO 4 — Add explanation of reg formula difference between toy and paper**

L04_B01 uses $\mathcal{L}_\text{reg} = \frac{1}{C}\sum_c \|\boldsymbol{\mu}_c\|$ (un-squared norm, matching the published paper). L03_B01 and L03_B02 use the squared norm $\|\boldsymbol{\mu}_c\|^2$. Students will notice this when comparing the recognition slide. Rebuild instruction: "Rebuild block L04_B01 with this feedback: in the section '⚙️ Formulação e Conexão com a Física', after the equation for $\mathcal{L}_\text{reg}$, add a sentence: 'Nota: o notebook [[L03_B02]] usa a norma ao quadrado $\|\boldsymbol{\mu}_c\|^2$ no termo de regularização, enquanto o artigo usa a norma sem elevar ao quadrado $\|\boldsymbol{\mu}_c\|$. Ambas as formas têm o mesmo efeito de ancoragem; a diferença não afeta a intuição física do potencial.'"

**TODO 5 — Align L02_B01 K-sweep table with exact notebook values**

The K-sweep table in L02_B01 uses rounded values (e.g., SSDA K=50 = 0.92 vs. actual 0.918) that are slightly inconsistent with the exact values shown in L02_B02 and produced by the notebook. Rebuild instruction: "Rebuild block L02_B01 with this feedback: in the section '⚙️ Formulação e Conexão com a Física', replace the K-sweep table with the exact values from `nb1_ksweep.npz` (which match the L02_B02 table exactly). The table should also include K=200 row: A=0.615, B=0.938, C=0.929. Update the prose following the table: the SSDA advantage at K=10 is '~8 puntos percentuais' (0.873 - 0.793 = 0.080 = 8.0 pp, not 10 pp as currently stated). The '~10 pontos percentuais' claim on the same page should be corrected to '~8 pontos percentuais'."

---

### P3 — Medium (fix before second iteration)

**TODO 6 — Fix three missing static PNG wikilinks in L03_B02**

`![[assets/nb2_fig_sandbox_initial.png]]`, `![[assets/nb2_sandbox_collapsed.png]]`, and `![[assets/nb2_fig_kmeans.png]]` are referenced in L03_B02 but the files do not exist in `jax-examples/assets/`. Rebuild instruction: "Add the three missing static PNGs to `build_assets.py`: (a) save the initial particle scatter from NB2 cell 4 as `nb2_fig_sandbox_initial.png`; (b) save the collapsed-state scatter from NB2 cell 8 as `nb2_sandbox_collapsed.png`; (c) save the k-means comparison scatter from NB2 cell 27 as `nb2_fig_kmeans.png`. Alternatively, remove the three `![[assets/...]]` wikilinks from the 'Visualização e Slides' section of block L03_B02 if the cells already have adequate fallback PNGs or render live without issue."

**TODO 7 — Fix L04_B01 semantic result rounding inconsistency**

The slide description for Fig. 5 (approximately line 213 of L04_B01) states "acc = 0,86, F1 = 0,83" but the quantitative metrics table in the same block shows "Modelo ML: TPR=0.838, F1=0.838, ACC=0.864". The slide values are rounded approximations, not exact. Rebuild instruction: "Rebuild block L04_B01 with this feedback: in the 'Visualização e Slides' section, Fig. 5 description, change 'acc = 0,86, F1 = 0,83' to 'acc = 0,864, F1 = 0,838' for precision, or add '(arredondado)' parenthetical to signal intentional rounding."

**TODO 8 — Fix L04_B02 frontmatter related_blocks**

L04_B02 has `[[L01_B01]]` in the footer wikilinks but not in the YAML `related_blocks` list. Rebuild instruction: "Rebuild block L04_B02 with this feedback: add `L01_B01` to the `related_blocks` YAML frontmatter list, changing it from `[L02_B01, L02_B02, L03_B01, L03_B02, L04_B01]` to `[L02_B01, L02_B02, L03_B01, L03_B02, L04_B01, L01_B01]`. This aligns the frontmatter with the footer wikilink that references L01_B01 for the course-map relight."

**TODO 9 — Update manifest asset names (NB0 epoch checkpoints)**

The manifest (§NOTEBOOK BRIEF: 00_caixa_de_ferramentas.ipynb) lists `nb0_epoch10_params.pkl` and `nb0_epoch100_params.pkl` as required assets but these were never built. The actual on-disk files are `nb0_epoch200_params.pkl` and `nb0_epoch500_params.pkl`. Rebuild instruction: "Update `.dev/agents/work/course_manifest.md`: in the NB0 'Data and Asset Requirements' section, replace `nb0_epoch10_params.pkl` with `nb0_epoch200_params.pkl` and `nb0_epoch100_params.pkl` with `nb0_epoch500_params.pkl`. Also update the 'PRETRAINED Checkpoints to Produce During Build' list in the same section. Do the same in cell 16's content spec: change '(épocas 0, 10, 100, 1000)' to '(épocas 0, 200, 500, 1000)'."

---

### P4 — Low (polish pass)

**TODO 10 — Minor pt-BR phrasing improvements in L04_B02**

Rebuild instruction: "Rebuild block L04_B02 with this feedback: (a) in the 'Visualização e Slides' section, ROC slide description line 'É aqui que uma curva ROC vira decisão de telescópio', change to 'É nesse ponto que uma curva ROC se traduz em decisão operacional de telescópio'; (b) in the '🔗 Referências' section, the placeholder reference 'Villar et al. 2024 (e referências internas)' has no DOI, journal, or link — either complete the citation or remove it."

---

*Coherence report generated by course-reviewer agent, Pass 4. Assets reviewed: 8 blocks, 3 notebooks (executed), manifest. Findings: 18 items across 5 categories. TODOs: 10 items (1 critical, 4 high, 4 medium, 1 low).*
