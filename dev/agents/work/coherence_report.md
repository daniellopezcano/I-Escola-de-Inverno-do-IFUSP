# Coherence Report — I Escola de Inverno do IFUSP
# Post-revision-pass QA | Generated: 2026-07-15

---

## SUMMARY

### Assets found vs. manifest

| Asset | Status |
|-------|--------|
| course-materials/L01_B01.md | PRESENT — slide-schematic mode, 13 slides roteiro |
| course-materials/L01_B02.md | PRESENT — thin-supplement mode, notebook pointer table |
| course-materials/L02_B01.md | PRESENT — slide-schematic mode, 13 slides roteiro |
| course-materials/L02_B02.md | PRESENT — thin-supplement mode, notebook pointer table |
| course-materials/L03_B01.md | PRESENT — slide-schematic mode, 16 slides roteiro |
| course-materials/L03_B02.md | PRESENT — thin-supplement mode, notebook pointer table |
| course-materials/L04_B01.md | PRESENT — deep-prose case study (original pass, untouched) |
| course-materials/L04_B02.md | PRESENT — deep-prose case study (original pass, untouched) |
| jax-examples/notebooks/00_caixa_de_ferramentas.ipynb | BUILD GREEN (13 s, 38 cells) |
| jax-examples/notebooks/01_domain_shift_toy.ipynb | BUILD GREEN (12.6 s, 34 cells) |
| jax-examples/notebooks/02_contrastive_embeddings.ipynb | BUILD GREEN (25 s, 35 cells) |
| dev/agents/work/course_manifest.md | PRESENT |
| course-materials/00_INDEX.md | REWRITTEN this pass |

**Missing vs. manifest:** None. All 8 blocks and 3 notebooks are present and green.

**Note on block formats:** L01_B01, L02_B01, L03_B01 are in slide-schematic mode (compact per-slide roteiros with Google Slides link replacing deep-prose). L01_B02, L02_B02, L03_B02 are in thin-supplement mode (notebook-pointer tables). L04_B01, L04_B02 remain in deep-prose mode from the original pass — deliberate, as they are case-study blocks, not slides or demos.

---

## NARRATIVE CONTINUITY

Checking each consecutive block pair for recap/teaser alignment.

### L01_B01 to L01_B02

L01_B01 Slide 13 bridge: "Depois do intervalo: as ferramentas concretas — Python, JAX e uma rede do zero." (pointer to [[L1B2]])

L01_B02 instructor framing: "demo guiada pelo instrutor; você recebe o notebook depois."

L01_B02 Conexão section: "Este bloco é a contraparte prática de [[L1B1]]: o vocabulário introduzido lá (encoder, espaço latente, representação aprendida) ganha corpo em código executável."

VERDICT: Clean bridge. No mismatch.

### L01_B02 to L02_B01

L01_B02 overfitting cell comment (NB0 cell 33, last markdown comment): "Na quarta-feira, veremos o que acontece quando a distribuição de TESTE é diferente da de TREINO."

L02_B01 Slide 2 recap: "Recap de 90 s: NB0 mostrou que generalizar dentro da mesma distribuição já era difícil. Hoje: 'E quando a distribuição de teste é diferente da de treino?'"

VERDICT: Clean, well-matched seed/germination pair.

### L02_B01 to L02_B02

L02_B01 Slide 13 teaser: "No próximo bloco, quebramos um classificador de propósito e o consertamos com exatamente estas ideias."

L02_B02 instructor framing: "laboratório de patologia: causar a doença, diagnosticar, tratar."

L02_B02 Conexão section: "Este bloco é a contraparte prática de [[L2B1]]: o diagrama encoder + cabeça e a analogia dos sotaques, introduzidos no bloco de teoria, ganham existência executável aqui."

L02_B01 instructor notes state that the encoder+head diagram "must be the exact same visual used in NB1." Confirmed: NB1 ATO 2 header (cell 8) reproduces the identical two-row encoder/cabeça table with matching vocabulary.

VERDICT: Clean bridge. No mismatch.

### L02_B02 to L03_B01

L02_B02 ATO 3.2 comment: "Gancho para amanhã: este scatter é o 'palco' que [[L3B1]] articula e [[L3B2]] esculpe ativamente."

L02_B02 optional t-SNE cell (NB1 cell 32): "gancho para amanhã: t-SNE é uma ferramenta de inspeção de embeddings — amanhã veremos por quê e quando ela mente."

L03_B01 Slide 1 recap (spoken, 1 line): "o espaço latente era o palco em L02; hoje ele é o protagonista."

VERDICT: Clean bridge. "palco to protagonista" framing is consistent across both files.

### L03_B01 to L03_B02

L03_B01 Slide 16 teaser: "No próximo bloco, vamos VER um espaço se organizar em tempo real."

L03_B02 instructor framing (min 0-3): "primeiro dinâmica molecular; depois deep learning; no fim vocês percebem que era a mesma coisa."

VERDICT: Clean bridge. The "em tempo real" promise is delivered by the NB2 particle relaxation filmstrip (Act 1) and the MNIST embedding evolution filmstrip (Act 2.6).

### L03_B02 to L04_B01

L03_B02 Resumo cell (NB2 cell 31): "amanhã: esta máquina exata prevendo onde nascem os halos de matéria escura."

L04_B01 "A regra do dia": "Três dias de ingredientes. Hoje, a cozinha de verdade." Recognition moment: "Vocês treinaram esta perda ontem, com bolinhas coloridas; aqui, as bolinhas são partículas do universo."

L04_B01 cross-references NB2 cell 6 (fazer_weinberger_loss) as the code shown side-by-side with the paper equations. Confirmed: NB2 cell 6 is the loss factory function.

VERDICT: Clean bridge. Recognition moment setup is technically correct.

### L04_B01 to L04_B02

L04_B01 bridge (last paragraph): "em [[L4B2]], voltamos ao problema de mudança de domínio do Dia 2 — mas com dados reais de levantamento astronômico. O mesmo par encoder + cabeça, os mesmos três regimes (zero-shot, só alvo, SSDA)..."

L04_B02 opening: "Este é o momento em que o arco se completa. O experimento de três regimes que você executou na quarta-feira em 2D... agora aparece em escala real."

VERDICT: Clean bridge.

---

## NOTEBOOK-BLOCK ALIGNMENT

### NB0 / L01_B02

L01_B02 block table "Figura-troféu" lists "4 painéis: épocas 0/200/500/1000." NB0 cell 30 uses EPOCAS_TROPHY = [0, 200, 500, 1000] and loads nb0_epoch0_params.pkl, nb0_epoch200_params.pkl, nb0_epoch500_params.pkl, nb0_fcnn_params.pkl. MATCH.

NOTE — manifest vs. actual: The manifest NB0 brief (course_manifest.md) lists trophy checkpoints as "epoch 0, epoch 10, epoch 100, epoch 1000." The actual implementation uses 0, 200, 500, 1000. The block file L01_B02 correctly reflects the actual notebook. The manifest is the outlier on this point (harmless but stale).

L01_B02 timing row (min 12-30): "explicit SGD loop (1000 epochs)." NB0 cell 28 uses explicit SGD without any optimizer library. MATCH.

L01_B02 block table "🟡 Pergunta-relâmpago": matches NB0 cell 31 content verbatim. MATCH.

L01_B02 block table "🟢 Mapa de vocabulário": matches NB0 cell 34. MATCH.

MINOR DIVERGENCE — NB0 cell 0 vs. block table: The manifest spec requires the first cell to be a 🟢 Mapa do Curso table. NB0 cell 0 contains both the Colab badge AND a full 4-day Mapa do Curso table (with "L01_B02 <- voce esta aqui" in bold) — so the spec IS met. However, the L01_B02 block pointer table does not list this combined cell; its first row starts at "🟢 O que e este ambiente?" (NB0 cell 1). The block table undercounts the cells by one. Low severity; does not affect student experience.

### NB1 / L02_B02

L02_B02 ATO 3.1: "AUC ≈ 0,785 (cálculo sobre conjunto combinado fonte+alvo — bug corrigido na revisão anterior)." NB1 cell 17 printed output: "AUC do classificador de domínio: 0.785 / AUC > 0,7: shift detectável." MATCH — bug fix confirmed working.

L02_B02 instructor timing (min 10-17): "Encoder+head architecture (picks up the diagram from [[L2B1]])." NB1 ATO 2 header cell (cell 8) reproduces the identical encoder/cabeça table with matching role labels. MATCH.

L02_B02 ATO 4.5: "K-sweep pre-computado (nb1_ksweep.npz), K in {10, 25, 50, 100, 200}; curvas se cruzam." Build log confirms nb1_ksweep.npz generated and nb1_fig_k_sweep.png present. MATCH.

L02_B02 ATO 4.3 instructor note: "jax.grad(..., argnums=0) — gradiente apenas no encoder." NB1 cell 22 implements this pattern. MATCH.

MINOR DIVERGENCE — "Mapa do Curso" table absent from NB1: NB1 cell 0 combines Colab badge + notebook title + subtitle "O ciclo de vida do domain shift em 4 atos" + short "Modo de uso" note, but does NOT include a Markdown table of the 4-day arc. The L02_B02 block table lists the first section as "🟢 | Mapa do Curso | Tabela dos 4 dias; L02_B02 em negrito." The subtitle is present; the full table is absent. NB0 has the full table. See TODO 5.

### NB2 / L03_B02

L03_B02 instructor timing: "🟡 Poll before collapse cell: 'O que acontece com delta_push = 0?'" NB2 cell 7 is the poll markdown, cell 8 runs the collapse. MATCH.

L03_B02 block table "ATO 1.2 — A Perda Contrastiva em JAX": notes "a mesma funcao sera reutilizada no Ato 2." NB2 cell 5 markdown explicitly states this. MATCH.

L03_B02 block table ATO 3.3: "t-SNE 3 perplexidades (5/30/100) — mapa muda visivelmente." L03_B01 Slide 13 specifies the same three perplexities. NB2 cell 29 runs t-SNE with exactly perplexities 5, 30, 100 and prints the constellation warning. Consistent across all three files.

L03_B02 block table notes ARI result is visible in ATO 3.1. NB2 cell 27 output: "ARI (k-means no embedding 2D): 0.743." The INDEX.md (this pass) correctly states ARI = 0,743. MATCH.

L03_B02 "🟣 mini-SimCLR": NB2 cell 32 is the optional SimCLR markdown section, cell 33 the implementation. MATCH.

MINOR DIVERGENCE — "Mapa do Curso" table absent from NB2: Same issue as NB1. NB2 cell 0 combines badge + title + subtitle "Em tres atos — de particulas a digitos a halos" but no 4-day map table. L03_B02 block table first entry lists "Mapa do Curso | Tabela dos 4 dias; L03_B02 em negrito; subtitulo 'Em tres atos...'" The subtitle matches; the table is absent. See TODO 5.

---

## TERMINOLOGY

| Term | Variants found | Files | Status |
|------|---------------|-------|--------|
| encoder | encoder, $f_\theta$, $h_\psi$ | All blocks | Intentional: $f_\theta$ in L01/L03 (full model); $h_\psi$ in L02/L04 (decomposed model with $g_\varphi$ head). No harmful drift. |
| espaço latente | espaço latente, espaço de embeddings, pseudo-espaço | All blocks | "pseudo-espaço" is L04_B01-only, reflecting the paper's term. Consistent within scope. |
| cabeça / head | cabeça, head, cabeça (Head), $g_\varphi$ | L02_B01, L02_B02, L04_B02 | NB1 ATO 2 header uses "Cabeça (Head)" — acceptable parenthetical. Slightly inconsistent: some slides say only "head", others "cabeça". Recommend: "cabeça" in prose, "(head)" in parentheses on first use per block. |
| perda contrastiva / discriminativa / de Weinberger | perda contrastiva (L03_B01), perda discriminativa (NB2 code "perda_discriminativa_2d"), perda de Weinberger (L04_B01) | L03_B01, L03_B02, L04_B01 | Three names for the same pull/push/reg loss. Planned reveal: "contrastiva" in theory -> "discriminativa" in code -> "Weinberger" in paper recognition. L03_B01 instructor notes explicitly prohibit naming the paper early. No harmful drift. |
| sobreajuste / overfitting | sobreajuste (*overfitting*), overfitting | L01_B02 | Correctly uses parenthetical translation on first use. No drift. |
| embed-then-cluster | embed-then-cluster, embed -> cluster, embed-then-cluster | L03_B01, L03_B02, L04_B01 | Hyphen form canonical in prose, arrow form in code snippets. Consistent. |
| AUC / AUC-ROC | AUC, AUC-ROC | L02_B01, L02_B02 | No harmful drift. L02_B01 Slide 13 uses "AUC > 0,7" as detection threshold; L02_B02 reports AUC = 0,785. Both consistent with NB1 output. |
| chiasmo / chiasmus / chiasma | chiasmo (L04_B01), chiasmus (L01_B01), chiasma (L04_B02) | L01_B01, L04_B01, L04_B02 | Three spellings; "chiasma" in L04_B02 student-facing prose is non-standard in pt-BR. See TODO 6. |

---

## LINKS

### Broken Colab badge URLs — HIGH PRIORITY

**Finding 1 — L04_B01.md, section "Demonstracao Pratica" (line ~253):**
Badge reads: `.../jax-examples/02_contrastive_embeddings.ipynb`
Missing `notebooks/` subdirectory. Correct path: `.../jax-examples/notebooks/02_contrastive_embeddings.ipynb`

**Finding 2 — L04_B02.md, section "Demonstracao Pratica" (line ~314):**
Badge reads: `.../jax-examples/01_domain_shift_toy.ipynb`
Missing `notebooks/` subdirectory. Correct path: `.../jax-examples/notebooks/01_domain_shift_toy.ipynb`

**Finding 3 — Previous 00_INDEX.md (now fixed this pass):**
All three notebook Colab badges and "Arquivo:" paths used `jax-examples/` without the `notebooks/` subdirectory. The rewritten 00_INDEX.md (this pass) corrects all three. The two L04 block files still need manual correction (see TODO 1).

### Wikilinks

All [[L0X_B0Y]] wikilinks in all 8 blocks are internally consistent. "Blocos adjacentes" footers at the bottom of each block file correctly reference neighbors. No broken wikilinks detected.

### Asset references

All `jax-examples/assets/nb*.png`, `nb*.pkl`, and `nb*.npz` files referenced in block tables and notebook cells match the build log asset list. Confirmed for all three notebooks.

### Obsidian image placeholders in L04 blocks

L04_B01 and L04_B02 contain approximately 12 Obsidian image wikilinks (`![[assets/L04_B01_*.png]]`). These are instructor preparation references, not student-facing rendered content. They will appear as broken images in Obsidian until the paper figure PNGs are exported and committed. Not a course-delivery blocker for the July dates.

---

## REG TERM NOTATION MISMATCH

**Finding 4 — L03_B01 vs. L04_B01 regularization term formula:**

L03_B01 Slide 7 (and "Notas de Referencia" section) defines the regularization term as:
  L_reg = (1/C) * sum_c ||mu_c||^2   (squared norm)

L04_B01 "Etapa 2 — Perda de instancias" writes:
  L_reg = (1/C) * sum_c ||mu_c||     (non-squared norm)

NB2 cell 6 implements: `L_reg = jnp.mean(jnp.sum(centros**2, axis=-1))` — this is the **squared** norm, consistent with L03_B01 and inconsistent with L04_B01.

This is likely a transcription error in L04_B01 when the paper equations were written into the block file. The functional impact is low (both forms anchor centroids near the origin), but the inconsistency will confuse students who follow the instruction in L04_B01 to "compare line by line with fazer_weinberger_loss." Verify against A&A 685 A37 Section 2.3 LaTeX; then either add a note explaining the paper's convention or correct L04_B01 to show the squared form.

---

## ADAM ATTRIBUTION ERROR

**Finding 5 — L02_B02.md, section "Conexao com a teoria":**

The text states: "O loop de treino reutiliza o mesmo padrao de pytrees e **Adam manual** construido em [[L1B2]]."

L01_B02 and NB0 use raw SGD (no momentum, no second moment). NB0 cell 34 vocabulary map explicitly maps "params - lr x grad = descida do gradiente" — this is SGD. NB0 instructor note: "Do NOT introduce optax or flax — raw pytrees keep the autodiff magic visible." Adam appears for the first time in NB1 (the notebook this block describes).

Correct text: "O loop de treino reutiliza o mesmo padrao de pytrees introduzido em [[L1B2]], estendendo o otimizador de SGD explícito para **Adam manual** (introduzido neste notebook)."

This is a factual error in one sentence of L02_B02. The notebook itself (NB1) is correct; the error is only in the block file prose.

---

## TIMING RISKS

### L03_B01 — 16 slides in 36 content minutes

The timing plan allocates four segments. The highest-risk segment is min 10-18 (8 min for 5 slides: contrastive principle + potentials diagram + pull/push/reg equations + trivial minimum poll + InfoNCE). This segment contains the conceptual climax of the entire course (the interaction potentials as physics analogy), two full equation displays, and one poll. At 8 min for 5 slides, average time per slide is 1.6 min — leaving almost no room for questions or poll processing time.

Specific risk: if students ask questions after the pull/push formula slide (Slide 7), time for the InfoNCE slide (Slide 9, Decision D4) compresses. The block correctly marks InfoNCE as "one slide, enrichment not foundation" — but if it gets squeezed, the spoken sentence "o notebook tem uma celula 🟣 que vai mais fundo" must be the fallback.

Recommended mitigation: move Poll 2 (trivial collapse question) to immediately after Slide 6 (potentials diagram) and before Slide 7 (equations), so the equations follow the poll reveal rather than being interrupted by it.

Optional Slide 14 (Damrich et al. unification) is already marked as sacrificial. Confirm it is removed from the deck if the potentials segment overruns at all.

### L04_B01 — results tour may overrun

Instructor timing allocates min 17-26 (9 min) for a 5-figure results tour (Figs. 5, 7, 9 + watershed slide + speed comparison). At approximately 1.8 min per figure, the HMF discussion (Fig. 9) — which requires explaining log-log axes, the "emergent HMF" concept, and two orders of magnitude in mass — risks consuming 3 min alone. The block correctly hard-caps this segment and prohibits Fig. 8 (violin plot). Enforce the 9-min cap strictly.

### L04_B02 — frontier + second relight in 7 min

The segment min 26-33 contains 3 frontier slides plus a deliberate 4-min course-map relight (8 blocks x 30 s each). Total is 7 min for content that could easily expand to 12 min if frontier slides attract questions. The instructor note correctly identifies this risk and prescribes "shorten frontier to one slide if behind at min 26." The course-map relight is the emotional close of the four-day arc and must not be shortened or cut — frontier slides are the sacrificial content.

---

## PT-BR QUALITY

**Passage 1 — L04_B02 homework exercise (student-facing, high severity):**
File: L04_B02.md, section "Demonstracao Pratica," final paragraph.
Current text: "Reproduced the K-sweep do artigo variando o tamanho do subconjunto..."
Issue: "Reproduced" is English and grammatically incorrect in pt-BR context.
Fix: "**Reproduza** o K-sweep do artigo, variando o tamanho do subconjunto de rotulos J-PAS de 100 a 15 000 objetos. Em qual faixa o SSDA tem vantagem sobre o baseline de so alvo? Em qual faixa os dois convergem? Como isso informa a estrategia de acompanhamento espectroscopico para futuros levantamentos?"

**Passage 2 — L04_B02 "chiasma" (student-facing, medium severity):**
File: L04_B02.md, section "A chiasmo se fecha."
Current text: "o arco do curso foi descrito como uma *chiasma*"
Issue: "chiasma" is Greek/English; pt-BR standard is "o quiasmo" (masculine noun).
Fix: "o arco do curso foi descrito como um *quiasmo*"
Also: L01_B01 Slide 10 uses "chiasmus" — standardize to "quiasmo" across both files.

**Passage 3 — L01_B01 Slide 5 (student-facing, low severity):**
"Nao extrapola de forma confiavel alem do suporte dos dados de treino"
"Suporte" in the statistical sense (support of a distribution) is correct but potentially opaque to undergrads without a probability theory background.
Suggested simplification: "Nao extrapola de forma confiavel para alem da regiao coberta pelos dados de treino."

**Passage 4 — L02_B01 slide descriptions (instructor-facing, cosmetic):**
"Equacao(oes):" appears as a section label in multiple slide descriptions inside L02_B01 and L03_B01. The parenthetical plural is visually awkward. Not student-facing (inside instructor block file). Consider standardizing to "Equacoes:" throughout for cleaner formatting.

---

## PRIORITIZED TODO

Items ordered by impact on student experience and course fidelity. Each is phrased as a ready-to-paste rebuild or edit instruction.

**TODO 1 — Fix broken Colab badge URLs in L04_B01 and L04_B02 (HIGH — will 404 when clicked)**
In L04_B01.md, find the "Demonstracao Pratica" section badge (approximately line 253) and change:
  `jax-examples/02_contrastive_embeddings.ipynb`
to:
  `jax-examples/notebooks/02_contrastive_embeddings.ipynb`

In L04_B02.md, find the "Demonstracao Pratica" section badge (approximately line 314) and change:
  `jax-examples/01_domain_shift_toy.ipynb`
to:
  `jax-examples/notebooks/01_domain_shift_toy.ipynb`

These are two one-line edits; no block rebuild required.

**TODO 2 — Fix Adam attribution error in L02_B02 (MEDIUM — factual error in block prose)**
Rebuild L02_B02 with this feedback: In the "Conexao com a teoria" section, the sentence "O loop de treino reutiliza o mesmo padrao de pytrees e Adam manual construido em [[L1B2]]" is factually incorrect — L01_B02/NB0 uses raw SGD, not Adam. Replace with: "O loop de treino reutiliza o mesmo padrao de pytrees introduzido em [[L1B2]], estendendo o otimizador de SGD explícito para **Adam manual** (introduzido pela primeira vez neste notebook)." All other content in L02_B02 is correct.

**TODO 3 — Fix or clarify reg term formula in L04_B01 (MEDIUM — equation mismatch vs. NB2 code)**
In L04_B01.md "Etapa 2 — Perda de instancias," the formula for L_reg shows ||mu_c|| (non-squared norm). L03_B01 and NB2 cell 6 both use the squared norm ||mu_c||^2. Verify against A&A 685 A37 Section 2.3 LaTeX source. If the paper uses the non-squared form, add an explicit note: "Nota: o artigo usa a norma simples ||mu_c||; o notebook [[L3B2]] usa a norma ao quadrado ||mu_c||^2 por conveniencia de implementacao — o comportamento qualitativo e o mesmo." If the paper uses squared, correct the L04_B01 formula to match.

**TODO 4 — Fix English verb in L04_B02 homework (MEDIUM — pt-BR quality failure in student-facing text)**
In L04_B02.md, section "Demonstracao Pratica," last paragraph: replace "Reproduced the K-sweep do artigo" with "Reproduza o K-sweep do artigo." One-word fix, no rebuild required.

**TODO 5 — Add 4-day Mapa do Curso table to NB1 and NB2 cell 0 (LOW — manifest spec compliance)**
NB0 cell 0 correctly includes a full Markdown table of all 4 days with the current block in bold. NB1 cell 0 and NB2 cell 0 have badge+title+subtitle but no table. To comply with the manifest spec ("continuous orientation for a 130-person heterogeneous room") and the L02_B02/L03_B02 block descriptions, add a table to each cell 0 matching NB0's format. Rebuild instruction for NB1: add the 8-row Mapa do Curso table to `jax-examples/src_01_domain_shift_toy.py` cell 0, with "L02_B02 <- voce esta aqui" in bold, then re-execute and confirm BUILD GREEN. Same for NB2 with L03_B02 in bold.

**TODO 6 — Standardize "quiasmo" spelling in L01_B01 and L04_B02 (LOW — pt-BR quality)**
Rebuild L01_B01 with this feedback: On Slide 10, change "chiasmus" to "quiasmo" in the student-facing description. Rebuild L04_B02 with this feedback: In section "A chiasmo se fecha," change "uma *chiasma*" to "um *quiasmo*." No content change; spelling only.

**TODO 7 — Update course_manifest.md trophy-figure epoch list (INFORMATIONAL)**
The manifest NB0 brief lists checkpoint epochs as "0, 10, 100, 1000." The actual implementation (build log confirmed) uses 0, 200, 500, 1000. The block file L01_B02 correctly reflects the actual notebook. Update the manifest NB0 brief to list "nb0_epoch0_params.pkl, nb0_epoch200_params.pkl, nb0_epoch500_params.pkl, nb0_fcnn_params.pkl" to avoid confusion in future rebuild passes. No code change required.

**TODO 8 — Export and commit L04 slide figure PNGs (INFORMATIONAL — not a July blocker)**
L04_B01 and L04_B02 contain approximately 12 Obsidian image wikilinks referencing PNG files (e.g., `![[assets/L04_B01_fig1_problem.png]]`) that do not exist on disk. These are instructor-side references for slide preparation. Export from the paper PDFs (A&A 685 A37 for L04_B01; arXiv:2602.13902 for L04_B02) and commit to `jax-examples/assets/` before the course for clean Obsidian rendering.

---

*Review scope: 8 block files (L01_B01 through L04_B02), 3 executed notebooks (38+34+35 cells), course manifest, build logs. 8 findings, 8 TODOs.*
