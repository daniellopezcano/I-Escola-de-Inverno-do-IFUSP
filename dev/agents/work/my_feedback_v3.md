_Date: 2026-07-21_
## 0. Purpose of this file

This is the authoritative brief for the course-development agents. It records the refactor applied to **Lecture 1** and to the repository, and defines the conventions and standard the agents must apply when revisiting and enhancing **Lecture 2** and **Lecture 3** (and, at lower priority, **Lecture 4**). Where this file disagrees with `my_feedback.md` **this file wins**.

The overriding rule: **Lecture 1 is the gold standard. Make everything else match its structure and philosophy.**

## 1. Guiding philosophy

- **Simplicity first.** Clean, compact materials. Never overwhelm the student with walls of text or scattered resources.
- **The README is the single navigational hub.**
- **Theory markdowns are narrative companions to the slides** — readable prose that follows the deck, slightly more discursive than the slides themselves.
- **Notebook markdowns are minimal** — essentially the Colab link plus one line of framing.
- Every markdown does two jobs: **centralize the relevant links** and **structure the information**. Theory markdowns additionally **narrate the slides**.

## 2. Current repository structure

```
.
├── .claude/
│   ├── agents/{block-writer,course-architect,course-reviewer,notebook-builder}.md
│   └── settings.json
├── CLAUDE.md
├── Course Development Multi-Agent Pipeline — Complete Setup Guide.md
├── course-materials/{L1B1,L1B2,L2B1,L2B2,L3B1,L3B2,L4B1,L4B2}.md
├── dev/agents/work/
│   ├── build_logs/*.log
│   ├── coherence_report.md
│   ├── course_manifest.md
│   ├── my_feedback.md
│   └── my_feedback_v2.md
├── GoogleCollab_and_notebooks_setup.md
├── jax-examples/
│   ├── assets/                  (npz data, pkl checkpoints, png figures — see §5)
│   ├── notebooks/{00_caixa_de_ferramentas,01_domain_shift_toy,02_contrastive_embeddings}.ipynb
│   ├── requirements.txt
│   ├── src_0X_*.py
│   └── utils/make_assets_0X_*.py
├── README.md
└── references/{2311.12110v3.pdf, 2602.13902v1.pdf}
```

## 3. File-role conventions

- **`README.md` — single hub.** Compact navigation across all files + a short statement of each day's aim. The old standalone index file was removed and folded in here.
- **`course-materials/LxBy.md` — one markdown per block, two types:**
    - **Theory block markdown** (paired with a Google Slides deck) → a narrative companion to the slides. Blocks: **L1B1, L2B1, L3B1, L4B1, L4B2**.
    - **Notebook block markdown** (paired with a Colab notebook) → just the link + minimal framing. Blocks: **L1B2, L2B2, L3B2**.
- **`jax-examples/`** — Colab-runnable notebooks, their `src_*.py` mirrors, `utils/make_assets_*.py` generators, `requirements.txt`.
- **`references/`** — source PDFs underpinning the science.
- **`.claude/agents/`** — agent definitions. **`dev/agents/work/`** — agent working artifacts (logs, manifest, coherence report, feedback files).

**Apparent lecture map** (agents should confirm against `README.md` and `dev/agents/work/course_manifest.md`):

- **L1 — foundations / "the map of the territory".** B1 theory = the refactor's gold standard; B2 notebook = `00_caixa_de_ferramentas` (toolbox).
- **L2 — domain adaptation.** B2 notebook = `01_domain_shift_toy`.
- **L3 — contrastive learning.** B2 notebook = `02_contrastive_embeddings`.
- **L4 — inverse problems / reconstruction.** Two theory blocks, no notebook.

## 4. What Lecture 1 now looks like (the template to match)

### 4.1 L1B1 — theory markdown ↔ slides

**Narrative arc** (section order that L2B1/L3B1/L4B1 should emulate in spirit — motivation → concepts → honest limits → pointer to materials):

1. **Apresentação e introdução** — presenter/bio and framing.
2. **Contexto e motivação** — 2.1 _data deluge in cosmology_ (2dFGRS → SDSS-I/II → SDSS-III/IV → DESI DR1 → Rubin/LSST); 2.2 _parallel rise of ML_ (Deep Blue 1997 → ImageNet 2009 → AlexNet 2012 → GANs 2014 → Transformers 2017 → AlphaFold 2020 → ChatGPT 2022).
3. **Usos e aplicações do ML na astrofísica** — 3.1 _what ML does_ (emulation, pattern recognition, SBI [+ inverse problems, anomaly detection, real-time control]) and _what it does not_ (five honest limitations: no reliable extrapolation; no self-certified error / calibration; inherits data bias; no causal modeling; hard to interpret); 3.2 _course topics_, unified by the **latent-space** idea (raw → encoder → latent; "raw distances lie, learned distances reveal") and framed as **responses to the limitations** — **domain adaptation** (aligns train ↔ reality) and **contrastive learning** (builds a space where distance = meaning).
4. **Estrutura do curso e materiais** — points back to the README / GitHub hub.

**Style rules learned from the L1B1 polish:**

- Language: **Brazilian Portuguese**.
- More narrative than the slides, but **compact** — do not bloat.
- Keep the slide section headers; rewrite bodies into flowing prose.
- Inline references favor the **original paper** (DOI / arXiv). A closing "Referências" section lists learning resources and is kept as-is.
- A theory markdown may be a mild superset of the slides, but must **flag any item not on the deck**.

### 4.2 L1B2 — notebook markdown ↔ Colab

- Minimal: the **Colab link + one line of framing**. No narrative.

### 4.3 Repository

- README promoted to the single hub; the standalone index file was removed.

## 5. Repository cleanup tasks (jax-examples / notebooks)

**Do not version-control generated artifacts.** Concretely:

- Remove committed **figures**: `jax-examples/assets/*.png`.
- Remove committed **model checkpoints / params**: `jax-examples/assets/*.pkl`.
- Remove committed **training-output data**: produced `*.npz` (e.g. sandbox states, k-sweeps).
- Keep only **genuine, small input datasets** a notebook truly needs and that are impractical to regenerate — and even then, **prefer generating or downloading them at runtime**.

Then:

- Modify the notebooks (`00`/`01`/`02`) and `utils/make_assets_*.py` so any figures, checkpoints, or training data are written to an **ignored/temporary** location (e.g. a gitignored `assets/` or `/tmp`), never to a committed path. Notebooks must **self-generate or download** whatever they load.
- Update **`.gitignore`** to exclude the assets output directory (and `*.png` / `*.pkl` / produced `*.npz` within it).
- **Verify** each notebook still runs top-to-bottom in a clean Colab with no reliance on committed artifacts.

## 6. Standard to apply to Lectures 2 and 3 (L4 optional)

- **Theory blocks (L2B1, L3B1; then L4B1/L4B2):** a narrative PT markdown companion to the slides, following the L1B1 arc and style — motivation → concepts → honest limits/caveats where relevant → pointer to materials — compact, with original-paper references inline.
- **Notebook blocks (L2B2, L3B2):** a minimal link-only markdown, plus a Colab-clean notebook that stores nothing in the repo (§5).

**Definition of done (per block):**

- **Theory:** reachable from the README; links to its slides; narrative follows the deck's order; Brazilian Portuguese; compact; references are original papers; no orphan or broken links.
- **Notebook:** reachable from the README; markdown is just the Colab link + framing; notebook runs clean in Colab and commits no artifacts.

## 7. Agent pipeline (optional, low priority)

The agent / directory layout (`.claude/agents/*`, `dev/agents/work/*`) **may be simplified if convenient, but this is not a priority.** Keep the four roles' responsibilities intact:

- **course-architect** — README, overall structure, `course_manifest.md`.
- **block-writer** — theory markdowns.
- **notebook-builder** — Colab notebooks **and the §5 cleanup**.
- **course-reviewer** — cross-course coherence (`coherence_report.md`).

## 8. Priorities

1. **Adapt the pipeline** to this file and the new structure — _before running it_.
2. **Lectures 2 and 3** to the L1 standard (theory markdowns + clean notebooks + §5 cleanup).
3. **Lecture 4** only if convenient.