# Course Development — Authoritative Feedback Brief
_Date: 2026-07-21_

## 0. Purpose of this file

This is the authoritative brief for the course-development agents. It records the refactor applied to **Lecture 1** and to the repository, and defines the conventions and standard the agents must apply when revisiting and enhancing **Lecture 2** and **Lecture 3** (and, at lower priority, **Lecture 4**). Where this file disagrees with any earlier note, **this file (`my_feedback_v2.md`) wins**.

The overriding rule: **Lecture 1 (L1B1) is the gold standard. Make everything else match its structure and philosophy.**

## 1. Guiding philosophy

- **Simplicity first.** Clean, compact materials. Never overwhelm the student with walls of text or scattered resources.
- **The README is the single navigational hub.**
- **Theory markdowns are the narrative source for the slides.** They carry a clear, compact, narrative explanation of the concepts a block must convey. The instructor then builds the Google Slides deck *from* the markdown (see §1.1).
- **Notebook markdowns are minimal** — essentially the Colab link plus one line of framing.
- Every markdown does two jobs: **centralize the relevant links** and **structure the information**. Theory markdowns additionally **narrate the concepts of the block**.

### 1.1 Direction of authoring (important change)

L1B1 was originally written *slides-first*: the deck was built, then the markdown described it. **From now on, invert this.** The agents craft the theory markdown first — a carefully structured, narrative, compact explanation of the block's concepts, in the same format and spirit as L1B1 — and the instructor will afterwards build the Google Slides deck from that markdown. So the markdown is the **primary artifact**; the slides are derived from it.

Consequences:
- The markdown must be self-sufficient as a narrative: concepts explained clearly, in an order that flows, so slides can be lifted directly from its structure.
- Keep explanations **clear and compact** — narrative, but not excessively extended. Match L1B1's density, not more.
- There is no need to reference or reconcile against pre-existing slide content; the markdown defines the content.

## 2. Current repository structure
.  
├── .claude/  
│ ├── agents/{block-writer,course-architect,course-reviewer,notebook-builder}.md  
│ └── settings.json  
├── CLAUDE.md  
├── Course Development Multi-Agent Pipeline — Complete Setup Guide.md  
├── course-materials/{L1B1,L1B2,L2B1,L2B2,L3B1,L3B2,L4B1,L4B2}.md  
├── dev/agents/work/  
│ ├── build_logs/_.log  
│ ├── coherence_report.md  
│ ├── course_manifest.md  
│ ├── my_feedback.md  
│ └── my_feedback_v2.md (this file — authoritative)  
├── GoogleCollab_and_notebooks_setup.md  
├── jax-examples/  
│ ├── assets/ (ONLY genuinely-needed input datasets, if any — see §5)  
│ ├── notebooks/{00_caixa_de_ferramentas,01_domain_shift_toy,02_contrastive_embeddings}.ipynb  
│ ├── requirements.txt  
│ ├── src_0X__.py  
│ └── utils/make_assets_0X_*.py  
├── README.md  
└── references/{2311.12110v3.pdf, 2602.13902v1.pdf}

## 3. File-role conventions

- **`README.md` — single hub.** Compact navigation across all files + a short statement of each day's aim. The old standalone index file was removed and folded in here.
- **`course-materials/LxBy.md` — one markdown per block, two types:**
    - **Theory block markdown** (the narrative source the slides will be built from) → a clear, compact narrative explanation of the block's concepts. Blocks: **L1B1, L2B1, L3B1, L4B1, L4B2**.
    - **Notebook block markdown** (paired with a Colab notebook) → just the link + minimal framing. Blocks: **L1B2, L2B2, L3B2**.
- **`jax-examples/`** — Colab-runnable notebooks, their `src_*.py` mirrors, `utils/make_assets_*.py` generators, `requirements.txt`.
- **`references/`** — source PDFs underpinning the science.
- **`.claude/agents/`** — agent definitions. **`dev/agents/work/`** — agent working artifacts (logs, manifest, coherence report, feedback files).

**Lecture map** (agents should confirm against `README.md` and `dev/agents/work/course_manifest.md`):

- **L1 — foundations / "the map of the territory".** B1 theory = the refactor's gold standard; B2 notebook = `00_caixa_de_ferramentas` (toolbox).
- **L2 — domain adaptation.** B2 notebook = `01_domain_shift_toy`.
- **L3 — contrastive learning.** B2 notebook = `02_contrastive_embeddings`.
- **L4 — scientific case studies (two theory blocks, no notebook).** Backed by the two papers in `references/`: the instance-segmentation / structure-formation paper (`2311.12110v3.pdf`, the public arXiv version) and the J-PAS sim-to-obs domain-adaptation paper (`2602.13902v1.pdf`). The agents should confirm which block anchors to which paper from the manifest; treat both broadly and conceptually, connecting them back to the L2 (domain adaptation) and L3 (contrastive learning / instance segmentation) ideas.

## 4. What Lecture 1 now looks like (the template to match)

### 4.1 L1B1 — theory markdown (the narrative source)

**Narrative arc** (section order that L2B1/L3B1/L4B1/L4B2 should emulate in spirit — motivation → concepts → honest limits → pointer to materials):

1. **Apresentação e introdução** — presenter/framing.
2. **Contexto e motivação** — 2.1 _data deluge in cosmology_ (2dFGRS → SDSS-I/II → SDSS-III/IV → DESI DR1 → Rubin/LSST); 2.2 _parallel rise of ML_ (Deep Blue 1997 → ImageNet 2009 → AlexNet 2012 → GANs 2014 → Transformers 2017 → AlphaFold 2020 → ChatGPT 2022).
3. **Usos e aplicações do ML na astrofísica** — 3.1 _what ML does_ (emulation, pattern recognition, SBI [+ inverse problems, anomaly detection, real-time control]) and _what it does not_ (five honest limitations: no reliable extrapolation; no self-certified error / calibration; inherits data bias; no causal modeling; hard to interpret); 3.2 _course topics_, unified by the **latent-space** idea (raw → encoder → latent; "raw distances lie, learned distances reveal") and framed as **responses to the limitations** — **domain adaptation** (aligns train ↔ reality) and **contrastive learning** (builds a space where distance = meaning).
4. **Estrutura do curso e materiais** — points back to the README / GitHub hub.

**Style rules (the L1B1 standard):**

- Language: **Brazilian Portuguese**.
- **Narrative and self-sufficient** — the markdown is the primary artifact from which slides will be built (§1.1). Concepts explained clearly, in a flowing order.
- **Compact** — clear and to the point; do not bloat. Match L1B1's density.
- Structure the content in clear sections with headers, so slides can be lifted directly from the markdown's structure.
- Inline references favor the **original paper** (DOI / arXiv). A closing "Referências" section lists learning resources.
- Equations in LaTeX only where they earn their place; define each symbol on first use in ≤1 line.

### 4.2 L1B2 — notebook markdown ↔ Colab

- Minimal: the **Colab link + one line of framing**. No narrative.

### 4.3 Repository

- README promoted to the single hub; the standalone index file was removed.

## 5. Repository cleanup tasks (jax-examples / notebooks)

**Do not version-control generated artifacts.** Concretely:

- Do **not** store notebook-generated **figures** (`*.png`) in the repo.
- Do **not** store **model checkpoints / params** (`*.pkl`).
- Do **not** store **produced training data** (`*.npz` outputs such as sandbox states, k-sweeps).
- The `jax-examples/assets/` folder should hold **only genuine input datasets** that a notebook truly needs to run and that must be present locally. **If a notebook needs no such downloaded input, the `assets/` folder can be deleted entirely.** Prefer accessing data directly from within the notebook (compute it, or download it at runtime) over committing anything.

Then:

- Modify the notebooks (`00`/`01`/`02`) and `utils/make_assets_*.py` so any figures, checkpoints, or produced data are written to an **ignored/temporary** location (e.g. a gitignored path or `/tmp`), never to a committed path. Notebooks must **self-generate or download at runtime** whatever they load.
- For any genuinely-needed input dataset (e.g. **MNIST** for `02_contrastive_embeddings`): prefer loading it **directly from the notebook without committing anything** — download it at runtime if necessary; only fall back to a file in `assets/` if direct access is impractical.
- Update **`.gitignore`** to exclude generated artifacts (`*.png`, `*.pkl`, produced `*.npz`) and the ignored assets output path.
- **Verify** each notebook still runs top-to-bottom in a clean Colab with **no reliance on committed artifacts**.

## 6. Standard to apply to Lectures 2 and 3 (L4 optional)

- **Theory blocks (L2B1, L3B1; then L4B1/L4B2):** a narrative PT markdown that is the **source for the slides** (§1.1), following the L1B1 arc and style — motivation → concepts → honest limits/caveats where relevant → pointer to materials — clear, compact, with original-paper references inline. The instructor will build the slides from this markdown afterwards.
- **Notebook blocks (L2B2, L3B2):** a minimal link-only markdown, plus a Colab-clean notebook that stores nothing in the repo (§5).

**Definition of done (per block):**

- **Theory:** reachable from the README; is a clear, compact, self-sufficient narrative of the block's concepts, structured so slides can be built directly from it; Brazilian Portuguese; references are original papers; no orphan or broken links. (A `slides:` link may be recorded in the frontmatter as a placeholder for the deck the instructor will create.)
- **Notebook:** reachable from the README; markdown is just the Colab link + framing; notebook runs clean in Colab and commits no artifacts.

## 7. Agent pipeline (optional, low priority)

The agent / directory layout (`.claude/agents/*`, `dev/agents/work/*`) **may be simplified if convenient, but this is not a priority.** Keep the four roles' responsibilities intact:

- **course-architect** — README, overall structure, `course_manifest.md`.
- **block-writer** — theory markdowns (narrative source for the slides).
- **notebook-builder** — Colab notebooks **and the §5 cleanup**.
- **course-reviewer** — cross-course coherence (`coherence_report.md`).

## 8. Priorities

1. **Adapt the pipeline** to this file and the new structure — _before running it_.
2. **Lectures 2 and 3** to the L1 standard (theory markdowns + clean notebooks + §5 cleanup).
3. **Lecture 4** only if convenient.