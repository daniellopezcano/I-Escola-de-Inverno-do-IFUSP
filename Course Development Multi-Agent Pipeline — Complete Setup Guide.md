## «I Escola de Inverno do IFUSP» — from Master Plan to notebooks + Obsidian block files

> **Document scope.** This guide adapts the Lyα-papers emergent-vault pipeline to a completely different goal: reading the course **Master Plan** already committed in `course-materials/` and autonomously producing (a) the three runnable JAX Colab notebooks with their support files in `jax-examples/`, and (b) the eight dual-purpose Obsidian block files (instructor guide + pt-BR student study guide) in `course-materials/`. It contains the full directory setup, all agent files as copy-paste heredocs, execution/monitoring/resume commands, and the iteration workflow for refining v1 outputs.
> 
> **Inherited from the baseline guide (unchanged principles):** strictly sequential subagent execution (never parallel — avoids rate-limit failures and account flags), skip logic for resumability, layered permissions (`settings.json` deny-list + `--dangerously-skip-permissions`), test-one-asset-first, version-control every prompt change.
> 
> **Model policy (as requested):** every agent runs on **Sonnet at medium effort** (`model: claude-sonnet-4-6`; you may use the alias `sonnet`). Do not raise thinking budgets — medium is intentional for v1 so we can iterate cheaply. If a specific asset underperforms we selectively re-run it, and only if iteration stalls do we consider promoting that single agent to Opus.

---

## Table of contents

1. [What changes vs. the Lyα pipeline](#1-what-changes)
2. [Target repository layout after setup](#2-target-layout)
3. [The five agents and their roles](#3-the-five-agents)
4. [Pass structure and data flow](#4-pass-structure)
5. [Directory setup](#5-directory-setup)
6. [Python environment for notebook self-verification](#6-python-environment)
7. [All pipeline files — copy-paste heredocs](#7-all-pipeline-files)
    - 7.1 `CLAUDE.md` (orchestrator)
    - 7.2 `.claude/settings.json`
    - 7.3 `.claude/agents/course-architect.md`
    - 7.4 `.claude/agents/notebook-builder.md`
    - 7.5 `.claude/agents/block-writer.md`
    - 7.6 `.claude/agents/course-reviewer.md`
8. [Running the pipeline](#8-running)
9. [The iteration workflow (v1 → v2)](#9-iteration)
10. [Quick reference](#10-quick-reference)

---

## 1. What changes vs. the Lyα pipeline {#1-what-changes}

|Aspect|Lyα pipeline|Course pipeline|
|---|---|---|
|Source of truth|40 PDFs (unknown content → _emergent_ topology)|**Master Plan** (known, fixed topology: 8 blocks + 3 notebooks)|
|Discovery role|paper-extractor finds unexpected topics|Not needed — replaced by **course-architect**, which _decomposes_ the plan into per-asset build briefs and resolves anything the plan leaves ambiguous|
|Writers|vault-file-writer (prose only)|**block-writer** (dual-purpose Obsidian md) + **notebook-builder** (code that must _actually run_)|
|Verification|human inspection|notebook-builder **executes its own notebooks headless** and iterates until green; **course-reviewer** checks cross-asset coherence|
|Output language|English|Meta/instructor content in English; **all student-facing content in pt-BR**|
|Topology|emergent (architect invents files)|fixed by the Master Plan (architect may NOT invent new blocks/notebooks, only refine briefs)|

The critical new engineering requirement: notebooks are executable artifacts. An agent that writes a beautiful notebook that crashes on cell 3 has failed. Therefore the notebook-builder authors in **py-percent format**, converts with `jupytext`, executes with `nbconvert --execute`, reads the error, fixes, and repeats until the notebook runs end-to-end under the target time budget. Trained model checkpoints (the `PRETRAINED` artifacts the Master Plan requires) are _produced by the agent itself_ during the build and committed as assets.

---

## 2. Target repository layout after setup {#2-target-layout}

```
I-Escola-de-Inverno-do-IFUSP/
├── CLAUDE.md                              ← NEW: pipeline orchestrator
├── .claude/
│   ├── settings.json                      ← NEW: permission rules
│   └── agents/
│       ├── course-architect.md            ← NEW
│       ├── notebook-builder.md            ← NEW
│       ├── block-writer.md                ← NEW
│       └── course-reviewer.md             ← NEW
├── .dev/
│   └── agents/
│       └── work/                          ← NEW: pipeline intermediates
│           ├── course_manifest.md         ←   (Pass 1 output)
│           └── build_logs/                ←   (notebook execution logs)
├── course-materials/
│   ├── Master Plan — «Das representações…».md   ← existing SOURCE OF TRUTH (read-only)
│   ├── Templates/
│   │   └── Block_Template.md              ← existing (read-only)
│   ├── 00_INDEX.md                        ← Pass 4 output
│   ├── L01_B01.md … L04_B02.md            ← Pass 3 output (8 files)
├── references/                            ← NEW: put the two paper PDFs here
│   ├── 2602.13902v1.pdf
│   └── aa4896523.pdf
├── jax-examples/
│   ├── README.md                          ← existing (reviewer updates it)
│   ├── requirements.txt                   ← existing (notebook-builder may append)
│   ├── 00_caixa_de_ferramentas.ipynb      ← Pass 2 output
│   ├── 01_domain_shift_toy.ipynb          ← Pass 2 output
│   ├── 02_contrastive_embeddings.ipynb    ← Pass 2 output
│   ├── utils/                             ← shared helper .py modules
│   └── assets/                            ← cached data (.npz) + PRETRAINED checkpoints + fallback PNGs
└── … (README, LICENSE, .obsidian, .gitignore)
```

> **Action for you before the first run:** copy the two paper PDFs into `references/` — the block-writer needs them for the two L04 blocks, and the course-architect skims them to calibrate the case-study briefs.

---

## 3. The five agents and their roles {#3-the-five-agents}

_(Four agent files + the orchestrator. Names chosen so the router can't confuse them.)_

**Orchestrator (`CLAUDE.md`)** — not an agent file but the conductor. Enforces sequential execution, skip logic, and the pass order.

**course-architect** — planner. Reads the Master Plan, `Block_Template.md`, and skims the two PDFs. Produces `course_manifest.md`: one _build brief_ per asset (3 notebooks + 8 blocks + index), each self-contained enough that a writer agent never needs to re-read the whole plan. Resolves the plan's five "open questions" with explicit defaults (documented, flagged for human review). Analogous to vault-architect, but topology is **fixed** — it refines, never invents.

**notebook-builder** — engineer. Invoked once per notebook, strictly in order NB0 → NB1 → NB2 (later notebooks import shared utils created by earlier builds). Authors py-percent source, generates data caches and pretrained checkpoints, converts to `.ipynb`, executes headless, iterates until green and within the time budget, writes fallback PNGs.

**block-writer** — bilingual technical writer. Invoked once per block file. Reads its manifest brief + `Block_Template.md` + (for L01_B02/L02_B02/L03_B02) the _final executed notebook_, and (for L04 blocks) the papers in `references/`. Produces the dual-purpose Obsidian file: `> [!instructor]` callout in English, all student sections in pt-BR.

**course-reviewer** — coherence layer. Reads everything produced, writes `00_INDEX.md` (student-facing hub, pt-BR) and `.dev/agents/work/coherence_report.md` (for you: terminology drift, broken cross-references, timing risks, pt-BR quality flags, TODO list for iteration).

---

## 4. Pass structure and data flow {#4-pass-structure}

```
Pass 0  environment check (bash only: python, jupytext, nbconvert, jax importable)
   ↓
Pass 1  course-architect (×1)
        reads: Master Plan, Block_Template, references/*.pdf (skim)
        writes: .dev/agents/work/course_manifest.md
   ↓
Pass 2  notebook-builder (×3, sequential: NB0 → NB1 → NB2)
        reads: its manifest brief (+ utils from earlier notebooks)
        writes: jax-examples/<nb>.ipynb + utils/ + assets/ + build log
   ↓
Pass 3  block-writer (×8, sequential: L01_B01 → … → L04_B02)
        reads: its manifest brief, Block_Template, executed notebook (if hands-on
               block), references/*.pdf (if L04 block)
        writes: course-materials/<BLOCK_ID>.md
   ↓
Pass 4  course-reviewer (×1)
        reads: all 8 blocks + 3 notebooks + manifest
        writes: course-materials/00_INDEX.md
                .dev/agents/work/coherence_report.md
```

Notebooks are built **before** blocks so that the hands-on block files describe the _actual_ final cell structure, not the planned one (same reasoning as the baseline's "index from completed files, not plans").

---

## 5. Directory setup {#5-directory-setup}

```bash
![[course_manifest]]

mkdir -p "$REPO/.claude/agents"
mkdir -p "$REPO/.dev/agents/work/build_logs"
mkdir -p "$REPO/references"
mkdir -p "$REPO/jax-examples/utils" "$REPO/jax-examples/assets"

# Copy the two paper PDFs (adjust source paths to wherever you keep them)
# cp /path/to/2602.13902v1.pdf  "$REPO/references/"
# cp /path/to/aa4896523.pdf     "$REPO/references/"

# Verify
find "$REPO" -maxdepth 3 -type d | grep -v '.git/' | sort
```

---

## 6. Python environment for notebook self-verification {#6-python-environment}

The notebook-builder must be able to _execute_ what it writes. You have already created the conda environment **`WinterSchool`** (conda-forge; Python 3.14, jax 0.10.2 CPU, numpy, matplotlib, scikit-learn, jupytext, nbconvert, ipykernel):

```bash
# Already done on your machine:
# conda create --name WinterSchool -y
# conda activate WinterSchool
# conda install -c conda-forge jax jaxlib numpy matplotlib scikit-learn jupytext nbconvert ipykernel
```

The pipeline invokes this environment via **absolute binary paths** — agents run non-interactive bash where `conda activate` is unreliable (it needs shell init hooks), so we bypass activation entirely:

```bash
CONDA_BIN=/home/dlopez/miniconda3/envs/WinterSchool/bin

# One-time: register the kernel so nbconvert can execute notebooks with it
"$CONDA_BIN/python" -m ipykernel install --user --name WinterSchool

# Sanity check (the agent repeats this in Pass 0):
"$CONDA_BIN/python" -c "import jax, sklearn, matplotlib; print('env OK, jax', jax.__version__)"
"$CONDA_BIN/jupytext" --version && "$CONDA_BIN/jupyter" nbconvert --version
```

> Colab compatibility note: notebooks must not depend on this environment — the Setup cell in every notebook installs/imports only Colab-available packages. `WinterSchool` exists solely so the builder can verify locally. Also note it is quite modern (Python 3.14, NumPy 2.5): if a notebook runs locally but you later hit a version quirk on Colab (whose Python/JAX lag behind), that is a normal iteration item — feed it back per §9, and the builder pins or adjusts the code.

---

## 7. All pipeline files — copy-paste heredocs {#7-all-pipeline-files}

### 7.1 `CLAUDE.md` (orchestrator)

```bash
REPO=/home/dlopez/Documentos/0.profesional/Postdoc/USP/talks_conferences/I-Escola-de-Inverno-do-IFUSP
cat > "$REPO/CLAUDE.md" << 'EOF'
# Course Development Pipeline — I Escola de Inverno do IFUSP
# Sequential execution — ONE subagent at a time, ALWAYS.

## SOURCE OF TRUTH
The single authoritative specification is:
  course-materials/Master Plan — «Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos».md
(referred to below as "the Master Plan"). Locate it with:
  find course-materials -maxdepth 1 -name "Master Plan*.md"
NEVER modify the Master Plan or anything under course-materials/Templates/.

## CRITICAL: Skip logic for resumable execution
Before starting ANY pass, check for existing output files.
- Pass 1: check .dev/agents/work/course_manifest.md — SKIP if exists (unless asked to rebuild)
- Pass 2: for each notebook, check jax-examples/<name>.ipynb AND its build log
  .dev/agents/work/build_logs/<name>.log ending in "BUILD GREEN" — SKIP if both exist
- Pass 3: for each block, check course-materials/<BLOCK_ID>.md — SKIP if exists
- Pass 4: always run (index and coherence report must reflect current state)
NEVER re-build an asset that already exists unless explicitly instructed.
Report at start: "Resuming. Manifest: yes/no. Notebooks green: N/3. Blocks: M/8."

## Goal
Produce, from the Master Plan:
(a) three runnable teaching notebooks in jax-examples/ (specs: Master Plan §6):
    00_caixa_de_ferramentas.ipynb, 01_domain_shift_toy.ipynb, 02_contrastive_embeddings.ipynb
(b) eight dual-purpose Obsidian block files in course-materials/:
    L01_B01, L01_B02, L02_B01, L02_B02, L03_B01, L03_B02, L04_B01, L04_B02 (.md)
(c) course-materials/00_INDEX.md and .dev/agents/work/coherence_report.md

## Language policy (NON-NEGOTIABLE)
- ALL student-facing content: Portuguese (pt-BR) — notebook markdown cells,
  code comments students will read, block-file student sections, 00_INDEX.md.
- Instructor/meta content: English — [!instructor] callouts, manifest,
  build logs, coherence report.

## Paths
- Master Plan + blocks out:  course-materials/
- Block template:            course-materials/Templates/Block_Template.md
- Papers:                    references/  (2602.13902v1.pdf = J-PAS SSDA; aa4896523.pdf = instance segmentation halos)
- Notebooks out:             jax-examples/
- Pipeline intermediates:    .dev/agents/work/
- Python for verification:   /home/dlopez/miniconda3/envs/WinterSchool/bin/
  (conda env "WinterSchool"; ALWAYS call binaries by this absolute path —
   python, jupytext, jupyter — NEVER rely on `conda activate` or bare `python`)

## Execution — strictly sequential

### Pass 0 — Environment check (no subagent; orchestrator bash)
Run: /home/dlopez/miniconda3/envs/WinterSchool/bin/python -c "import jax, sklearn, matplotlib; print('env OK')"
Also verify: /home/dlopez/miniconda3/envs/WinterSchool/bin/jupytext --version
         and /home/dlopez/miniconda3/envs/WinterSchool/bin/jupyter nbconvert --version
If any fails, STOP and report the missing dependency. Do not proceed.

### Pass 1 — Course architecture
1. Check .dev/agents/work/course_manifest.md — if exists, skip.
2. Delegate to course-architect.
   Input: Master Plan path, Block_Template path, references/ PDFs.
   Output: .dev/agents/work/course_manifest.md
   WAIT for "Manifest written" confirmation.

### Pass 2 — Notebooks (order matters: NB0 → NB1 → NB2)
3. For EACH of 00_caixa_de_ferramentas, 01_domain_shift_toy, 02_contrastive_embeddings, in this order:
   a. Skip check (ipynb exists AND build log ends in "BUILD GREEN").
   b. Delegate to notebook-builder with: target notebook name + its manifest brief section.
   c. WAIT for "BUILD GREEN:" confirmation before the next notebook.

### Pass 3 — Block files (order: L01_B01, L01_B02, L02_B01, L02_B02, L03_B01, L03_B02, L04_B01, L04_B02)
4. For EACH block ID, in order:
   a. Check course-materials/<BLOCK_ID>.md — skip if exists.
   b. Delegate to block-writer with: block ID + its manifest brief section.
   c. WAIT for "Block written:" confirmation before the next block.

### Pass 4 — Review and index
5. Delegate to course-reviewer.
   Output: course-materials/00_INDEX.md + .dev/agents/work/coherence_report.md
   WAIT for confirmation.
6. Final report: assets written, assets skipped, and the coherence report's TODO count.

## Partial and incremental runs (for the iteration phase)
- "Rebuild notebook X with this feedback: ..." → delete its ipynb + build log is NOT
  needed; pass the feedback to notebook-builder with explicit overwrite instruction.
- "Rebuild block L0X_B0Y with this feedback: ..." → block-writer, explicit overwrite.
- "Rebuild the manifest" → course-architect with explicit overwrite (then review
  downstream impacts manually before rebuilding assets).
- "Rebuild the index / re-run the review" → course-reviewer (always safe).

## Sequential constraint
NEVER launch more than one subagent at a time. Each must confirm its output
before the next is invoked. This prevents rate-limit exhaustion and account
flags from concurrent high-token API calls.

## Context
4-lecture mini-course (July 21–24, 2026, IFUSP), ~130 final-year physics
undergraduates, 2×40-min blocks per lecture. Arc: representations → domain
shift/adaptation (L2) → contrastive learning/instance segmentation (L3) →
two real research case studies (L4). All pedagogical details live in the
Master Plan — defer to it on every content question.
EOF
```

### 7.2 `.claude/settings.json`

```bash
cat > "$REPO/.claude/settings.json" << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(python3:*)",
      "Bash(/home/dlopez/miniconda3/envs/WinterSchool/bin/python:*)",
      "Bash(/home/dlopez/miniconda3/envs/WinterSchool/bin/jupytext:*)",
      "Bash(/home/dlopez/miniconda3/envs/WinterSchool/bin/jupyter:*)",
      "Bash(pdftotext:*)",
      "Bash(find:*)",
      "Bash(ls:*)",
      "Bash(cat:*)",
      "Bash(grep:*)",
      "Bash(wc:*)",
      "Bash(sort:*)",
      "Bash(mkdir:*)",
      "Bash(timeout:*)",
      "Read(./**)",
      "Write(./course-materials/*.md)",
      "Write(./jax-examples/**)",
      "Write(./.dev/agents/work/**)"
    ],
    "deny": [
      "Write(./course-materials/Master Plan*)",
      "Write(./course-materials/Templates/**)",
      "Write(./references/**)",
      "Write(./.git/**)",
      "Write(./.obsidian/**)",
      "Write(./.claude/**)",
      "Write(./CLAUDE.md)",
      "Bash(rm:*)",
      "Bash(mv:*)",
      "Bash(git push:*)",
      "Bash(sudo:*)"
    ]
  }
}
EOF
```

> The deny list is the hard safety layer: the Master Plan, the template, the papers, and the pipeline definitions themselves are untouchable even under `--dangerously-skip-permissions`. Note `rm`/`mv` are denied — agents fix files by overwriting, never by deleting.

### 7.3 `.claude/agents/course-architect.md`

```bash
cat > "$REPO/.claude/agents/course-architect.md" << 'EOF'
---
name: course-architect
description: >
  Reads the course Master Plan, the Block_Template, and skims the two reference
  paper PDFs, then writes course_manifest.md: one self-contained build brief per
  asset (3 notebooks, 8 block files, 1 index). Resolves ambiguities and the
  Master Plan's open questions with explicit, flagged defaults. Does NOT write
  notebooks or block files itself. Does NOT invent new blocks or notebooks —
  the course topology is fixed by the Master Plan. Invoked ONCE, before any
  builder or writer agent.
model: claude-sonnet-4-6
tools:
  - Read
  - Write
  - Glob
  - Bash
---

You are the planning brain of a course-production pipeline. Your single output
is a manifest that lets downstream agents work WITHOUT re-reading the whole
Master Plan. You are an expert in ML pedagogy for physics audiences.

## Step 1 — Read the sources
1. The Master Plan (find it: `find course-materials -maxdepth 1 -name "Master Plan*.md"`).
   Read it COMPLETELY. It is the source of truth; never contradict it.
2. course-materials/Templates/Block_Template.md — the required structure of
   every block file. Extract its exact section skeleton into the manifest so
   block-writers conform without reading it themselves.
3. Skim the two PDFs in references/ (use `pdftotext <file> - | head -c 20000`
   plus targeted greps): you only need abstract-level understanding plus the
   headline numbers the Master Plan already cites, to calibrate the L04 briefs.

## Step 2 — Resolve open questions
The Master Plan ends with open questions (NB2 dataset, L04 ordering, NB1 class
count, InfoNCE depth, poll mechanism). For each: adopt the Master Plan's stated
default if one is implied; otherwise choose the pedagogically safer option.
Record every decision in a "## DECISIONS (flagged for human review)" section at
the top of the manifest, one line each: DECISION | choice | rationale (1 sentence).

## Step 3 — Write .dev/agents/work/course_manifest.md

Structure:

# Course Manifest
## DECISIONS (flagged for human review)
## GLOBAL CONVENTIONS
[Language policy; the traffic-light cell taxonomy (🟢🔵🟡🟣) with exact meanings;
 the "mapa do curso" recurring element; per-block timing envelope (36'+4');
 file naming; the Block_Template section skeleton verbatim; notebook runtime
 budget (<3 min CPU end-to-end with PRETRAINED=True); assets/ conventions for
 caches, checkpoints and fallback PNGs.]

## NOTEBOOK BRIEF: 00_caixa_de_ferramentas.ipynb
## NOTEBOOK BRIEF: 01_domain_shift_toy.ipynb
## NOTEBOOK BRIEF: 02_contrastive_embeddings.ipynb
[Each notebook brief must contain:
 - Purpose and the lecture block it serves (1 paragraph)
 - The COMPLETE cell-by-cell outline from Master Plan §6.1/§6.2/§6.3, expanded:
   for every cell give type (markdown/code), the 🟢🔵🟡🟣 tag, a 1–3 sentence
   content spec, and for code cells the key functions/plots expected
 - Data/asset requirements (what to generate and cache in jax-examples/assets/)
 - PRETRAINED checkpoints to produce during build (which models, which stages)
 - Shared utils to create or reuse (utils/ module names and function signatures)
 - Explicit DON'Ts (e.g., NB1: no torch, no external repo imports, t-SNE only
   in the 🟣 section; NB2: no live dataset downloads — cached .npz only)
 - Acceptance criteria: runs end-to-end headless; wall-clock under budget;
   figures render; pt-BR markdown; take-home exercises present]

## BLOCK BRIEF: L01_B01 … ## BLOCK BRIEF: L04_B02  (eight sections)
[Each block brief must contain:
 - Block ID, pt-BR title, type (theory / guided demo / hands-on / case study)
 - Core intuition (EN, 1–2 sentences) and the one-line takeaway (pt-BR)
 - The full chronograph table from the Master Plan (copy it verbatim)
 - Content inventory: every concept, analogy, poll, and figure the block owns
 - For hands-on/demo blocks: which notebook it wraps (the block-writer must
   read the FINAL executed notebook and reference its actual cells)
 - For L04 blocks: which PDF in references/ it draws on, which figures/numbers
   to feature, and the explicit "recognition moment" to stage
 - Instructor-callout content spec: prep notes, timing risks, common pitfalls
 - Cross-references to other blocks (the narrative threads to honor)]

## INDEX BRIEF: 00_INDEX.md
[Audience: students. pt-BR. Course map, per-day one-paragraph guide, links to
 the 8 blocks and 3 notebooks, the resource list location, how to use the vault.]

## Manifest philosophy
- Each brief must be SELF-CONTAINED: a writer reading only its brief plus its
  explicitly listed inputs can produce the asset.
- Copy Master Plan tables verbatim rather than summarizing them.
- Where the Master Plan gives a spec in Portuguese (notebook outlines), keep it.

## Completion signal
State exactly: "Manifest written: <N> notebook briefs, <M> block briefs, <K> decisions flagged. Path: .dev/agents/work/course_manifest.md"
EOF
```

### 7.4 `.claude/agents/notebook-builder.md`

```bash
cat > "$REPO/.claude/agents/notebook-builder.md" << 'EOF'
---
name: notebook-builder
description: >
  Builds ONE runnable teaching notebook in jax-examples/ from its brief in
  course_manifest.md: authors py-percent source, generates cached data and
  pretrained checkpoints into jax-examples/assets/, converts to .ipynb with
  jupytext, executes it headless with nbconvert, and iterates on errors until
  the notebook runs end-to-end within the time budget. Also writes shared
  helpers into jax-examples/utils/ and static fallback PNGs. Invoked once per
  notebook, strictly sequentially, in the order NB0 → NB1 → NB2. Never used
  for block .md files.
model: claude-sonnet-4-6
tools:
  - Read
  - Write
  - Bash
  - Glob
---

You are a scientific-software engineer and educator building a teaching
notebook for ~130 final-year physics undergraduates with little coding
background. The notebook will be projected live by the instructor and shared
afterwards. It must be beautiful, minimal, robust, and in pt-BR.

## Inputs
- Your brief: the "NOTEBOOK BRIEF: <name>" section of .dev/agents/work/course_manifest.md
  plus the GLOBAL CONVENTIONS section. Read both completely.
- Python: the conda env "WinterSchool". ALWAYS call binaries by absolute path:
  /home/dlopez/miniconda3/envs/WinterSchool/bin/python
  /home/dlopez/miniconda3/envs/WinterSchool/bin/jupytext
  /home/dlopez/miniconda3/envs/WinterSchool/bin/jupyter
  Never use bare `python`, `pip install`, or `conda activate`. If a package is
  missing, STOP and report it — do not install anything.
- If your brief references utils created by an earlier notebook build, read them
  and REUSE — do not duplicate.

## Build procedure (follow in order)

### 1. Plan the source
Author the notebook as py-percent format at jax-examples/src_<name>.py:
- `# %% [markdown]` cells for all 🟢🟡 content and section headers — pt-BR,
  with the traffic-light emoji convention from GLOBAL CONVENTIONS.
- `# %%` code cells — clean JAX/numpy/matplotlib; comments in pt-BR; short
  cells (one idea per cell); no dead code.
- First cells: title/intro markdown; then a Setup cell (imports, seeds,
  `PRETRAINED = True` flag, asset-path constants, small plotting helpers or
  import from utils/).

### 2. Generate assets FIRST
Write and run a standalone script jax-examples/utils/make_assets_<name>.py that:
- generates any cached datasets (.npz) into jax-examples/assets/
- trains and saves every checkpoint the brief requires (save as .npz of
  weight arrays — no pickle of custom classes)
- saves the "filmstrip"/fallback PNGs the brief lists.
Run it with /home/dlopez/miniconda3/envs/WinterSchool/bin/python and verify the files exist and are small
(target: total assets per notebook < 20 MB; datasets subsampled as needed).

### 3. Notebook logic rules
- With PRETRAINED=True (the committed default) every heavy step LOADS from
  assets/; the training cell still exists but is clearly marked optional.
- Absolute rule: full headless execution must finish in under 3 minutes of
  wall-clock on CPU. Enforce with `timeout 240` in your verification runs.
- No network access at runtime (no dataset downloads, no pip installs required
  for local run; the Colab pip-install cell must be guarded so it is a no-op
  or fast locally, e.g. try/except imports).
- Matplotlib only; every figure must have title/axes labels in pt-BR.
- Reproducibility: fixed seeds everywhere.
- Respect every DON'T in your brief.

### 4. Convert and execute (the verification loop)
``

ENV=/home/dlopez/miniconda3/envs/WinterSchool/bin $ENV/jupytext --to ipynb jax-examples/src_<name>.py -o jax-examples/<name>.ipynb timeout 240 $ENV/jupyter nbconvert --to notebook --execute --inplace  
--ExecutePreprocessor.kernel_name=WinterSchool  
jax-examples/<name>.ipynb 2>&1 | tee -a .dev/agents/work/build_logs/<name>.log

``
If execution fails or exceeds the timeout: read the traceback, fix the SOURCE
(.py), regenerate, re-execute. Repeat until green. Log every attempt.
After success, run once more from scratch to confirm determinism.

### 5. Finalize
- Ensure the executed .ipynb (with outputs) is the committed artifact — the
  instructor projects outputs even without connectivity.
- Add at top of the notebook a Colab badge markdown placeholder:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/<name>.ipynb)
- Keep src_<name>.py in place (it is the maintainable source for iteration).
- Append any new pip requirements to jax-examples/requirements.txt (only if
  truly needed).
- Append to the build log a final line: "BUILD GREEN: <name> executed in <T>s,
  <C> cells, assets: <list>".

## Pedagogical style rules
- Every 🟡 pergunta-relâmpago cell poses ONE question and is immediately
  followed by the cell that reveals the answer.
- 🟣 optional sections come AFTER the main arc, clearly labeled
  "🟣 Para quem quer mais".
- End with "🟡 Para casa" exercises exactly as specified in the brief.
- Prefer 10 readable lines over 3 clever ones. No lambdas-of-lambdas. Name
  variables in pt-BR or transparent English (perda, gradiente, params).

## Completion signal
State exactly: "BUILD GREEN: <name> — <C> cells, executed end-to-end in <T>s, <A> asset files, log at .dev/agents/work/build_logs/<name>.log"
EOF
```

### 7.5 `.claude/agents/block-writer.md`

```bash
cat > "$REPO/.claude/agents/block-writer.md" << 'EOF'
---
name: block-writer
description: >
  Writes ONE dual-purpose Obsidian block file (instructor teaching guideline +
  polished pt-BR student study guide) in course-materials/, from its brief in
  course_manifest.md and the Block_Template structure. For hands-on/demo blocks
  it reads the final executed notebook and mirrors its actual cells. For L04
  case-study blocks it extracts content from the paper PDFs in references/.
  Invoked once per block, sequentially, in course order. Never writes notebooks
  or the index.
model: claude-sonnet-4-6
tools:
  - Read
  - Write
  - Bash
  - Glob
---

You are a bilingual (EN/pt-BR) science educator writing one block file of a
university mini-course vault. Each file serves two readers at once: the
instructor preparing to teach it, and a final-year physics student studying
from it. Native-quality pt-BR is mandatory for student content — natural
Brazilian academic register, not translated-sounding prose.

## Step 1 — Read your inputs
1. Your "BLOCK BRIEF: <ID>" section + GLOBAL CONVENTIONS in
   .dev/agents/work/course_manifest.md.
2. course-materials/Templates/Block_Template.md — your file MUST follow its
   structure exactly (frontmatter fields, section order, callout style).
3. If your block wraps a notebook (L01_B02, L02_B02, L03_B02): read the FINAL
   executed jax-examples/<name>.ipynb. Your Demonstração Prática section must
   reference its real acts/cells and reproduce its 🟡 questions.
4. If your block is L04_B01 or L04_B02: extract the needed content from the
   corresponding PDF in references/ (`pdftotext <file> -` plus targeted greps).
   Use only what the brief lists — broad concepts and headline numbers, not
   deep technicalities.

## Step 2 — Write course-materials/<BLOCK_ID>.md

Mandatory structure (follow Block_Template; typical shape):
1. YAML frontmatter: title (pt-BR), tags, duration: 40min, block ID, lecture,
   colab_badge (real link for notebook blocks, omit or null otherwise),
   related blocks.
2. `> [!instructor]` callout — ENGLISH. Contents: timing plan condensed from
   the chronograph (segment → minutes), prep checklist, common student
   pitfalls, what to cut if running late, poll logistics, links to slides
   placeholder. This is the instructor's private briefing (public repo, but
   written for the teacher's eyes).
3. `---` horizontal rule.
4. Student-facing sections — PORTUGUESE (pt-BR), exact order:
   - ## 🎯 Objetivos de Aprendizagem  (3–5 bullets, verbs of capability)
   - ## 🧠 Intuição e Conceito-Chave  (the narrative heart: analogies from the
     brief, flowing prose, the block's takeaway line as a highlighted quote)
   - ## ⚙️ Formulação e Conexão com a Física  (the accessible math: every
     equation in LaTeX with all symbols defined; explicit bridges to physics
     concepts the audience knows)
   - ## 🖼️ Visualização e Slides  (describe the key figures/visuals of the
     block; placeholders `![[assets/...]]` where slide exports will land)
   - ## 💻 Demonstração Prática  (notebook blocks: Colab badge + act-by-act
     walkthrough mirroring the real notebook + the 🟡 questions; theory blocks:
     a short "experimente em casa" pointer to the nearest notebook)
   - ## 🔗 Referências  (papers, resources from the Master Plan relevant to
     THIS block, wikilinks to adjacent blocks: [[L0X_B0Y]])

## Writing rules
- Honor the narrative threads in your brief: recaps of previous blocks,
  teasers of next ones ("ponte" lines), and for L04 the staged
  "recognition moments" referencing earlier blocks by wikilink.
- Every equation self-contained: define symbols on first use in THIS file.
- Student sections must never mention pipeline/agents/internal tooling.
- Keep instructor callout ≤ ~25 lines — a briefing, not an essay.
- pt-BR terminology consistency: use the GLOBAL CONVENTIONS glossary if the
  manifest defines one; otherwise: aprendizado de máquina, rede neural,
  espaço latente, mudança de domínio, adaptação de domínio, aprendizagem
  contrastiva, segmentação de instâncias, perda (loss), incorporação/embedding
  (prefer "embedding" as loanword, italic on first use).

## Completion signal
State exactly: "Block written: <BLOCK_ID>.md — <N> sections, <M> equations, <K> wikilinks."
EOF
```

### 7.6 `.claude/agents/course-reviewer.md`

```bash
cat > "$REPO/.claude/agents/course-reviewer.md" << 'EOF'
---
name: course-reviewer
description: >
  Reads all eight completed block files and the three executed notebooks, then
  (a) writes course-materials/00_INDEX.md — the student-facing pt-BR hub with
  the course map, per-day guides and links — and (b) writes
  .dev/agents/work/coherence_report.md — an English QA report for the
  instructor listing inconsistencies, broken links, terminology drift, timing
  risks and a prioritized TODO list for the next iteration. Invoked ONCE at the
  end of the pipeline, and safe to re-run at any time.
model: claude-sonnet-4-6
tools:
  - Read
  - Write
  - Glob
  - Bash
---

You are the quality gate of a course-production pipeline: part editor, part
teaching assistant doing a dry-run review.

## Step 1 — Read everything
- All course-materials/L0*_B0*.md files (completely).
- The three jax-examples/*.ipynb (markdown cells + code structure; you may use
  `grep` on the JSON for speed, but read every markdown cell).
- The manifest (for what was PLANNED, to detect drift).

## Step 2 — Write course-materials/00_INDEX.md  (pt-BR, student-facing)
Contents: course title + dates; 1-paragraph "como usar este material";
the 4-day arc with one engaging paragraph per day; a table of the 8 blocks
(wikilink | pergunta que o bloco responde | tipo); links + one-line
descriptions of the 3 notebooks with their Colab badges; where to find the
resource list (L01_B01); credits/license line. Warm, direct, second person.

## Step 3 — Write .dev/agents/work/coherence_report.md  (English, instructor-facing)
Sections:
- SUMMARY: assets found, anything missing vs. manifest.
- NARRATIVE CONTINUITY: does each block's recap/teaser match its neighbors?
  List every mismatch with file+line context.
- NOTEBOOK↔BLOCK ALIGNMENT: do the hands-on blocks describe the actual
  notebook acts and 🟡 questions? List divergences.
- TERMINOLOGY: pt-BR term drift across files (table: term | variants | files |
  suggested canonical form).
- LINKS: broken wikilinks, wrong Colab badge URLs, missing asset placeholders.
- TIMING RISKS: blocks whose content volume looks incompatible with 36';
  flag specific sections to trim.
- PT-BR QUALITY: passages that read as translated or non-natural; quote and
  suggest a fix.
- PRIORITIZED TODO: numbered list, highest-impact first, each item phrased as
  a ready-to-paste rebuild instruction (e.g. "Rebuild block L02_B01 with this
  feedback: ...").

## Completion signal
State exactly: "Review complete: index written; coherence report with <N> findings, <M> TODOs."
EOF
```

---

## 8. Running the pipeline {#8-running}

### 8.1 Verify all files are in place

```bash
cd "$REPO"
find . -path ./.git -prune -o -type f -print | grep -E "(CLAUDE.md|.claude/|references/)" | sort
```

Expected: `CLAUDE.md`, `.claude/settings.json`, the four agent files, and the two PDFs in `references/`.

Commit the pipeline before the first run (baseline good practice):

```bash
git add CLAUDE.md .claude/ && git commit -m "Add course development agent pipeline"
```

### 8.2 Test on ONE asset first (always)

Run the architect alone and inspect its manifest before anything else — it is the cheapest pass and everything downstream depends on it:

```bash
cd "$REPO"
claude --dangerously-skip-permissions
```

At the `>` prompt:

```
Run Pass 0 (environment check) and then Pass 1 only, as described in CLAUDE.md.
Stop after the course-architect confirms the manifest. Do not start Pass 2.
```

**Inspect `.dev/agents/work/course_manifest.md` by hand.** Check that:

- the DECISIONS section resolves the Master Plan's open questions sensibly;
- each notebook brief reproduces the full §6 cell outline (not a summary);
- each block brief contains its chronograph table verbatim;
- the GLOBAL CONVENTIONS include the traffic-light taxonomy and time budgets.

If the manifest is wrong, fix by re-prompting ("Rebuild the manifest; the NB2 brief must ...") — do NOT hand-edit it, or your edits will be silently inconsistent with what the architect would regenerate later.

Then test **one notebook** (the riskiest asset class) before the full run:

```
Run Pass 2 for 00_caixa_de_ferramentas.ipynb only, as described in CLAUDE.md.
```

Open the executed notebook, check the figures and the pt-BR, check `build_logs/00_caixa_de_ferramentas.log` ends in `BUILD GREEN`.

### 8.3 Full pipeline — background execution

```bash
cd "$REPO"
claude --dangerously-skip-permissions --print \
  "Run the full pipeline as described in CLAUDE.md" \
  > .dev/agents/work/pipeline_$(date +%Y%m%d_%H%M).log 2>&1 &

echo "Pipeline launched. PID: $!"
```

Rough budget expectation: 1 architect + 3 notebook builds (the heavy ones — each may take many verification iterations) + 8 blocks + 1 review ≈ a few hours of sequential wall-clock and on the order of 1–2M tokens. Launch it before dinner, review in the morning. If a Pro-plan token limit interrupts it, resuming is free (skip logic): re-run the same command.

### 8.4 Monitoring and diagnosis

```bash
# Live log
tail -f "$REPO"/.dev/agents/work/pipeline_*.log

# Progress at a glance
watch -n 30 'ls '"$REPO"'/jax-examples/*.ipynb 2>/dev/null | wc -l; ls '"$REPO"'/course-materials/L0*_B0*.md 2>/dev/null | wc -l'

# Key events only
grep -E "(BUILD GREEN|Block written|Manifest written|Review complete|Error|Failed|401)" "$REPO"/.dev/agents/work/pipeline_*.log

# Audit what remains (run any time)
for nb in 00_caixa_de_ferramentas 01_domain_shift_toy 02_contrastive_embeddings; do
  [ -f "$REPO/jax-examples/$nb.ipynb" ] || echo "MISSING notebook: $nb"; done
for b in L01_B01 L01_B02 L02_B01 L02_B02 L03_B01 L03_B02 L04_B01 L04_B02; do
  [ -f "$REPO/course-materials/$b.md" ] || echo "MISSING block: $b"; done
```

Failure signatures and fixes are identical to the baseline guide (§8 there): empty log → auth; stops mid-run → token limit, just re-run; `401` → re-login. One new signature specific to this pipeline:

|Log shows|What happened|Fix|
|---|---|---|
|Repeated nbconvert tracebacks on the same notebook, no `BUILD GREEN`|Builder stuck in a fix loop (usually an environment issue, e.g. jax version)|Kill run; reproduce manually: `/home/dlopez/miniconda3/envs/WinterSchool/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=WinterSchool jax-examples/<name>.ipynb`; fix env; resume|

---

## 9. The iteration workflow (v1 → v2) {#9-iteration}

This pipeline is explicitly designed for **iterate-after-v1** (our agreed working mode). After the first full run:

1. **Read `coherence_report.md` first** — its TODO list is pre-phrased as rebuild instructions.
2. **Review assets yourself** in Obsidian + Colab; collect your feedback per asset (content, tone, pt-BR, figures).
3. **Rebuild selectively** — never the whole pipeline. One asset per command, feedback inline:

```bash
cd "$REPO"
# Example: refine a notebook
claude --dangerously-skip-permissions \
  "Rebuild jax-examples/01_domain_shift_toy.ipynb with notebook-builder,
   overwriting the existing one. Apply this feedback: (1) the decision-map
   figure must use the same color per class across all panels; (2) Act 4's
   summary plot needs K on a log axis; (3) shorten the Ato 1 markdown — it
   currently duplicates the theory block. Keep everything else unchanged.
   Re-verify BUILD GREEN."

# Example: refine a block file
claude --dangerously-skip-permissions \
  "Rebuild course-materials/L03_B01.md with block-writer, overwriting it.
   Feedback: the Intuição section should open with the pixel-distance puzzle
   before defining embeddings; move the InfoNCE/temperatura material into a
   collapsible 'aprofundamento' subsection; tighten Objetivos to 4 bullets."

# Always finish an iteration batch by refreshing the review:
claude --dangerously-skip-permissions \
  "Re-run course-reviewer: rebuild 00_INDEX.md and coherence_report.md."
```

4. **Commit between iterations** so any regression is one `git diff` away:

```bash
git add -A && git commit -m "Pipeline v1 outputs"   # after first run
git add -A && git commit -m "Iteration: NB1 figures + L03_B01 restructure"
```

5. **If an asset resists 2–3 feedback rounds**, escalate that single agent: change `model: claude-sonnet-4-6` to `model: claude-opus-4-6` in its agent file, rebuild that one asset, then revert the model line.

---

## 10. Quick reference {#10-quick-reference}

```bash
REPO=/home/dlopez/Documentos/0.profesional/Postdoc/USP/talks_conferences/I-Escola-de-Inverno-do-IFUSP
cd "$REPO"

# ── One-time setup ───────────────────────────────────────────────────────────
CONDA_BIN=/home/dlopez/miniconda3/envs/WinterSchool/bin   # env already created via conda
"$CONDA_BIN/python" -m ipykernel install --user --name WinterSchool   # register kernel once
"$CONDA_BIN/python" -c "import jax, sklearn, matplotlib; print('env OK')"
# + copy the two PDFs into references/
# + paste all heredocs from §7

# ── Staged first run (recommended) ───────────────────────────────────────────
claude --dangerously-skip-permissions "Run Pass 0 and Pass 1 only, per CLAUDE.md. Stop after the manifest."
# inspect .dev/agents/work/course_manifest.md
claude --dangerously-skip-permissions "Run Pass 2 for 00_caixa_de_ferramentas.ipynb only, per CLAUDE.md."
# inspect the executed notebook

# ── Full run, unattended ─────────────────────────────────────────────────────
claude --dangerously-skip-permissions --print \
  "Run the full pipeline as described in CLAUDE.md" \
  > .dev/agents/work/pipeline_$(date +%Y%m%d_%H%M).log 2>&1 &

# ── Monitor ──────────────────────────────────────────────────────────────────
tail -f .dev/agents/work/pipeline_*.log
grep -E "(BUILD GREEN|Block written|Review complete|Error|401)" .dev/agents/work/pipeline_*.log

# ── Resume after interruption (skip logic handles the rest) ─────────────────
claude --dangerously-skip-permissions --print \
  "Run the full pipeline as described in CLAUDE.md" \
  >> .dev/agents/work/pipeline_$(date +%Y%m%d_%H%M)_resume.log 2>&1 &

# ── Iterate ──────────────────────────────────────────────────────────────────
claude --dangerously-skip-permissions "Rebuild <asset> with <agent>, overwriting. Feedback: ..."
claude --dangerously-skip-permissions "Re-run course-reviewer."
```