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
