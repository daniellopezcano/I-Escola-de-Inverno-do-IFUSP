# Course Development Pipeline — I Escola de Inverno do IFUSP
# Sequential execution — ONE subagent at a time, ALWAYS.

## SOURCE OF TRUTH
course-materials/Master Plan — «...».md  (find: `find course-materials -maxdepth 1 -name "Master Plan*.md"`)
NEVER modify the Master Plan or course-materials/Templates/.

## WORKDIR PATHS (updated: .dev → dev)
- Blocks out:        course-materials/
- Block template:    course-materials/Templates/Block_Template.md
- Papers:            references/  (2602.13902v1.pdf = J-PAS SSDA; aa4896523.pdf = instance-segmentation halos)
- Notebooks out:     jax-examples/notebooks/          (MOVED — .ipynb live here now)
- Notebook source:   jax-examples/src_<name>.py       (py-percent source)
- Notebook assets:   jax-examples/assets/             (caches, checkpoints, fallback PNGs)
- Notebook helpers:  jax-examples/utils/
- Pipeline work:     dev/agents/work/                 (manifest, logs, reports, feedback)
- Feedback file:     dev/agents/work/my_feedback.md   (human review notes for this pass)
- Python env:        /home/dlopez/miniconda3/envs/WinterSchool/bin/  (call binaries by ABSOLUTE path; never `conda activate`, never bare `python`)

## CRITICAL: Skip logic for resumable execution
- Pass 1: check dev/agents/work/course_manifest.md — SKIP if exists (unless asked to rebuild)
- Pass 2: for each notebook, check jax-examples/notebooks/<name>.ipynb AND its
  dev/agents/work/build_logs/<name>.log ending in "BUILD GREEN" — SKIP if both exist
- Pass 3: for each block, check course-materials/<BLOCK_ID>.md — SKIP if exists
- Pass 4: always run (index + coherence report reflect current state)
Report at start: "Resuming. Manifest: yes/no. Notebooks green: N/3. Blocks: M/8."

## Language policy (NON-NEGOTIABLE)
- Student-facing content (notebook markdown, block student sections, 00_INDEX.md): Portuguese (pt-BR).
- Instructor/meta content ([!instructor] callouts, manifest, logs, reports): English.

## Execution — strictly sequential

### Pass 0 — Environment check (orchestrator bash, no subagent)
/home/dlopez/miniconda3/envs/WinterSchool/bin/python -c "import jax, sklearn, matplotlib; print('env OK')"
/home/dlopez/miniconda3/envs/WinterSchool/bin/jupytext --version
/home/dlopez/miniconda3/envs/WinterSchool/bin/jupyter nbconvert --version
If any fails, STOP.

### Pass 1 — Course architecture
Check dev/agents/work/course_manifest.md — if exists, skip. Else delegate to course-architect.

### Pass 2 — Notebooks (order NB0 → NB1 → NB2)
For each of 00_caixa_de_ferramentas, 01_domain_shift_toy, 02_contrastive_embeddings:
  a. Skip if jax-examples/notebooks/<name>.ipynb exists AND log ends "BUILD GREEN".
  b. Else delegate to notebook-builder. WAIT for "BUILD GREEN:" before next.

### Pass 3 — Block files
For each block ID in order (L01_B01, L01_B02, L02_B01, L02_B02, L03_B01, L03_B02, L04_B01, L04_B02):
  a. Skip if course-materials/<BLOCK_ID>.md exists.
  b. Else delegate to block-writer. WAIT for "Block written:" before next.

### Pass 4 — Review and index
Delegate to course-reviewer → course-materials/00_INDEX.md + dev/agents/work/coherence_report.md.

## Partial / incremental runs (THIS revision pass)
- "Rebuild notebook X with feedback: ..." → notebook-builder, explicit overwrite, re-verify BUILD GREEN.
- "Rebuild block L0X_B0Y with feedback: ..." → block-writer, explicit overwrite.
- "Re-run course-reviewer" → always safe.

## Sequential constraint
NEVER launch more than one subagent at a time. Each confirms its output before the next.

## Context
4-lecture mini-course (Jul 21–24 2026, IFUSP), ~130 final-year physics undergrads,
2×40-min blocks/lecture. Arc: representations & tooling (L1) → domain shift/adaptation
(L2) → contrastive learning/instance segmentation (L3) → two research case studies (L4).
Defer to the Master Plan on all content questions.
