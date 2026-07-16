# Course Development Pipeline — I Escola de Inverno do IFUSP
# Sequential execution — ONE subagent at a time, ALWAYS.

## AUTHORITATIVE BRIEF (read this FIRST, every pass)
dev/agents/work/my_feedback_v2.md is the authoritative spec. It overrides the
Master Plan and all earlier feedback where they conflict. Every subagent MUST
read it before acting. Its overriding rule: **Lecture 1 is the gold standard;
make L2/L3/L4 match its structure and philosophy.**

## SOURCE OF TRUTH (science + original plan)
course-materials/Master Plan — «...».md  (find: `find course-materials -maxdepth 1 -name "Master Plan*.md"`)
NEVER modify the Master Plan or course-materials/Templates/.
Where my_feedback_v2.md and the Master Plan disagree, my_feedback_v2.md wins.

## THE GOLD-STANDARD EXEMPLARS (read before writing any block)
- Theory block standard: course-materials/L1B1.md  (arc: apresentação → contexto/
  motivação → usos e limites honestos → estrutura/materiais; pt-BR narrative
  companion to the slides; compact; original-paper refs inline)
- Notebook block standard: course-materials/L1B2.md  (Colab link + one line of framing)
Do NOT rewrite L1B1 or L1B2 — they are the templates. Match them.

## PATHS
- Blocks:            course-materials/LxBy.md   (naming: L1B1, L2B1, … NOT L01_B01)
- Master Plan:       course-materials/
- Block template:    course-materials/Templates/Block_Template.md
- Papers:            references/  (2602.13902v1.pdf = J-PAS SSDA; 2311.12110v3.pdf = 2nd case study)
- Notebooks:         jax-examples/notebooks/<name>.ipynb
- Notebook source:   jax-examples/src_<name>.py
- Asset generators:  jax-examples/utils/make_assets_<name>.py
- Notebook assets:   jax-examples/assets/   (GITIGNORED — artifacts never committed)
- Pipeline work:     dev/agents/work/
- README hub:        README.md   (single navigational hub; no standalone index file)
- Python env:        /home/dlopez/miniconda3/envs/WinterSchool/bin/  (ABSOLUTE paths; never `conda activate`)

## BLOCK TYPES (from my_feedback_v2 §3)
- THEORY (narrative companion to slides): L1B1, L2B1, L3B1, L4B1, L4B2
- NOTEBOOK (minimal, link-only):          L1B2, L2B2, L3B2

## Skip logic (resumable)
- README.md present and current → skip architect unless asked to rebuild.
- Block LxBy.md exists → skip unless this pass targets it.
- Notebook <name>.ipynb exists AND dev/agents/work/build_logs/<name>.log ends
  "BUILD GREEN" → skip unless this pass targets it.
Report at start: "Resuming. README: yes/no. Blocks done: M/8. Notebooks green: N/3."

## Language
Student-facing content: Brazilian Portuguese (pt-BR). Instructor/meta: English.

## Execution — strictly sequential

### Pass 0 — Environment check
/home/dlopez/miniconda3/envs/WinterSchool/bin/python -c "import jax, sklearn, matplotlib; print('env OK')"
/home/dlopez/miniconda3/envs/WinterSchool/bin/jupyter nbconvert --version
If any fails, STOP.

### This revision pass (targeted; NOT a full rebuild)
Read my_feedback_v2.md, then work in this order, one subagent at a time:
1. course-architect → refresh README.md as the single hub + update course_manifest.md
   to reflect README-centric structure, block types, and the L1 standard.
2. notebook-builder → for 01_domain_shift_toy then 02_contrastive_embeddings:
   apply my_feedback_v2 §5 cleanup (artifacts to gitignored assets/; self-generate
   or download inputs at runtime; nothing committed), apply any block-specific
   feedback (incl. the NB1 AUC bug), re-verify BUILD GREEN in a clean run.
   Also apply §5 cleanup to 00_caixa_de_ferramentas.
3. block-writer → rewrite L2B1 then L3B1 (THEORY mode) to the L1B1 standard;
   then L4B1, L4B2 only if time permits (low priority per §8).
   Rewrite L2B2, L3B2 (NOTEBOOK mode) to the L1B2 minimal standard.
   NEVER touch L1B1 / L1B2.
4. course-reviewer → coherence pass; update dev/agents/work/coherence_report.md;
   confirm README links resolve and no artifacts are tracked.

## Sequential constraint
NEVER launch more than one subagent at a time. Each confirms before the next.
