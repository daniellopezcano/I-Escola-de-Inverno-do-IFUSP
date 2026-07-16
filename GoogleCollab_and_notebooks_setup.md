# Google Colab & notebook setup

Practical guide for how the three course notebooks move from local development to something ~130 students can open with one click. Student-facing instructions live in each hands-on block's `.md`; this file is for whoever maintains the notebooks.

## 1. Local development & debugging

Notebooks are authored and debugged locally first, never written directly as `.ipynb`. The source of truth for each notebook is a [jupytext](https://jupytext.readthedocs.io/) py-percent script:

```
jax-examples/src_00_caixa_de_ferramentas.py    →  jax-examples/notebooks/00_caixa_de_ferramentas.ipynb
jax-examples/src_01_domain_shift_toy.py        →  jax-examples/notebooks/01_domain_shift_toy.ipynb
jax-examples/src_02_contrastive_embeddings.py  →  jax-examples/notebooks/02_contrastive_embeddings.ipynb
```

Everything runs inside the `WinterSchool` conda environment. Always call the binaries by absolute path — never `conda activate`, never a bare `python` — so the right interpreter is used regardless of shell state:

```bash
# convert py-percent source → notebook
/home/dlopez/miniconda3/envs/WinterSchool/bin/jupytext --to notebook \
    --output jax-examples/notebooks/00_caixa_de_ferramentas.ipynb \
    jax-examples/src_00_caixa_de_ferramentas.py

# execute headless end-to-end (writes outputs back into the .ipynb)
/home/dlopez/miniconda3/envs/WinterSchool/bin/jupyter nbconvert --to notebook --execute \
    --inplace jax-examples/notebooks/00_caixa_de_ferramentas.ipynb
```

Cached data, pretrained checkpoints, and fallback figures generated along the way live in `jax-examples/assets/` and are loaded with a path that resolves relative to the notebook's own location (`jax-examples/notebooks/`), so execution works the same whether nbconvert runs it headless or a student runs it cell-by-cell.

Only once nbconvert finishes clean (no errors, all cells executed) is a notebook considered ready to share.

## 2. Getting a notebook onto Google Colab

Two ways to do it, for two different situations:

### Option A — Manual upload (quick one-off checks)

In Colab: **File → Upload notebook**, then pick the `.ipynb` from `jax-examples/notebooks/` on your machine. Good for a quick sanity check on Colab's runtime, but it creates a disconnected copy with no link back to the repo — not how the notebooks are actually distributed to students.

### Option B — GitHub badge (the real distribution path)

Every hands-on block's `.md` and the root [README](README.md) carry an **"Open in Colab"** badge that points straight at the file in this GitHub repo. Clicking it opens a live, always-up-to-date Colab copy — no upload, no local file needed. This is the route used to share notebooks with the ~130 students.

The badge is just a link with a fixed structure:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<user>/<repo>/blob/<branch>/<path-to-notebook>)
```

For this repo, concretely:

```
https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/00_caixa_de_ferramentas.ipynb
```

- `github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP` — the GitHub `user/repo`.
- `blob/main/...` — same branch+path convention as a normal GitHub file URL, just swapped onto the `colab.research.google.com` domain.
- Colab fetches the `.ipynb` straight from GitHub at click time, so **the badge always reflects whatever is currently pushed to `main`** — push a fix, the badge picks it up immediately, no separate re-export step.

Consequence: the badge route only works for content that is actually pushed to GitHub. A notebook rebuilt locally but not yet committed/pushed will 404 or open a stale version for students.

## 3. Colab-vs-local caveats

### Version drift

Colab's runtime is not the `WinterSchool` conda env — it's whatever Google ships that week. The notebooks were built and verified against:

| Package | Local (`WinterSchool`) |
|---|---|
| Python | 3.14 |
| JAX | 0.10.2 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.0 |

Colab typically ships an older/different Python and package set. JAX in particular changes its API often enough that a notebook proven "BUILD GREEN" locally can still break on Colab. Before sharing a rebuilt notebook widely:

- Actually open it via the badge and run it top-to-bottom on Colab at least once — don't assume local success transfers.
- If a version-specific API is used, prefer pinning with a `!pip install` cell near the top rather than assuming Colab's preinstalled version matches.
- Keep JAX usage to well-established APIs where possible; these notebooks intentionally avoid exotic/bleeding-edge JAX features for this reason.

### Assets need to be reachable from Colab

The notebooks load cached data/checkpoints/figures from `jax-examples/assets/` using a path relative to the notebook's own folder. That works locally and it works on Colab too, **but only if the assets are actually present in the runtime's filesystem** — and a Colab session opened via the GitHub badge does not clone the whole repo, it only fetches the single `.ipynb` file. Two ways to make assets reachable:

**Option 1 — commit the assets to the repo (what this repo does today).**
`jax-examples/assets/` is small enough to live in git (~32 MB total; the largest single file, `mnist_4k.npz`, is 15 MB — well under GitHub's 100 MB hard limit). With assets committed, add a setup cell at the top of the notebook that `git clone`s the repo (shallow) into the Colab runtime so the relative `../assets` path resolves exactly like it does locally:

```python
import os, subprocess, pathlib
if not pathlib.Path("assets").exists() and "COLAB_RELEASE_TAG" in os.environ:
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/daniellopezcano/I-Escola-de-Inverno-do-IFUSP.git",
        "/content/repo",
    ], check=True)
    os.chdir("/content/repo/jax-examples/notebooks")
```

This is the simplest option and the one used here — since all current assets are small, there's no reason to reach for anything heavier.

**Option 2 — download assets on demand from a Colab cell.**
If a future notebook needs assets too large or inappropriate to commit (raw datasets, large pretrained weights), skip committing them and instead add a cell that downloads just what's needed (e.g. from a release artifact, cloud bucket, or `gdown` for a Google Drive file) into `assets/` before the rest of the notebook runs. Only reach for this when Option 1's size limit becomes a real constraint — it adds a network dependency and a point of failure (broken link, rate limit) that Option 1 avoids entirely.

Either way, the notebook should **detect it's running on Colab** (`"COLAB_RELEASE_TAG" in os.environ` or `"google.colab" in sys.modules`) and only run the clone/download cell in that case — locally, `assets/` is already sitting next to the notebook and nothing extra should happen.
