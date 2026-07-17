# Enhancement brief — Notebook 00_caixa_de_ferramentas (block L1B2)

## Goal
Rewrite this notebook for maximum clarity, step-by-step pedagogy, and good pacing,
for ~130 final-year physics undergrads with little coding background. Produce a NEW
revised version (see output-naming in the agent file); do not overwrite the current one.
pt-BR throughout. Simplicity first — no walls of text, one idea per cell.

## Global changes
- Minimize top-of-notebook auxiliary functions. Ideally almost none up front. DEFINE
  each helper INLINE, at the point where it is first needed, so the opening isn't a
  confusing block of abstract utilities.
- Interleave small plotting cells throughout wherever a visual aids understanding.
- Keep the artifact-hygiene rules (nothing committed; self-generate/download at runtime;
  outputs only to the gitignored assets path or /tmp).

## New narrative order (replaces the current NumPy-first opening)
1. **Backbone matemático em JAX — arrays, shapes, broadcasting (SEM NumPy).**
   Remove ALL NumPy. Do everything with jax.numpy directly, since JAX already covers it.
   Step-by-step, print-driven cells so students SEE what happens:
   - creating arrays; inspecting .shape / .dtype
   - broadcasting: split into several tiny cells, each showing one broadcasting rule
     with explicit prints of shapes before/after
   - matrix products done different ways (elementwise vs @ vs jnp.dot / einsum),
     contrasting results and shapes so the "mathematical backbone of ML = matrix ops"
     lands concretely
   Frame this as: matrix products and array ops ARE the machinery of ML.
2. **CPU vs GPU — por que ML ficou viável.** ONLY AFTER the array intro. A small but
   representative timing comparison (e.g. a sizeable matmul) on CPU vs GPU, so students
   see the speed effect that makes training/deployment feasible. Keep it compact.
3. **Plot mínimo em Matplotlib.** The minimalist plotting example (as currently done) —
   also introduces the damped-sinusoid function family we'll study next.
4. **(Opcional) Autodiff visual.** Now that plots exist: optional cells using jax.grad on
   the damped sinusoid, visualizing the derivative(s) w.r.t. different parameters in a
   visual way (curve + its gradient). Optional/🟣.
5. **(Opcional) jit e vmap.** Postpone compilation here as a small optional section
   (micro-benchmark), AFTER the above — not in the opening.

## The FCNN example (core of the notebook)
Keep the current data-generation setup (sample y = f(x) + noise from the damped sinusoid).
Then present the SAME regression twice, mapped as 1-to-1 as possible so students can
compare them side by side:

**Block A — "do zero" (from scratch).** Students see the math directly:
   - define weight matrices + biases explicitly
   - forward pass = explicit matrix products + activations
   - loss (MSE)
   - backprop via jax.grad; SGD update written out explicitly, using mini-batches
   - train; plot the fit evolving
   Goal: feel how the math enters directly (matrices → products → activations → SGD).

**Block B — mesmo exercício com Equinox + Optax.** Reproduce Block A's exact task, but
   now with libraries: define layers via Equinox modules, use a predefined Optax
   optimizer and (where natural) predefined loss utilities, standard training loop.
   Structure the code to mirror Block A step-for-step so the mapping is obvious
   ("this call replaces those explicit lines"). Goal: students see how the from-scratch
   concepts are encapsulated by real JAX ML packages that simplify life.

**Overfitting demo.** After Block B, use a compact network (with the library tools just
   introduced) to demonstrate overfitting on the noisy data (train too long / too big
   capacity → interpolating the noise). This plants "generalization" for L2.

## Dependencies
Equinox and Optax are now required. Ensure the notebook installs them in Colab
(a guarded pip cell) and that they are importable in the WinterSchool env for local
verification; if missing locally, STOP and report so the instructor can add them.
Append them to jax-examples/requirements.txt.

## Keep
- pt-BR markdown, traffic-light convention if already used elsewhere, "Para casa"
  exercises at the end, the Colab badge, and the block's takeaway line.
