#!/usr/bin/env python
"""
make_assets_00_caixa_de_ferramentas.py
---------------------------------------
Generates all cached checkpoints and static PNG fallbacks for
notebook 00_caixa_de_ferramentas.ipynb.

Outputs (all in jax-examples/assets/):
  nb0_epoch0_params.pkl          – initial random model
  nb0_epoch200_params.pkl        – partial fit (Adam epoch 200)
  nb0_epoch500_params.pkl        – improving fit (Adam epoch 500)
  nb0_fcnn_params.pkl            – final model (Adam epoch 1000)
  nb0_overfit_params.pkl         – over-wide network (Adam epoch 5000)
  nb0_fig_trophy.png             – 4-panel trophy figure
  nb0_fig_overfit.png            – side-by-side overfit comparison

Design notes:
  - Adam is used here (manually, no optax) so the checkpoints are visually
    beautiful.  The notebook's training cell uses vanilla SGD to expose the
    concept; PRETRAINED=True (the default) loads these checkpoints instead.
  - Trophy epochs chosen as {0, 200, 500, 1000} instead of {0, 10, 100, 1000}
    because Adam's convergence profile produces clearly-distinguishable fits
    at those milestones on this dataset.

Run:
  /home/dlopez/miniconda3/envs/WinterSchool/bin/python \
    jax-examples/utils/make_assets_00_caixa_de_ferramentas.py
"""

import os
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

t0 = time.time()

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
MASTER_KEY = jax.random.PRNGKey(42)

# ── Data parameters (must match notebook exactly) ──────────────────────────────
N_POINTS = 200
A        = 1.5
LAM      = 2.0    # λ — wavelength
TAU      = 6.0    # τ — decay constant
SIGMA    = 0.15   # noise std
X_MAX    = 4.0 * np.pi

# ── Generate dataset ───────────────────────────────────────────────────────────
key = MASTER_KEY
key, k_x, k_noise = jax.random.split(key, 3)

x_raw = jax.random.uniform(k_x, (N_POINTS,), minval=0.0, maxval=X_MAX)
x_np  = np.sort(np.array(x_raw))          # sorted for clean line-plots
noise = np.array(jax.random.normal(k_noise, (N_POINTS,))) * SIGMA


def target_fn(x):
    return A * np.sin(2.0 * np.pi * x / LAM) * np.exp(-x / TAU)


y_true_np  = target_fn(x_np)
y_noisy_np = y_true_np + noise

# Dense evaluation grid (plotting only)
x_grid      = np.linspace(0.0, X_MAX, 500)
y_grid      = target_fn(x_grid)

# ── Input normalisation: x in [0, X_MAX] → [-1, 1] ───────────────────────────
def normalise_x(x):
    return 2.0 * x / X_MAX - 1.0

x_norm    = normalise_x(x_np)
xg_norm   = normalise_x(x_grid)

print(f"Dataset ready: {N_POINTS} pts, x in [0, 4π], "
      f"y_noisy in [{y_noisy_np.min():.3f}, {y_noisy_np.max():.3f}]")
print(f"Noise floor = σ² = {SIGMA**2:.4f}")

# ── MLP functions (must match notebook exactly) ─────────────────────────────
def init_params(layer_sizes, key):
    """Glorot uniform initialisation. Returns list of (W, b) tuples."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, kw = jax.random.split(key)
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        scale   = np.sqrt(6.0 / (fan_in + fan_out))
        W = jax.random.uniform(kw, (fan_in, fan_out),
                               minval=-scale, maxval=scale)
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def forward(params, x):
    """Forward pass: tanh hidden layers, linear output.
    x: (N, in_dim) → (N, out_dim)."""
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def mse_loss(params, x_batch, y_batch):
    """Mean squared error — the only loss used in this notebook."""
    y_pred = forward(params, x_batch).squeeze(-1)
    return jnp.mean((y_pred - y_batch) ** 2)


def predict_np(params, x_norm_1d):
    """Convenience wrapper: 1-D normalised numpy array → numpy output."""
    x_in = jnp.array(x_norm_1d, dtype=jnp.float32).reshape(-1, 1)
    return np.array(forward(params, x_in).squeeze(-1))


# ── Adam helper (no optax; used only in this asset script) ────────────────────
def adam_step(params, grads, m_state, v_state, t,
              lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam update. t is the 1-based iteration counter."""
    new_m = [(beta1 * mW + (1 - beta1) * gW,
              beta1 * mb + (1 - beta1) * gb)
             for (mW, mb), (gW, gb) in zip(m_state, grads)]
    new_v = [(beta2 * vW + (1 - beta2) * gW ** 2,
              beta2 * vb + (1 - beta2) * gb ** 2)
             for (vW, vb), (gW, gb) in zip(v_state, grads)]
    mh = [(mW / (1 - beta1 ** t), mb / (1 - beta1 ** t))
          for mW, mb in new_m]
    vh = [(vW / (1 - beta2 ** t), vb / (1 - beta2 ** t))
          for vW, vb in new_v]
    new_p = [(W - lr * mWh / (jnp.sqrt(vWh) + eps),
              b - lr * mbh / (jnp.sqrt(vbh) + eps))
             for (W, b), (mWh, mbh), (vWh, vbh)
             in zip(params, mh, vh)]
    return new_p, new_m, new_v


def train_adam(layer_sizes, n_epochs, lr, init_key, save_at,
               print_every=200):
    """Train with Adam; return (final_params, {epoch: params_snapshot})."""
    params = init_params(layer_sizes, init_key)
    xi = jnp.array(x_norm, dtype=jnp.float32).reshape(-1, 1)
    yi = jnp.array(y_noisy_np, dtype=jnp.float32)

    m_state = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    v_state = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]

    grad_fn = jax.grad(mse_loss)
    checkpoints = {}
    if 0 in save_at:
        checkpoints[0] = [(np.array(W), np.array(b)) for W, b in params]

    for t in range(1, n_epochs + 1):
        grads = grad_fn(params, xi, yi)
        params, m_state, v_state = adam_step(
            params, grads, m_state, v_state, t, lr=lr)

        if t in save_at:
            checkpoints[t] = [(np.array(W), np.array(b)) for W, b in params]

        if print_every and t % print_every == 0:
            l = float(mse_loss(params, xi, yi))
            print(f"  epoch {t:5d}  loss={l:.6f}")

    final_loss = float(mse_loss(params, xi, yi))
    print(f"  epoch {n_epochs:5d} (final)  loss={final_loss:.6f}  "
          f"[noise floor ≈ {SIGMA**2:.4f}]")
    return params, checkpoints


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Normal network  [1, 32, 32, 1] — 1000 Adam epochs
#     Trophy checkpoints at {0, 200, 500, 1000}
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Training normal network [1,32,32,1], 1000 Adam epochs ===")
key, init_key = jax.random.split(key)

TROPHY_EPOCHS = {0, 200, 500, 1000}

params_normal, ckpts = train_adam(
    layer_sizes=[1, 32, 32, 1],
    n_epochs=1000,
    lr=0.01,
    init_key=init_key,
    save_at=TROPHY_EPOCHS,
    print_every=200,
)

# Save
epoch_to_fname = {
    0:    "nb0_epoch0_params.pkl",
    200:  "nb0_epoch200_params.pkl",
    500:  "nb0_epoch500_params.pkl",
    1000: "nb0_fcnn_params.pkl",
}
for ep, fname in epoch_to_fname.items():
    path = os.path.join(ASSETS_DIR, fname)
    with open(path, "wb") as f:
        pickle.dump(ckpts[ep], f)
    print(f"  Saved {fname}  ({os.path.getsize(path)/1024:.1f} KB)")

# Sanity check
y_pred_normal = predict_np(params_normal, xg_norm)
print(f"  Final pred range: [{y_pred_normal.min():.3f}, {y_pred_normal.max():.3f}]"
      f"  (true: [{y_grid.min():.3f}, {y_grid.max():.3f}])")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Overfit network  [1, 128, 128, 128, 1] — 5000 Adam epochs
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Training overfit network [1,128,128,128,1], 5000 Adam epochs ===")
key, init_key_of = jax.random.split(key)

params_overfit, _ = train_adam(
    layer_sizes=[1, 128, 128, 128, 1],
    n_epochs=5000,
    lr=0.005,
    init_key=init_key_of,
    save_at=set(),
    print_every=1000,
)

overfit_save = [(np.array(W), np.array(b)) for W, b in params_overfit]
path_of = os.path.join(ASSETS_DIR, "nb0_overfit_params.pkl")
with open(path_of, "wb") as f:
    pickle.dump(overfit_save, f)
print(f"  Saved nb0_overfit_params.pkl  ({os.path.getsize(path_of)/1024:.1f} KB)")

xi_chk = jnp.array(x_norm, dtype=jnp.float32).reshape(-1, 1)
yi_chk = jnp.array(y_noisy_np, dtype=jnp.float32)
overfit_loss = float(mse_loss(params_overfit, xi_chk, yi_chk))
y_pred_overfit = predict_np(params_overfit, xg_norm)
print(f"  Overfit train loss: {overfit_loss:.5f}  "
      f"(noise floor={SIGMA**2:.4f}, ratio={overfit_loss/SIGMA**2:.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Trophy figure  — 4 panels, clearly distinct stages
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Rendering trophy figure ===")

TROPHY_FNAMES = [
    ("nb0_epoch0_params.pkl",   "Época 0"),
    ("nb0_epoch200_params.pkl", "Época 200"),
    ("nb0_epoch500_params.pkl", "Época 500"),
    ("nb0_fcnn_params.pkl",     "Época 1000"),
]

C_DATA  = "#aaaaaa"
C_MODEL = "#e74c3c"
C_TRUE  = "#2980b9"

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
fig.suptitle("Progressão do treino — senoide amortecida", fontsize=14,
             fontweight="bold")

for ax, (fname, title) in zip(axes, TROPHY_FNAMES):
    path = os.path.join(ASSETS_DIR, fname)
    with open(path, "rb") as f:
        p = pickle.load(f)
    y_pred = predict_np(p, xg_norm)
    loss_val = float(mse_loss(p, xi_chk, yi_chk))

    ax.scatter(x_np, y_noisy_np, s=8, alpha=0.45, color=C_DATA,
               label="dados ruidosos", zorder=2)
    ax.plot(x_grid, y_grid, "--", lw=1.5, color=C_TRUE,
            label="função verdadeira", zorder=3)
    ax.plot(x_grid, y_pred, "-",  lw=2.0, color=C_MODEL,
            label="modelo", zorder=4)

    ax.set_title(f"{title}\n(perda = {loss_val:.3f})", fontsize=11)
    ax.set_xlabel("x", fontsize=11)
    if ax is axes[0]:
        ax.set_ylabel("y", fontsize=11)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, alpha=0.3)

axes[-1].legend(loc="upper right", fontsize=9)
plt.tight_layout()

trophy_path = os.path.join(ASSETS_DIR, "nb0_fig_trophy.png")
fig.savefig(trophy_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved nb0_fig_trophy.png  ({os.path.getsize(trophy_path)/1024:.1f} KB)")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Overfit comparison figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Rendering overfit comparison figure ===")

# Load final normal model
path_good = os.path.join(ASSETS_DIR, "nb0_fcnn_params.pkl")
with open(path_good, "rb") as f:
    p_good = pickle.load(f)
y_pred_good = predict_np(p_good, xg_norm)
good_loss = float(mse_loss(p_good, xi_chk, yi_chk))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
fig.suptitle("Generalização vs. Sobreajuste (overfitting)",
             fontsize=14, fontweight="bold")

# Left — good fit
ax1.scatter(x_np, y_noisy_np, s=8, alpha=0.4, color=C_DATA,
            label="dados ruidosos")
ax1.plot(x_grid, y_grid, "--", lw=1.5, color=C_TRUE,
         label="função verdadeira")
ax1.plot(x_grid, y_pred_good, "-", lw=2, color=C_MODEL,
         label="modelo [1,32,32,1]")
ax1.set_title(f"Rede pequena — generaliza bem\n"
              f"perda de treino = {good_loss:.4f}", fontsize=11)
ax1.set_xlabel("x", fontsize=11)
ax1.set_ylabel("y", fontsize=11)
ax1.set_xlim(0, X_MAX)
ax1.set_ylim(-2.2, 2.2)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Right — overfit
ax2.scatter(x_np, y_noisy_np, s=8, alpha=0.4, color=C_DATA,
            label="dados ruidosos")
ax2.plot(x_grid, y_grid, "--", lw=1.5, color=C_TRUE,
         label="função verdadeira")
ax2.plot(x_grid, y_pred_overfit, "-", lw=2, color="#e67e22",
         label="modelo [1,128,128,128,1]")
ax2.set_title(f"Rede grande — sobreajuste!\n"
              f"perda de treino = {overfit_loss:.4f}  "
              f"(< ruído σ²={SIGMA**2:.4f})", fontsize=11)
ax2.set_xlabel("x", fontsize=11)
ax2.set_xlim(0, X_MAX)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
ov_path = os.path.join(ASSETS_DIR, "nb0_fig_overfit.png")
fig.savefig(ov_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved nb0_fig_overfit.png  ({os.path.getsize(ov_path)/1024:.1f} KB)")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Summary
# ══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t0
print(f"\n=== Asset summary  (total time: {elapsed:.1f}s) ===")
asset_files = [
    "nb0_epoch0_params.pkl",
    "nb0_epoch200_params.pkl",
    "nb0_epoch500_params.pkl",
    "nb0_fcnn_params.pkl",
    "nb0_overfit_params.pkl",
    "nb0_fig_trophy.png",
    "nb0_fig_overfit.png",
]
total_kb = 0.0
for fname in asset_files:
    path = os.path.join(ASSETS_DIR, fname)
    kb   = os.path.getsize(path) / 1024
    total_kb += kb
    print(f"  {fname:42s}  {kb:7.1f} KB")
print(f"\n  Total: {total_kb:.1f} KB  ({total_kb/1024:.2f} MB)")
print("\nBUILD ASSETS: ALL GENERATED SUCCESSFULLY")
