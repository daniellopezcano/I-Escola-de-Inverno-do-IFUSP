#!/usr/bin/env python
"""
make_assets_00_caixa_de_ferramentas.py
---------------------------------------
Pre-warms the checkpoint cache for notebook 00_caixa_de_ferramentas.ipynb
by training with Adam (fast, high-quality fits) and saving the results to
the gitignored jax-examples/assets/ directory.

This script is OPTIONAL — the notebook self-generates everything at runtime
(using the "generate if absent" pattern) if this script has not been run first.
Running it beforehand simply gives Adam-quality checkpoints instead of the
fallback SGD-quality ones the notebook generates on first run.

Outputs (all in jax-examples/assets/, gitignored):
  nb0_epoch0_params.pkl          - initial random model (before any training)
  nb0_epoch200_params.pkl        - partial fit (Adam epoch 200)
  nb0_epoch500_params.pkl        - improving fit (Adam epoch 500)
  nb0_fcnn_params.pkl            - final normal model (Adam epoch 1000)
  nb0_overfit_params.pkl         - over-wide network (Adam epoch 5000)

Note: PNG figures are NOT generated here — the notebook always renders figures
live from loaded/computed model weights (my_feedback_v2 §5).

Run (optional, from repo root):
  /home/dlopez/miniconda3/envs/WinterSchool/bin/python \
    jax-examples/utils/make_assets_00_caixa_de_ferramentas.py
"""

import os
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
x_np  = np.sort(np.array(x_raw))
noise = np.array(jax.random.normal(k_noise, (N_POINTS,))) * SIGMA


def target_fn(x):
    return A * np.sin(2.0 * np.pi * x / LAM) * np.exp(-x / TAU)


y_noisy_np = target_fn(x_np) + noise


def normalise_x(x):
    return 2.0 * x / X_MAX - 1.0


x_norm = normalise_x(x_np)

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
    """Forward pass: tanh hidden layers, linear output."""
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def mse_loss(params, x_batch, y_batch):
    """Mean squared error."""
    y_pred = forward(params, x_batch).squeeze(-1)
    return jnp.mean((y_pred - y_batch) ** 2)


# ── Adam helper ────────────────────────────────────────────────────────────────
def adam_init(params):
    m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    v = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    return m, v


def adam_step(params, grads, m, v, t,
              lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    new_m = [(b1*mW + (1-b1)*gW, b1*mb + (1-b1)*gb)
             for (mW, mb), (gW, gb) in zip(m, grads)]
    new_v = [(b2*vW + (1-b2)*gW**2, b2*vb + (1-b2)*gb**2)
             for (vW, vb), (gW, gb) in zip(v, grads)]
    mh = [(mW/(1-b1**t), mb/(1-b1**t)) for mW, mb in new_m]
    vh = [(vW/(1-b2**t), vb/(1-b2**t)) for vW, vb in new_v]
    new_p = [(W - lr*mWh/(jnp.sqrt(vWh)+eps),
              b - lr*mbh/(jnp.sqrt(vbh)+eps))
             for (W, b), (mWh, mbh), (vWh, vbh)
             in zip(params, mh, vh)]
    return new_p, new_m, new_v


def save_pkl(params, fname):
    path = os.path.join(ASSETS_DIR, fname)
    params_np = [(np.array(W), np.array(b)) for W, b in params]
    with open(path, "wb") as f:
        pickle.dump(params_np, f)
    print(f"  Saved {fname}  ({os.path.getsize(path)/1024:.1f} KB)")
    return path


xi = jnp.array(x_norm, dtype=jnp.float32).reshape(-1, 1)
yi = jnp.array(y_noisy_np, dtype=jnp.float32)
grad_fn = jax.jit(jax.grad(mse_loss))

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Normal network  [1, 32, 32, 1] — 1000 Adam epochs
#     Trophy checkpoints at {0, 200, 500, 1000}
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Training normal network [1,32,32,1], 1000 Adam epochs ===")
key, init_key = jax.random.split(key)

params_normal = init_params([1, 32, 32, 1], init_key)
save_pkl(params_normal, "nb0_epoch0_params.pkl")   # epoch 0 (before training)

m_state, v_state = adam_init(params_normal)

for t in range(1, 1001):
    grads = grad_fn(params_normal, xi, yi)
    params_normal, m_state, v_state = adam_step(
        params_normal, grads, m_state, v_state, t, lr=0.01)

    if t == 200:
        save_pkl(params_normal, "nb0_epoch200_params.pkl")
    if t == 500:
        save_pkl(params_normal, "nb0_epoch500_params.pkl")
    if t % 200 == 0:
        l = float(mse_loss(params_normal, xi, yi))
        print(f"  epoch {t:5d}  loss={l:.6f}")

save_pkl(params_normal, "nb0_fcnn_params.pkl")     # epoch 1000 (final)
final_loss = float(mse_loss(params_normal, xi, yi))
print(f"  Normal net final loss: {final_loss:.6f}  "
      f"[noise floor ≈ {SIGMA**2:.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Overfit network  [1, 128, 128, 128, 1] — 5000 Adam epochs
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Training overfit network [1,128,128,128,1], 5000 Adam epochs ===")
key, init_key_of = jax.random.split(key)

params_overfit = init_params([1, 128, 128, 128, 1], init_key_of)
m_of, v_of = adam_init(params_overfit)

for t in range(1, 5001):
    grads = grad_fn(params_overfit, xi, yi)
    params_overfit, m_of, v_of = adam_step(
        params_overfit, grads, m_of, v_of, t, lr=0.005)
    if t % 1000 == 0:
        l = float(mse_loss(params_overfit, xi, yi))
        print(f"  epoch {t:5d}  loss={l:.6f}")

save_pkl(params_overfit, "nb0_overfit_params.pkl")
overfit_loss = float(mse_loss(params_overfit, xi, yi))
print(f"  Overfit net final loss: {overfit_loss:.5f}  "
      f"(noise floor={SIGMA**2:.4f}, ratio={overfit_loss/SIGMA**2:.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Summary
# ══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t0
print(f"\n=== Asset summary  (total time: {elapsed:.1f}s) ===")
asset_files = [
    "nb0_epoch0_params.pkl",
    "nb0_epoch200_params.pkl",
    "nb0_epoch500_params.pkl",
    "nb0_fcnn_params.pkl",
    "nb0_overfit_params.pkl",
]
total_kb = 0.0
for fname in asset_files:
    path = os.path.join(ASSETS_DIR, fname)
    kb   = os.path.getsize(path) / 1024
    total_kb += kb
    print(f"  {fname:42s}  {kb:7.1f} KB")
print(f"\n  Total: {total_kb:.1f} KB  ({total_kb/1024:.2f} MB)")
print("\nBUILD ASSETS: ALL GENERATED SUCCESSFULLY")
