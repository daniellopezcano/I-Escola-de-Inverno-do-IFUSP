#!/usr/bin/env python
"""
make_assets_01_domain_shift_toy.py
-----------------------------------
Generates all cached checkpoints and static PNG fallbacks for
notebook 01_domain_shift_toy.ipynb.

Outputs (all in jax-examples/assets/):
  toy_2d_4class.npz          – 2-D Gaussian mixture, source + target
  nb1_encoder_source.pkl     – encoder trained on source domain
  nb1_head_source.pkl        – head trained on source domain
  nb1_encoder_targetonly.pkl – encoder trained target-only (K=50)
  nb1_head_targetonly.pkl    – head trained target-only (K=50)
  nb1_encoder_ssda.pkl       – encoder after SSDA adaptation (K=50)
  nb1_head_ssda.pkl          – head (unchanged) after SSDA
  nb1_ksweep.npz             – precomputed macro-F1 for K-sweep
  nb1_fig_gmm.png
  nb1_fig_decision_source.png
  nb1_fig_decision_target.png
  nb1_fig_latent_shift.png
  nb1_fig_comparison.png
  nb1_fig_k_sweep.png
  nb1_fig_latent_ssda.png

Run:
  /home/dlopez/miniconda3/envs/WinterSchool/bin/python \
    jax-examples/utils/make_assets_01_domain_shift_toy.py
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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score

t_start = time.time()

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
MASTER_KEY = jax.random.PRNGKey(42)

# ── Configuration (must match notebook exactly) ────────────────────────────────
N_CLASSES      = 4
N_PER_CLASS    = 300          # source samples per class
CLASS_WEIGHTS  = np.array([0.50, 0.25, 0.15, 0.10], dtype=np.float32)

# Source cluster centres
MEANS_SOURCE = np.array([
    [-3.0, -3.0],
    [ 3.0, -3.0],
    [-3.0,  3.0],
    [ 3.0,  3.0],
], dtype=np.float32)

# Target: two classes shifted (classes 1 and 3)
MEANS_TARGET = np.array([
    [-3.0, -3.0],   # class 0 — same
    [ 3.0,  1.5],   # class 1 — shifted +4.5 in y  (≈2.5 std)
    [-3.0,  3.0],   # class 2 — same
    [ 0.5,  3.0],   # class 3 — shifted -2.5 in x  (≈2.5 std)
], dtype=np.float32)

COV_SCALE = 1.0   # isotropic covariance, σ = 1.0

# ── Colour palette ─────────────────────────────────────────────────────────────
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
CLASS_LABELS = ["Classe 0", "Classe 1", "Classe 2", "Classe 3"]


# ══════════════════════════════════════════════════════════════════════════════
# Data generation helpers
# ══════════════════════════════════════════════════════════════════════════════

def generate_gmm_data(n_per_class, means, cov_scale, key):
    """Generate isotropic Gaussian mixture. Returns (X, y) numpy arrays."""
    X_list, y_list = [], []
    for c in range(N_CLASSES):
        key, k = jax.random.split(key)
        pts = jax.random.normal(k, (n_per_class, 2)) * cov_scale + means[c]
        X_list.append(np.array(pts))
        y_list.append(np.full(n_per_class, c, dtype=np.int32))
    return np.concatenate(X_list), np.concatenate(y_list)


def sample_k_labels(X, y, K, key, n_classes=N_CLASSES):
    """Stratified K-shot sampling: K total, balanced across classes."""
    k_per_class = max(1, K // n_classes)
    idx_list = []
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        key, k = jax.random.split(key)
        chosen = jax.random.permutation(k, len(idx_c))[:k_per_class]
        idx_list.append(idx_c[np.array(chosen)])
    idx = np.concatenate(idx_list)
    return X[idx], y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# MLP helpers — params as list of (W, b) tuples  (same style as NB0)
# ══════════════════════════════════════════════════════════════════════════════

def init_mlp(layer_sizes, key):
    """Glorot uniform init. Returns list of (W, b)."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, kw = jax.random.split(key)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        W = jax.random.uniform(kw, (fan_in, fan_out), minval=-scale, maxval=scale)
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def forward_tanh(params, x):
    """tanh hidden layers, linear output."""
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def forward_encoder(enc_params, x):
    return forward_tanh(enc_params, x)   # [2→32→32→2] with tanh


def forward_head(head_params, z):
    """Linear head: [2→4]."""
    W, b = head_params[0]
    return z @ W + b


def predict_classes(enc_params, head_params, X):
    """Predict class indices for numpy array X."""
    z    = forward_encoder(enc_params, jnp.array(X, dtype=jnp.float32))
    logits = forward_head(head_params, z)
    return np.array(jnp.argmax(logits, axis=-1))


def get_embeddings(enc_params, X):
    """Return 2-D latent embeddings as numpy array."""
    z = forward_encoder(enc_params, jnp.array(X, dtype=jnp.float32))
    return np.array(z)


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def weighted_ce_loss(enc_params, head_params, X, y, cw):
    """
    Weighted cross-entropy.
    cw: class_weights array shape (n_classes,)
    """
    z       = forward_encoder(enc_params, X)
    logits  = forward_head(head_params, z)
    log_p   = jax.nn.log_softmax(logits, axis=-1)         # (N, C)
    # gather log-prob of the true class
    N       = y.shape[0]
    log_py  = log_p[jnp.arange(N), y]                    # (N,)
    # per-sample weight = class_weight[y_i]
    weights = cw[y]                                        # (N,)
    return -jnp.mean(weights * log_py)


def ce_loss_only_enc(enc_params, head_params_frozen, X, y, cw):
    """CE loss that only takes enc_params as the differentiable argument."""
    return weighted_ce_loss(enc_params, head_params_frozen, X, y, cw)


# ══════════════════════════════════════════════════════════════════════════════
# Adam optimiser (manual, matching NB0 style — no optax)
# ══════════════════════════════════════════════════════════════════════════════

def _adam_init(params):
    m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    v = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    return m, v


def _adam_step(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    new_m = [(b1*mW + (1-b1)*gW, b1*mb + (1-b1)*gb)
             for (mW, mb), (gW, gb) in zip(m, grads)]
    new_v = [(b2*vW + (1-b2)*gW**2, b2*vb + (1-b2)*gb**2)
             for (vW, vb), (gW, gb) in zip(v, grads)]
    mhat  = [(mW/(1-b1**t), mb/(1-b1**t)) for mW, mb in new_m]
    vhat  = [(vW/(1-b2**t), vb/(1-b2**t)) for vW, vb in new_v]
    new_p = [(W - lr*mWh/(jnp.sqrt(vWh)+eps),
              b - lr*mbh/(jnp.sqrt(vbh)+eps))
             for (W, b), (mWh, mbh), (vWh, vbh) in zip(params, mhat, vhat)]
    return new_p, new_m, new_v


# ══════════════════════════════════════════════════════════════════════════════
# Training routines
# ══════════════════════════════════════════════════════════════════════════════

def train_source(X_src, y_src, key, n_epochs=600, lr=3e-3, print_every=200):
    """Train encoder + head on source domain with weighted CE."""
    cw_jnp = jnp.array(CLASS_WEIGHTS)
    key, ke, kh = jax.random.split(key, 3)
    enc  = init_mlp([2, 32, 32, 2], ke)
    head = init_mlp([2, N_CLASSES], kh)

    Xj = jnp.array(X_src, dtype=jnp.float32)
    yj = jnp.array(y_src, dtype=jnp.int32)

    grad_fn = jax.jit(jax.grad(weighted_ce_loss, argnums=(0, 1)))

    m_enc,  v_enc  = _adam_init(enc)
    m_head, v_head = _adam_init(head)

    for t in range(1, n_epochs + 1):
        g_enc, g_head = grad_fn(enc, head, Xj, yj, cw_jnp)
        enc,  m_enc,  v_enc  = _adam_step(enc,  g_enc,  m_enc,  v_enc,  t, lr)
        head, m_head, v_head = _adam_step(head, g_head, m_head, v_head, t, lr)
        if print_every and t % print_every == 0:
            l = float(weighted_ce_loss(enc, head, Xj, yj, cw_jnp))
            print(f"  [source] epoch {t:4d}  loss={l:.4f}")

    return enc, head


def train_from_scratch(X_lbl, y_lbl, key, n_epochs=600, lr=3e-3):
    """Train encoder + head from scratch (target-only)."""
    cw_jnp = jnp.array(CLASS_WEIGHTS)
    key, ke, kh = jax.random.split(key, 3)
    enc  = init_mlp([2, 32, 32, 2], ke)
    head = init_mlp([2, N_CLASSES], kh)

    Xj = jnp.array(X_lbl, dtype=jnp.float32)
    yj = jnp.array(y_lbl, dtype=jnp.int32)

    grad_fn = jax.jit(jax.grad(weighted_ce_loss, argnums=(0, 1)))
    m_enc,  v_enc  = _adam_init(enc)
    m_head, v_head = _adam_init(head)

    for t in range(1, n_epochs + 1):
        g_enc, g_head = grad_fn(enc, head, Xj, yj, cw_jnp)
        enc,  m_enc,  v_enc  = _adam_step(enc,  g_enc,  m_enc,  v_enc,  t, lr)
        head, m_head, v_head = _adam_step(head, g_head, m_head, v_head, t, lr)

    return enc, head


def train_ssda(enc_source, head_source, X_lbl, y_lbl, key,
               n_epochs=600, lr=3e-3):
    """SSDA: freeze head, adapt encoder only."""
    cw_jnp = jnp.array(CLASS_WEIGHTS)
    enc  = enc_source          # start from source encoder
    head = head_source         # frozen — never updated

    Xj = jnp.array(X_lbl, dtype=jnp.float32)
    yj = jnp.array(y_lbl, dtype=jnp.int32)

    # gradient only w.r.t. enc_params (argnums=0)
    grad_fn = jax.jit(jax.grad(ce_loss_only_enc, argnums=0))
    m_enc, v_enc = _adam_init(enc)

    for t in range(1, n_epochs + 1):
        g_enc = grad_fn(enc, head, Xj, yj, cw_jnp)
        enc, m_enc, v_enc = _adam_step(enc, g_enc, m_enc, v_enc, t, lr)

    return enc, head


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_mesh(xlim=(-7, 7), ylim=(-7, 7), step=0.12):
    xx, yy = np.meshgrid(
        np.arange(xlim[0], xlim[1], step),
        np.arange(ylim[0], ylim[1], step),
    )
    return xx, yy


def decision_map(enc_params, head_params, xx, yy):
    pts    = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    preds  = predict_classes(enc_params, head_params, pts)
    return preds.reshape(xx.shape)


def plot_decision_map_ax(ax, enc_params, head_params, X_overlay, y_overlay,
                         title, xx, yy, alpha_mesh=0.35):
    Z = decision_map(enc_params, head_params, xx, yy)
    cmap_bg = matplotlib.colors.ListedColormap(
        [matplotlib.colors.to_rgba(c, alpha_mesh) for c in CLASS_COLORS])
    ax.pcolormesh(xx, yy, Z, cmap=cmap_bg, shading="auto",
                  vmin=0, vmax=N_CLASSES - 1)
    for c in range(N_CLASSES):
        mask = y_overlay == c
        ax.scatter(X_overlay[mask, 0], X_overlay[mask, 1],
                   s=15, color=CLASS_COLORS[c], edgecolors="k",
                   linewidths=0.3, zorder=3, label=CLASS_LABELS[c])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


# ══════════════════════════════════════════════════════════════════════════════
# 1. Generate and save data
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. Generating 2-D Gaussian mixture data ===")

key = MASTER_KEY
key, ks, kt = jax.random.split(key, 3)

X_src, y_src = generate_gmm_data(N_PER_CLASS, MEANS_SOURCE, COV_SCALE, ks)
X_tgt, y_tgt = generate_gmm_data(N_PER_CLASS, MEANS_TARGET, COV_SCALE, kt)

data_path = os.path.join(ASSETS_DIR, "toy_2d_4class.npz")
np.savez(data_path,
         X_source=X_src, y_source=y_src,
         X_target=X_tgt, y_target=y_tgt,
         means_source=MEANS_SOURCE, means_target=MEANS_TARGET)
print(f"  Saved toy_2d_4class.npz  ({os.path.getsize(data_path)/1024:.1f} KB)")
print(f"  Source: {X_src.shape}  Target: {X_tgt.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Source model training
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Training source model ===")
key, ktrain = jax.random.split(key)
enc_src, head_src = train_source(X_src, y_src, ktrain,
                                 n_epochs=600, lr=3e-3, print_every=200)

def save_pkl(obj, fname):
    path = os.path.join(ASSETS_DIR, fname)
    with open(path, "wb") as f:
        pickle.dump([(np.array(W), np.array(b)) for W, b in obj], f)
    print(f"  Saved {fname}  ({os.path.getsize(path)/1024:.1f} KB)")

save_pkl(enc_src,  "nb1_encoder_source.pkl")
save_pkl(head_src, "nb1_head_source.pkl")

acc_src  = np.mean(predict_classes(enc_src, head_src, X_src) == y_src)
acc_tgt0 = np.mean(predict_classes(enc_src, head_src, X_tgt) == y_tgt)
print(f"  Source acc (source): {acc_src:.3f}   Source acc (target): {acc_tgt0:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Target-only model (K=50)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Training target-only model (K=50) ===")
key, kk = jax.random.split(key)
X_k50, y_k50 = sample_k_labels(X_tgt, y_tgt, 50, kk)
key, ktrain2 = jax.random.split(key)
enc_to, head_to = train_from_scratch(X_k50, y_k50, ktrain2, n_epochs=600, lr=3e-3)
save_pkl(enc_to,  "nb1_encoder_targetonly.pkl")
save_pkl(head_to, "nb1_head_targetonly.pkl")
acc_to = np.mean(predict_classes(enc_to, head_to, X_tgt) == y_tgt)
print(f"  Target-only acc (target): {acc_to:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SSDA model (K=50, same labelled set)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. SSDA adaptation (K=50) ===")
key, ktrain3 = jax.random.split(key)
enc_ssda, head_ssda = train_ssda(enc_src, head_src, X_k50, y_k50, ktrain3,
                                 n_epochs=600, lr=3e-3)
save_pkl(enc_ssda,  "nb1_encoder_ssda.pkl")
save_pkl(head_ssda, "nb1_head_ssda.pkl")
acc_ssda = np.mean(predict_classes(enc_ssda, head_ssda, X_tgt) == y_tgt)
print(f"  SSDA acc (target): {acc_ssda:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. K-sweep  K ∈ {10, 25, 50, 100, 200}
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. K-sweep ===")
K_VALUES  = [10, 25, 50, 100, 200]
N_REPEATS = 3   # average over random seeds

f1_zeroshot = []
f1_targetonly = []
f1_ssda = []

for K in K_VALUES:
    f1_to_list, f1_ss_list = [], []
    for rep in range(N_REPEATS):
        key, kk2, kt2, ks2 = jax.random.split(key, 4)
        Xk, yk = sample_k_labels(X_tgt, y_tgt, K, kk2)

        # Regime B: target-only from scratch
        e_to, h_to = train_from_scratch(Xk, yk, kt2, n_epochs=400, lr=3e-3)
        preds_to   = predict_classes(e_to, h_to, X_tgt)
        f1_to_list.append(f1_score(y_tgt, preds_to, average="macro", zero_division=0))

        # Regime C: SSDA
        e_ss, h_ss = train_ssda(enc_src, head_src, Xk, yk, ks2,
                                 n_epochs=400, lr=3e-3)
        preds_ss   = predict_classes(e_ss, h_ss, X_tgt)
        f1_ss_list.append(f1_score(y_tgt, preds_ss, average="macro", zero_division=0))

    # Regime A: zero-shot (constant across K)
    preds_zs = predict_classes(enc_src, head_src, X_tgt)
    f1_zs    = f1_score(y_tgt, preds_zs, average="macro", zero_division=0)
    f1_zeroshot.append(f1_zs)
    f1_targetonly.append(np.mean(f1_to_list))
    f1_ssda.append(np.mean(f1_ss_list))
    print(f"  K={K:3d}  zero-shot={f1_zs:.3f}  "
          f"target-only={f1_targetonly[-1]:.3f}  ssda={f1_ssda[-1]:.3f}")

ksweep_path = os.path.join(ASSETS_DIR, "nb1_ksweep.npz")
np.savez(ksweep_path,
         K_values=np.array(K_VALUES),
         f1_zeroshot=np.array(f1_zeroshot),
         f1_targetonly=np.array(f1_targetonly),
         f1_ssda=np.array(f1_ssda))
print(f"  Saved nb1_ksweep.npz  ({os.path.getsize(ksweep_path)/1024:.1f} KB)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Static PNG figures
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. Rendering figures ===")

# ── Fig 1: GMM data ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Universo de brinquedo: domínio fonte vs. domínio alvo", fontsize=13)

for c in range(N_CLASSES):
    ax1.scatter(X_src[y_src==c, 0], X_src[y_src==c, 1],
                s=18, color=CLASS_COLORS[c], alpha=0.65,
                edgecolors="none", label=CLASS_LABELS[c])
ax1.set_title("Fonte (Source)", fontsize=11)
ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

for c in range(N_CLASSES):
    ax2.scatter(X_tgt[y_tgt==c, 0], X_tgt[y_tgt==c, 1],
                s=18, color=CLASS_COLORS[c], alpha=0.65,
                edgecolors="none", label=CLASS_LABELS[c])
ax2.set_title("Alvo (Target)", fontsize=11)
ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb1_fig_gmm.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb1_fig_gmm.png  ({os.path.getsize(p)/1024:.1f} KB)")

# ── Fig 2 & 3: decision maps ───────────────────────────────────────────────────
xx, yy = make_mesh()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
plot_decision_map_ax(ax, enc_src, head_src, X_src, y_src,
                     "Mapa de Decisão — Fonte (modelo treinado na fonte)", xx, yy)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb1_fig_decision_source.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb1_fig_decision_source.png  ({os.path.getsize(p)/1024:.1f} KB)")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
plot_decision_map_ax(ax, enc_src, head_src, X_tgt, y_tgt,
                     "Mapa de Decisão — Alvo (falha catastrófica)", xx, yy)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb1_fig_decision_target.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb1_fig_decision_target.png  ({os.path.getsize(p)/1024:.1f} KB)")

# ── Fig 4: latent scatter source vs target ─────────────────────────────────────
Z_src = get_embeddings(enc_src, X_src)
Z_tgt = get_embeddings(enc_src, X_tgt)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Espaço Latente do Encoder — Shift visível", fontsize=13)

for c in range(N_CLASSES):
    ax1.scatter(Z_src[y_src==c, 0], Z_src[y_src==c, 1],
                s=18, color=CLASS_COLORS[c], alpha=0.65,
                edgecolors="none", label=CLASS_LABELS[c])
ax1.set_title("Fonte no espaço latente"); ax1.set_xlabel("z₁"); ax1.set_ylabel("z₂")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

for c in range(N_CLASSES):
    ax2.scatter(Z_tgt[y_tgt==c, 0], Z_tgt[y_tgt==c, 1],
                s=18, color=CLASS_COLORS[c], alpha=0.65,
                edgecolors="none", label=CLASS_LABELS[c])
ax2.set_title("Alvo no espaço latente (shift!)"); ax2.set_xlabel("z₁"); ax2.set_ylabel("z₂")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb1_fig_latent_shift.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb1_fig_latent_shift.png  ({os.path.getsize(p)/1024:.1f} KB)")

# ── Fig 5: 3-panel comparison A/B/C ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Comparação dos 3 Regimes — Alvo sobreposto", fontsize=13)

labels_abc = [
    ("(A) Zero-shot (fonte → alvo)", enc_src,  head_src),
    ("(B) Somente alvo (K=50)",      enc_to,   head_to),
    ("(C) SSDA (K=50)",              enc_ssda, head_ssda),
]
for ax, (title, enc_, head_) in zip(axes, labels_abc):
    plot_decision_map_ax(ax, enc_, head_, X_tgt, y_tgt, title, xx, yy)
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb1_fig_comparison.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb1_fig_comparison.png  ({os.path.getsize(p)/1024:.1f} KB)")

# ── Fig 6: K-sweep curves ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(K_VALUES, f1_zeroshot,   "o--", color="#7f8c8d", lw=2,
        label="(A) Zero-shot")
ax.plot(K_VALUES, f1_targetonly, "s-",  color="#3498db", lw=2,
        label="(B) Somente alvo")
ax.plot(K_VALUES, f1_ssda,       "^-",  color="#e74c3c", lw=2,
        label="(C) SSDA")
ax.set_xlabel("K (rótulos do alvo)"); ax.set_ylabel("Macro-F1")
ax.set_title("Macro-F1 × K — os 3 regimes de adaptação")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_xticks(K_VALUES)
plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb1_fig_k_sweep.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb1_fig_k_sweep.png  ({os.path.getsize(p)/1024:.1f} KB)")

# ── Fig 7: latent before vs after SSDA ────────────────────────────────────────
Z_tgt_before = get_embeddings(enc_src,  X_tgt)
Z_tgt_after  = get_embeddings(enc_ssda, X_tgt)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Espaço Latente: antes vs. após SSDA", fontsize=13)

for c in range(N_CLASSES):
    ax1.scatter(Z_tgt_before[y_tgt==c, 0], Z_tgt_before[y_tgt==c, 1],
                s=18, color=CLASS_COLORS[c], alpha=0.65, edgecolors="none",
                label=CLASS_LABELS[c])
ax1.set_title("Antes da adaptação (latente alvo)"); ax1.set_xlabel("z₁"); ax1.set_ylabel("z₂")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

for c in range(N_CLASSES):
    ax2.scatter(Z_tgt_after[y_tgt==c, 0], Z_tgt_after[y_tgt==c, 1],
                s=18, color=CLASS_COLORS[c], alpha=0.65, edgecolors="none",
                label=CLASS_LABELS[c])
ax2.set_title("Após SSDA (alvo entrou nas regiões da cabeça)"); ax2.set_xlabel("z₁"); ax2.set_ylabel("z₂")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb1_fig_latent_ssda.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb1_fig_latent_ssda.png  ({os.path.getsize(p)/1024:.1f} KB)")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Summary
# ══════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print(f"\n=== Asset summary  (total time: {elapsed:.1f}s) ===")
asset_files = [
    "toy_2d_4class.npz",
    "nb1_encoder_source.pkl",  "nb1_head_source.pkl",
    "nb1_encoder_targetonly.pkl", "nb1_head_targetonly.pkl",
    "nb1_encoder_ssda.pkl",    "nb1_head_ssda.pkl",
    "nb1_ksweep.npz",
    "nb1_fig_gmm.png",
    "nb1_fig_decision_source.png", "nb1_fig_decision_target.png",
    "nb1_fig_latent_shift.png",
    "nb1_fig_comparison.png",
    "nb1_fig_k_sweep.png",
    "nb1_fig_latent_ssda.png",
]
total_kb = 0.0
for fname in asset_files:
    path = os.path.join(ASSETS_DIR, fname)
    kb   = os.path.getsize(path) / 1024
    total_kb += kb
    print(f"  {fname:45s}  {kb:7.1f} KB")
print(f"\n  Total: {total_kb:.1f} KB  ({total_kb/1024:.2f} MB)")
print("\nBUILD ASSETS: ALL GENERATED SUCCESSFULLY")
