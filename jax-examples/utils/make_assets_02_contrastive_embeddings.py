#!/usr/bin/env python
"""
make_assets_02_contrastive_embeddings.py
-----------------------------------------
Generates all cached datasets, checkpoints, and static PNG fallbacks for
notebook 02_contrastive_embeddings.ipynb.

Outputs (all in jax-examples/assets/):
  mnist_4k.npz                  – MNIST subset 4000 train + 1000 test
  nb2_sandbox_initial.npz       – initial particle positions (5 groups x 40)
  nb2_sandbox_collapsed.npz     – positions after delta_push=0 relaxation
  nb2_sandbox_final.npz         – positions after delta_push>0 relaxation
  nb2_encoder_epoch0.pkl        – 2D encoder at epoch 0 (random init)
  nb2_encoder_early.pkl         – 2D encoder at epoch ~20
  nb2_encoder_late.pkl          – 2D encoder at epoch ~250
  nb2_encoder_16d_late.pkl      – 16D encoder at epoch ~250
  nb2_fig_sandbox_final.png     – filmstrip: initial / collapsed / separated
  nb2_fig_evolution.png         – filmstrip: epoch0 / early / late embedding
  nb2_fig_tsne.png              – t-SNE at perplexities 5, 30, 100
  nb2_fig_umap.png              – UMAP placeholder (umap-learn not installed)

Run:
  /home/dlopez/miniconda3/envs/WinterSchool/bin/python \
    jax-examples/utils/make_assets_02_contrastive_embeddings.py
"""

import os
import gzip
import struct
import pickle
import time
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

t_start = time.time()

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Reproducibility ────────────────────────────────────────────────────────────
MASTER_KEY = jax.random.PRNGKey(42)
np.random.seed(42)

# ── Colour palettes ────────────────────────────────────────────────────────────
CORES_10 = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#c0392b", "#27ae60",
]
CORES_5 = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


# ══════════════════════════════════════════════════════════════════════════════
# Weinberger loss factory
# n_classes is captured as a Python int (closed-over), making the returned
# function fully JIT-compatible.
# ══════════════════════════════════════════════════════════════════════════════

def make_weinberger_loss(n_classes):
    """
    Factory that returns weinberger_loss(x, labels, delta_pull, delta_push,
    lambda_reg) with n_classes baked in as a Python constant.
    """
    _nc = int(n_classes)

    def _loss(x, labels, delta_pull, delta_push, lambda_reg):
        # Cluster centers via one-hot aggregation (JIT-safe: _nc is a literal)
        one_hot = jax.nn.one_hot(labels, _nc)             # (N, K)
        counts  = one_hot.sum(axis=0)                      # (K,)
        centers = (one_hot.T @ x) / jnp.maximum(counts[:, None], 1.0)

        # Pull: each point within delta_pull of its class center
        c_pp     = centers[labels]
        dist_pt  = jnp.sqrt(jnp.sum((x - c_pp)**2, axis=-1) + 1e-8)
        L_pull   = jnp.mean(jnp.maximum(dist_pt - delta_pull, 0.0)**2)

        # Push: every pair of distinct centers at least delta_push apart
        ci        = centers[:, None, :]
        cj        = centers[None, :, :]
        dist_cc   = jnp.sqrt(jnp.sum((ci - cj)**2, axis=-1) + 1e-8)
        push_viol = jnp.maximum(delta_push - dist_cc, 0.0)**2
        off_diag  = 1.0 - jnp.eye(_nc)
        L_push    = (jnp.sum(off_diag * push_viol)
                     / jnp.maximum(off_diag.sum(), 1.0))

        # Regularisation: centers near origin
        L_reg = jnp.mean(jnp.sum(centers**2, axis=-1))

        return L_pull + L_push + lambda_reg * L_reg

    return _loss


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Download and cache MNIST
# ══════════════════════════════════════════════════════════════════════════════

def _parse_idx_gz(path, kind="images"):
    """Parse an MNIST IDX file (gzipped) from a local path."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if kind == "images":
        _, n, rows, cols = struct.unpack(">IIII", raw[:16])
        data = np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, rows * cols)
        return data.astype(np.float32) / 255.0
    else:
        _, n = struct.unpack(">II", raw[:8])
        return np.frombuffer(raw[8:], dtype=np.uint8).astype(np.int32)


def download_mnist(target_dir):
    url_base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    fnames   = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs(target_dir, exist_ok=True)
    paths = {}
    for key, fname in fnames.items():
        local = os.path.join(target_dir, fname)
        if not os.path.exists(local):
            print(f"  Downloading {fname} ...")
            urllib.request.urlretrieve(url_base + fname, local)
            print(f"  Done: {os.path.getsize(local)//1024} KB")
        else:
            print(f"  {fname} already cached")
        paths[key] = local
    return paths


def make_mnist_subset(paths, n_train=4000, n_test=1000, seed=42):
    """Balanced subsample: n_train/n_test evenly split across 10 classes."""
    rng      = np.random.default_rng(seed)
    X_tr_all = _parse_idx_gz(paths["train_images"], "images")
    y_tr_all = _parse_idx_gz(paths["train_labels"], "labels")
    X_te_all = _parse_idx_gz(paths["test_images"],  "images")
    y_te_all = _parse_idx_gz(paths["test_labels"],  "labels")
    n_cls    = 10
    ntr      = n_train // n_cls
    nte      = n_test  // n_cls
    tr_idx, te_idx = [], []
    for c in range(n_cls):
        idx = np.where(y_tr_all == c)[0]
        tr_idx.append(rng.choice(idx, ntr, replace=False))
        idx_te = np.where(y_te_all == c)[0]
        te_idx.append(rng.choice(idx_te, nte, replace=False))
    tr_idx = np.concatenate(tr_idx); rng.shuffle(tr_idx)
    te_idx = np.concatenate(te_idx); rng.shuffle(te_idx)
    return (X_tr_all[tr_idx], y_tr_all[tr_idx],
            X_te_all[te_idx], y_te_all[te_idx])


print("\n=== 1. Downloading and caching MNIST ===")
mnist_tmp  = os.path.join(ASSETS_DIR, "_mnist_raw")
raw_paths  = download_mnist(mnist_tmp)
X_train, y_train, X_test, y_test = make_mnist_subset(raw_paths)
mnist_path = os.path.join(ASSETS_DIR, "mnist_4k.npz")
np.savez(mnist_path, X_train=X_train, y_train=y_train,
         X_test=X_test, y_test=y_test)
print(f"  Saved mnist_4k.npz  ({os.path.getsize(mnist_path)//1024} KB)")
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Sandbox particle relaxation (Act 1)
# ══════════════════════════════════════════════════════════════════════════════
#
# Design principle:
#   - Groups start overlapping near origin BUT with a tiny ring-offset (0.4)
#     to break symmetry (otherwise push gradient is exactly zero at t=0).
#   - Collapse demo: delta_push=0, lambda_reg=2.0  -> strong pull to origin
#   - Separation demo: delta_push=1.5, lambda_reg=0.01 -> push spreads groups
#
# ══════════════════════════════════════════════════════════════════════════════

N_GROUPS       = 5
N_PER_GROUP    = 40
DELTA_PULL     = 0.5
DELTA_PUSH     = 1.5
LAMBDA_FINAL   = 0.01   # weak reg, push dominates -> separation
LAMBDA_COLLAPSE= 10.0   # strong reg, no push      -> collapse to origin
ETA            = 0.05
N_STEPS_COL    = 600
N_STEPS_SEP    = 1500

sandbox_loss   = make_weinberger_loss(N_GROUPS)
grad_sandbox   = jax.jit(jax.grad(sandbox_loss, argnums=0))


def generate_particles(n_groups, n_per_group, key):
    """
    Groups overlapping near origin with tiny ring-offset for symmetry breaking.
    Each group c is centred at 0.4*(cos, sin)(2pi*c/n_groups) with noise std=0.9.
    Groups visually overlap (noise >> offset) yet have distinct gradients.
    """
    angles   = jnp.linspace(0, 2 * jnp.pi, n_groups, endpoint=False)
    centers  = 0.4 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    pts_list = []
    for c in range(n_groups):
        key, k_c = jax.random.split(key)
        noise = jax.random.normal(k_c, (n_per_group, 2)) * 0.9
        pts_list.append(noise + centers[c])
    positions = jnp.concatenate(pts_list, axis=0)
    labels    = jnp.repeat(jnp.arange(n_groups), n_per_group)
    return np.array(positions, dtype=np.float32), np.array(labels, dtype=np.int32)


print("\n=== 2. Sandbox particle relaxation ===")
key = MASTER_KEY
key, kp = jax.random.split(key)
x_init_np, labels_np = generate_particles(N_GROUPS, N_PER_GROUP, kp)
lb_j = jnp.array(labels_np)
print(f"  Initial spread std: {x_init_np.std(axis=0)}")

# ── 2a. Collapse: delta_push=0, strong lambda_reg  ───────────────────────────
# Strong regularisation (2.0) forces all cluster centers to origin.
# Each group first tightens (L_pull), then drifts to origin (L_reg dominates).
print("  Running collapsed relaxation (delta_push=0, lambda_reg=2.0) ...")
x_col = jnp.array(x_init_np)
for _ in range(N_STEPS_COL):
    g = grad_sandbox(x_col, lb_j, DELTA_PULL, 0.0, LAMBDA_COLLAPSE)
    x_col = x_col - ETA * g
x_collapsed = np.array(x_col)

centers_col  = np.stack([x_collapsed[labels_np == c].mean(axis=0)
                          for c in range(N_GROUPS)])
dists_col    = [np.linalg.norm(centers_col[i] - centers_col[j])
                for i in range(N_GROUPS) for j in range(i+1, N_GROUPS)]
print(f"  Collapsed std:       {x_collapsed.std(axis=0)}")
print(f"  Collapsed max inter-center dist: {max(dists_col):.3f}")

# ── 2b. Separation: delta_push=1.5, weak lambda_reg  ─────────────────────────
# Push forces groups apart; weak reg anchors cluster slightly near origin.
print("  Running proper relaxation (delta_push=1.5, lambda_reg=0.01) ...")
x_sep     = jnp.array(x_init_np)
snapshots = [np.array(x_sep)]
mid       = N_STEPS_SEP // 2
for step in range(N_STEPS_SEP):
    g = grad_sandbox(x_sep, lb_j, DELTA_PULL, DELTA_PUSH, LAMBDA_FINAL)
    x_sep = x_sep - ETA * g
    if step == mid - 1:
        snapshots.append(np.array(x_sep))
x_final = np.array(x_sep)
snapshots.append(x_final)

centers_fin = np.stack([x_final[labels_np == c].mean(axis=0)
                         for c in range(N_GROUPS)])
dists_fin   = [np.linalg.norm(centers_fin[i] - centers_fin[j])
               for i in range(N_GROUPS) for j in range(i+1, N_GROUPS)]
print(f"  Final spread: {x_final.std(axis=0)}")
print(f"  Min inter-cluster dist (final): {min(dists_fin):.3f}")

# Save sandbox NPZ files
np.savez(os.path.join(ASSETS_DIR, "nb2_sandbox_initial.npz"),
         positions=x_init_np, labels=labels_np)
np.savez(os.path.join(ASSETS_DIR, "nb2_sandbox_collapsed.npz"),
         positions=x_collapsed, labels=labels_np)
np.savez(os.path.join(ASSETS_DIR, "nb2_sandbox_final.npz"),
         positions=x_final, labels=labels_np,
         snapshots=np.stack(snapshots))
print("  Saved sandbox NPZ files")

# ── Figure: 3-panel sandbox filmstrip ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Sandbox de Partículas — potencial pull/push", fontsize=13)

all_pts  = np.concatenate([x_init_np, x_collapsed, x_final])
plot_lim = max(abs(all_pts).max() * 1.15, 3.0)

for ax, (pts, title) in zip(axes, [
    (x_init_np,   "Estado inicial\n(grupos misturados)"),
    (x_collapsed, "δ_push = 0 → colapso trivial\n(tudo no mesmo ponto)"),
    (x_final,     f"δ_push = {DELTA_PUSH} → grupos separados\n(equilíbrio com repulsão)"),
]):
    for c in range(N_GROUPS):
        mask = labels_np == c
        ax.scatter(pts[mask, 0], pts[mask, 1],
                   s=22, color=CORES_5[c], alpha=0.85,
                   edgecolors="none", label=f"Grupo {c}")
    # Mark cluster centers
    ctrs = np.stack([pts[labels_np == c].mean(axis=0) for c in range(N_GROUPS)])
    for c in range(N_GROUPS):
        ax.scatter(ctrs[c, 0], ctrs[c, 1],
                   s=80, color=CORES_5[c], marker="X",
                   edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_xlim(-plot_lim, plot_lim); ax.set_ylim(-plot_lim, plot_lim)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb2_fig_sandbox_final.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb2_fig_sandbox_final.png  ({os.path.getsize(p)//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Encoder architecture helpers
# ══════════════════════════════════════════════════════════════════════════════

def init_encoder(layer_sizes, key):
    """He-normal init (suited for ReLU). Returns list of (W, b)."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, kw = jax.random.split(key)
        fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
        W = jax.random.normal(kw, (fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        b = jnp.zeros(fan_out)
        params.append((W, b))
    return params


def forward_encoder(params, x):
    """ReLU hidden layers, linear output."""
    h = x
    for W, b in params[:-1]:
        h = jax.nn.relu(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def adam_init(params):
    m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    v = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    return m, v


def adam_step(params, grads, m, v, t, lr=1e-3,
              b1=0.9, b2=0.999, eps=1e-8):
    new_m = [(b1*mW + (1-b1)*gW, b1*mb + (1-b1)*gb)
             for (mW, mb), (gW, gb) in zip(m, grads)]
    new_v = [(b2*vW + (1-b2)*gW**2, b2*vb + (1-b2)*gb**2)
             for (vW, vb), (gW, gb) in zip(v, grads)]
    mhat  = [(mW/(1-b1**t), mb/(1-b1**t)) for mW, mb in new_m]
    vhat  = [(vW/(1-b2**t), vb/(1-b2**t)) for vW, vb in new_v]
    new_p = [(W - lr * mWh / (jnp.sqrt(vWh) + eps),
              b - lr * mbh / (jnp.sqrt(vbh) + eps))
             for (W, b), (mWh, mbh), (vWh, vbh)
             in zip(params, mhat, vhat)]
    return new_p, new_m, new_v


def save_pkl(params, fname):
    path = os.path.join(ASSETS_DIR, fname)
    with open(path, "wb") as f:
        pickle.dump([(np.array(W), np.array(b)) for W, b in params], f)
    print(f"  Saved {fname}  ({os.path.getsize(path)//1024} KB)")


def load_pkl(fname):
    with open(os.path.join(ASSETS_DIR, fname), "rb") as f:
        return pickle.load(f)


def get_embeddings(enc, X):
    z = forward_encoder(enc, jnp.array(X, dtype=jnp.float32))
    return np.array(z)


def stratified_batch(rng, X, y, batch_size, n_cls=10):
    """Mini-batch with balanced class representation."""
    per_cls = batch_size // n_cls
    idx = []
    for c in range(n_cls):
        c_idx = np.where(y == c)[0]
        idx.append(rng.choice(c_idx, per_cls, replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx)
    return jnp.array(X[idx]), jnp.array(y[idx])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MNIST 2D encoder training
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== 3. Training MNIST 2D encoder ===")

N_CLASSES   = 10
DIM_2D      = 2
LR_MNIST    = 2e-3
DP_PULL     = 0.3    # tight pull radius
DP_PUSH     = 2.5    # generous push margin
LAMBDA_DISC = 0.001  # very weak reg, geometry unconstrained
BATCH_SZ    = 500    # 50 per class
EARLY_EP    = 20
LATE_EP     = 250

mnist_wloss = make_weinberger_loss(N_CLASSES)

def disc_loss_2d(enc_params, X_batch, y_batch, dp, dpu, lam):
    x = forward_encoder(enc_params, X_batch)
    return mnist_wloss(x, y_batch, dp, dpu, lam)

grad_2d   = jax.jit(jax.grad(disc_loss_2d, argnums=0))
rng_batch = np.random.default_rng(0)

key, ke = jax.random.split(key)
enc_2d  = init_encoder([784, 256, 64, DIM_2D], ke)
save_pkl(enc_2d, "nb2_encoder_epoch0.pkl")
m_e, v_e = adam_init(enc_2d)

for epoch in range(1, LATE_EP + 1):
    Xb, yb = stratified_batch(rng_batch, X_train, y_train, BATCH_SZ)
    g = grad_2d(enc_2d, Xb, yb, DP_PULL, DP_PUSH, LAMBDA_DISC)
    enc_2d, m_e, v_e = adam_step(enc_2d, g, m_e, v_e, epoch, lr=LR_MNIST)

    if epoch == EARLY_EP:
        save_pkl(enc_2d, "nb2_encoder_early.pkl")
        print(f"  Epoch {epoch} — early checkpoint saved")

    if epoch % 50 == 0:
        Xj = jnp.array(X_train, dtype=jnp.float32)
        yj = jnp.array(y_train, dtype=jnp.int32)
        lv  = float(disc_loss_2d(enc_2d, Xj, yj, DP_PULL, DP_PUSH, LAMBDA_DISC))
        Z2  = get_embeddings(enc_2d, X_test)
        ari = adjusted_rand_score(y_test,
                  KMeans(n_clusters=10, random_state=42, n_init=5).fit_predict(Z2))
        print(f"  Epoch {epoch:3d}  loss={lv:.4f}  ARI={ari:.3f}")

save_pkl(enc_2d, "nb2_encoder_late.pkl")
Z_test_2d = get_embeddings(enc_2d, X_test)
ari_2d    = adjusted_rand_score(y_test,
                KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(Z_test_2d))
print(f"  Final K-means ARI (2D, test): {ari_2d:.3f}")
if ari_2d < 0.7:
    print("  WARNING: ARI < 0.7")


# ── Evolution filmstrip figure ─────────────────────────────────────────────────
enc_e0    = load_pkl("nb2_encoder_epoch0.pkl")
enc_early = load_pkl("nb2_encoder_early.pkl")
enc_late  = load_pkl("nb2_encoder_late.pkl")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Evolução do Espaço de Embeddings (MNIST, encoder 2D)", fontsize=13)

for ax, (enc, title) in zip(axes, [
    (enc_e0,    "Época 0 — caos\n(inicialização aleatória)"),
    (enc_early, f"Época {EARLY_EP} — início do agrupamento"),
    (enc_late,  f"Época {LATE_EP} — classes separadas"),
]):
    Z = get_embeddings(enc, X_test)
    for c in range(N_CLASSES):
        mask = y_test == c
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   s=8, color=CORES_10[c], alpha=0.7,
                   edgecolors="none", label=str(c))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.legend(fontsize=7, title="Dígito", markerscale=2)
    ax.grid(True, alpha=0.15)

plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb2_fig_evolution.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb2_fig_evolution.png  ({os.path.getsize(p)//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MNIST 16D encoder training
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== 4. Training MNIST 16D encoder ===")

DIM_16 = 16

def disc_loss_16d(enc_params, X_batch, y_batch, dp, dpu, lam):
    x = forward_encoder(enc_params, X_batch)
    return mnist_wloss(x, y_batch, dp, dpu, lam)

grad_16d = jax.jit(jax.grad(disc_loss_16d, argnums=0))

key, ke16 = jax.random.split(key)
enc_16d   = init_encoder([784, 256, 64, DIM_16], ke16)
m_16, v_16 = adam_init(enc_16d)

for epoch in range(1, LATE_EP + 1):
    Xb, yb = stratified_batch(rng_batch, X_train, y_train, BATCH_SZ)
    g = grad_16d(enc_16d, Xb, yb, DP_PULL, DP_PUSH, LAMBDA_DISC)
    enc_16d, m_16, v_16 = adam_step(enc_16d, g, m_16, v_16, epoch, lr=LR_MNIST)
    if epoch % 50 == 0:
        Xj = jnp.array(X_train, dtype=jnp.float32)
        yj = jnp.array(y_train, dtype=jnp.int32)
        lv = float(disc_loss_16d(enc_16d, Xj, yj, DP_PULL, DP_PUSH, LAMBDA_DISC))
        print(f"  Epoch {epoch:3d}  loss={lv:.4f}")

save_pkl(enc_16d, "nb2_encoder_16d_late.pkl")
Z_test_16d = get_embeddings(enc_16d, X_test)
ari_16d    = adjusted_rand_score(y_test,
                 KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(Z_test_16d))
print(f"  K-means ARI (16D, test): {ari_16d:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — t-SNE figure (3 perplexities)
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== 5. Generating t-SNE figure ===")

rng_tsne = np.random.default_rng(7)
idx_ts   = rng_tsne.choice(len(X_test), min(1000, len(X_test)), replace=False)
Z_ts     = Z_test_16d[idx_ts]
y_ts     = y_test[idx_ts]

PERPLEXITIES = [5, 30, 100]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("t-SNE do Espaço 16D — a geometria muda com a perplexidade", fontsize=13)

for ax, perp in zip(axes, PERPLEXITIES):
    print(f"  t-SNE perplexity={perp} ...")
    tsne = TSNE(n_components=2, perplexity=perp,
                random_state=42, max_iter=500,
                learning_rate="auto", init="pca")
    Z2 = tsne.fit_transform(Z_ts)
    for c in range(N_CLASSES):
        mask = y_ts == c
        ax.scatter(Z2[mask, 0], Z2[mask, 1],
                   s=8, color=CORES_10[c], alpha=0.7,
                   edgecolors="none", label=str(c))
    ax.set_title(f"Perplexidade = {perp}", fontsize=11)
    ax.set_xlabel("t-SNE₁"); ax.set_ylabel("t-SNE₂")
    ax.legend(fontsize=7, title="Dígito", markerscale=2)
    ax.grid(True, alpha=0.15)

plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb2_fig_tsne.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb2_fig_tsne.png  ({os.path.getsize(p)//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — UMAP placeholder figure
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== 6. UMAP placeholder figure ===")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("UMAP — disponível no Colab: !pip install umap-learn==0.5.7",
             fontsize=13)
for ax, nn in zip(axes, [5, 15, 50]):
    ax.text(0.5, 0.5,
            f"UMAP  (n_neighbors = {nn})\n\n"
            "umap-learn não instalado\nneste ambiente local.\n\n"
            "No Colab, execute:\n"
            "!pip install umap-learn==0.5.7",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#ecf0f1", edgecolor="#bdc3c7"))
    ax.set_title(f"n_neighbors = {nn}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#fafafa")

plt.tight_layout()
p = os.path.join(ASSETS_DIR, "nb2_fig_umap.png")
fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
print(f"  Saved nb2_fig_umap.png  ({os.path.getsize(p)//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Summary
# ══════════════════════════════════════════════════════════════════════════════

elapsed = time.time() - t_start
print(f"\n=== Asset summary  (total time: {elapsed:.1f}s) ===")

asset_files = [
    "mnist_4k.npz",
    "nb2_sandbox_initial.npz",
    "nb2_sandbox_collapsed.npz",
    "nb2_sandbox_final.npz",
    "nb2_encoder_epoch0.pkl",
    "nb2_encoder_early.pkl",
    "nb2_encoder_late.pkl",
    "nb2_encoder_16d_late.pkl",
    "nb2_fig_sandbox_final.png",
    "nb2_fig_evolution.png",
    "nb2_fig_tsne.png",
    "nb2_fig_umap.png",
]

total_kb = 0.0
for fname in asset_files:
    path = os.path.join(ASSETS_DIR, fname)
    if os.path.exists(path):
        kb = os.path.getsize(path) / 1024
        total_kb += kb
        print(f"  {fname:45s}  {kb:8.1f} KB")
    else:
        print(f"  {fname:45s}  MISSING!")

print(f"\n  Total: {total_kb:.1f} KB  ({total_kb/1024:.2f} MB)")
print("\nBUILD ASSETS: ALL GENERATED SUCCESSFULLY")
