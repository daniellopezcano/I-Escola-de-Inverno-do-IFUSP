# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: WinterSchool
#     language: python
#     name: python3
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/02_contrastive_embeddings_v2.ipynb)
#
# # Notebook 02 — Embeddings Contrastivos
# ### Do espaco de pixels a representacoes sem rotulos
# **I Escola de Inverno do IFUSP — Bloco L3B2**
#
# > **Modo de uso:** demo guiada pelo instrutor; voces recebem o notebook depois.
# > Ao final, teremos construido representacoes de digitos MNIST de duas
# > formas — com rotulos (classificador) e sem rotulos (InfoNCE) — e visto
# > por que o espaco de pixels cru nao funciona.
#
# ---
#
# | Parte | Tema | Referencia em L3B1 |
# |-------|------|--------------------|
# | 1 | Pixels cru: distancia $\neq$ significado | §2.1 — Por que coordenadas brutas falham |
# | 2 | Espaco latente 2D com classificador | §2 — Espacos latentes e seu papel |
# | 3 | Augmentacoes como declaracao de invariancia | §3.3 — Augmentacoes como declaracoes de invariancia |
# | 4 | InfoNCE: estrutura sem rotulos | §4.4 — InfoNCE e a temperatura de Boltzmann |

# %%
# Instalacao de pacotes (so no Colab — localmente ja estao instalados)
import subprocess, sys
try:
    import google.colab  # noqa: F401
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "jax", "jaxlib", "equinox", "optax", "matplotlib", "scikit-learn"])
except ImportError:
    pass

# %%
import gzip
import struct
import urllib.request
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from pathlib import Path

# Semente de reprodutibilidade
SEED = 42
KEY = jax.random.PRNGKey(SEED)

# Estilo dos graficos
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

# Paleta de cores (10 classes MNIST)
CORES = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#c0392b", "#27ae60",
]

print(f"JAX versao  : {jax.__version__}")
print(f"Dispositivo : {jax.devices()[0]}")

# %% [markdown]
# ### Carregar o MNIST
#
# Baixamos o MNIST direto da internet. Para treino usamos um subconjunto
# pequeno (2 000 imagens) para manter o runtime curto. Para **avaliacao
# e scatter plots** usamos o conjunto de teste completo (10 000 imagens)
# — gerar embeddings e barato e mais pontos revelam melhor a estrutura
# dos clusters.

# %%
def baixar_mnist(cache_dir="/tmp/mnist_cache"):
    """Baixa o MNIST e retorna (X_train, y_train, X_test, y_test) completos."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    nomes = {
        "tr_img": "train-images-idx3-ubyte.gz",
        "tr_lab": "train-labels-idx1-ubyte.gz",
        "te_img": "t10k-images-idx3-ubyte.gz",
        "te_lab": "t10k-labels-idx1-ubyte.gz",
    }
    dados = {}
    for k, fname in nomes.items():
        local = cache / fname
        if not local.exists():
            print(f"  Baixando {fname}...")
            urllib.request.urlretrieve(url + fname, str(local))
        with gzip.open(str(local), "rb") as f:
            raw = f.read()
        if "img" in k:
            _, n, r, c = struct.unpack(">IIII", raw[:16])
            dados[k] = (np.frombuffer(raw[16:], dtype=np.uint8)
                        .reshape(n, r, c).astype(np.float32) / 255.0)
        else:
            _, n = struct.unpack(">II", raw[:8])
            dados[k] = np.frombuffer(raw[8:], dtype=np.uint8).astype(np.int32)
    return dados["tr_img"], dados["tr_lab"], dados["te_img"], dados["te_lab"]


def subset_balanceado(X, y, n_por_classe, rng):
    """Cria subset com n_por_classe amostras de cada digito."""
    idx = []
    for c in range(10):
        ic = np.where(y == c)[0]
        idx.append(rng.choice(ic, n_por_classe, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


X_full_tr, y_full_tr, X_full_te, y_full_te = baixar_mnist()
rng_data = np.random.default_rng(SEED)
X_train, y_train = subset_balanceado(X_full_tr, y_full_tr, 200, rng_data)

# Avaliacao: conjunto de teste COMPLETO (10 000 imagens)
X_eval, y_eval = X_full_te, y_full_te

print(f"Treino      : {X_train.shape[0]:,} imagens  (200/classe x 10 classes)")
print(f"Avaliacao   : {X_eval.shape[0]:,} imagens  (teste MNIST completo)")
print(f"Pixels      : [{X_train.min():.0f}, {X_train.max():.0f}]")

# %%
# Grade 5x10 — uma amostra dos dados
fig, axes = plt.subplots(5, 10, figsize=(12, 6))
fig.suptitle("MNIST — 5 amostras por digito (0-9)", fontsize=13)
for c in range(10):
    idx_c = np.where(y_eval == c)[0][:5]
    for row, i in enumerate(idx_c):
        ax = axes[row, c]
        ax.imshow(X_eval[i], cmap="gray_r", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(str(c), fontsize=11, color=CORES[c], fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Parte 1 — O espaco de pixels cru e dificil de interpretar
#
# *Referencia: L3B1 §2.1 — Por que coordenadas brutas falham*
#
# Cada imagem MNIST e um vetor de **784 dimensoes** (28 x 28 pixels).
# Sera que a distancia euclidiana nesse espaco reflete similaridade
# entre digitos? Vamos verificar.

# %%
# Cinco imagens do digito "7" — mesma classe, aparencias muito diferentes
idx_7 = np.where(y_eval == 7)[0][:5]

fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
fig.suptitle('Cinco "7"s do MNIST — mesma classe, aparencias diferentes', fontsize=12)
for ax, i in zip(axes, idx_7):
    ax.imshow(X_eval[i], cmap="gray_r", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

print("Cada imagem e um vetor de 784 numeros.")
print("A olho: sao todos '7'. Mas os vetores de pixels sao muito diferentes.")

# %%
# Punchline: dois "7"s podem estar MAIS DISTANTES que um "7" e um "1"!
# (Usamos um subset de 100 por classe para calcular distancias pairwise.)

# Subset pequeno para pairwise distances (evitar estouro de memoria)
rng_part1 = np.random.default_rng(0)
X_p1, y_p1 = subset_balanceado(X_eval, y_eval, 100, rng_part1)
X_flat_p1 = X_p1.reshape(len(X_p1), -1)

# Procurar par do mesmo digito (7-7) com grande distancia
idx_7_all = np.where(y_p1 == 7)[0]
X_7 = X_flat_p1[idx_7_all]
dists_77 = np.linalg.norm(X_7[:, None] - X_7[None, :], axis=-1)
np.fill_diagonal(dists_77, 0)
i7, j7 = np.unravel_index(dists_77.argmax(), dists_77.shape)
d_same = dists_77[i7, j7]

# Procurar par de digitos diferentes (7 vs 1) com distancia menor
idx_1_all = np.where(y_p1 == 1)[0]
X_1 = X_flat_p1[idx_1_all]
dists_71 = np.linalg.norm(X_7[:, None] - X_1[None, :], axis=-1)
i71, j71 = np.unravel_index(dists_71.argmin(), dists_71.shape)
d_diff = dists_71[i71, j71]

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
fig.suptitle("Distancia de pixels e enganosa!", fontsize=13)

axes[0].imshow(X_p1[idx_7_all[i7]], cmap="gray_r"); axes[0].set_title("7 (A)")
axes[1].imshow(X_p1[idx_7_all[j7]], cmap="gray_r"); axes[1].set_title("7 (B)")
axes[2].imshow(X_p1[idx_7_all[i71]], cmap="gray_r"); axes[2].set_title("7 (C)")
axes[3].imshow(X_p1[idx_1_all[j71]], cmap="gray_r"); axes[3].set_title("1 (D)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.show()

print(f"Distancia entre dois '7's (A e B):  {d_same:.1f}")
print(f"Distancia entre um '7' e um '1' (C e D): {d_diff:.1f}")
print()
if d_same > d_diff:
    print(f"  -> O par do MESMO digito esta MAIS LONGE ({d_same:.1f} > {d_diff:.1f})!")
    print("     Distancia de pixels NAO mede similaridade de significado.")

# %%
# Distribuicao sistematica: intra-classe vs inter-classe
n_amostra = 300
idx_s = rng_part1.choice(len(X_p1), n_amostra, replace=False)
X_s = X_flat_p1[idx_s]
y_s = y_p1[idx_s]

dists_todos = np.linalg.norm(X_s[:, None] - X_s[None, :], axis=-1)

intra, inter = [], []
for i in range(n_amostra):
    for j in range(i + 1, n_amostra):
        if y_s[i] == y_s[j]:
            intra.append(dists_todos[i, j])
        else:
            inter.append(dists_todos[i, j])

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(intra, bins=50, alpha=0.6, label="Mesmo digito", color="#3498db", density=True)
ax.hist(inter, bins=50, alpha=0.6, label="Digitos diferentes", color="#e74c3c", density=True)
ax.set_xlabel("Distancia L2 (espaco de pixels, 784D)")
ax.set_ylabel("Densidade")
ax.set_title("As distribuicoes se sobrepoem — pixels nao separam classes!")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Translacao: mesma imagem deslocada 3 pixels — semanticamente identica, longe em pixels
idx_demo = np.where(y_eval == 3)[0][0]
img_orig = X_eval[idx_demo]
img_shift = np.roll(img_orig, 3, axis=1)  # 3 pixels para a direita

d_trans = np.linalg.norm(img_orig.flatten() - img_shift.flatten())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
fig.suptitle(f"Translacao de 3 pixels — distancia L2 = {d_trans:.1f}", fontsize=12)
ax1.imshow(img_orig, cmap="gray_r"); ax1.set_title("Original")
ax2.imshow(img_shift, cmap="gray_r"); ax2.set_title("Deslocado 3px")
for ax in (ax1, ax2):
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

print("Visualmente identicos, mas a distancia de pixels e grande.")
print("Precisamos de um espaco onde 'perto' signifique 'parecido'.")

# %% [markdown]
# **Resumo da Parte 1:** o espaco de pixels (784D) e enganoso.
# Distancia euclidiana nele nao reflete similaridade semantica.
# Duas imagens do mesmo digito podem estar mais longe que duas de
# digitos diferentes. Precisamos de um **espaco latente** melhor.

# %% [markdown]
# ---
# ## Parte 2 — Um espaco latente 2D emerge de um classificador
#
# *Referencia: L3B1 §2 — Espacos latentes e seu papel no ML moderno*
#
# Vamos construir um **encoder CNN** que comprime a imagem 28x28
# para apenas **2 numeros** (coordenadas latentes), seguido de uma
# cabeca de classificacao (2D $\to$ 10 classes).
#
# Se o bottleneck funcionar, todas as informacoes sobre o digito
# passam por esses 2 numeros — e as classes devem se separar no
# espaco 2D.

# %%
# Arquitetura: CNN encoder com bottleneck 2D

class Encoder(eqx.Module):
    """CNN encoder: imagem (1,28,28) -> vetor latente (dim_latente,)."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc_out: eqx.nn.Linear

    def __init__(self, dim_latente, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, key=k2)
        # Apos conv2: (32, 7, 7) -> flatten -> 1568
        self.fc1 = eqx.nn.Linear(32 * 7 * 7, 64, key=k3)
        self.fc_out = eqx.nn.Linear(64, dim_latente, key=k4)

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))    # (16, 14, 14)
        x = jax.nn.relu(self.conv2(x))    # (32, 7, 7)
        x = x.reshape(-1)                 # (1568,)
        x = jax.nn.relu(self.fc1(x))      # (64,)
        return self.fc_out(x)             # (dim_latente,)


class Classificador(eqx.Module):
    """Encoder 2D + cabeca de classificacao (2D -> 10 classes)."""
    encoder: Encoder
    head: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.encoder = Encoder(dim_latente=2, key=k1)
        self.head = eqx.nn.Linear(2, 10, key=k2)

    def __call__(self, x):
        z = self.encoder(x)   # bottleneck 2D
        return self.head(z)   # logits (10,)


# Criar modelo
KEY, k_modelo = jax.random.split(KEY)
classificador = Classificador(key=k_modelo)

# Teste de shape
x_teste = jnp.zeros((1, 28, 28))
z_teste = classificador.encoder(x_teste)
logits_teste = classificador(x_teste)
n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(classificador, eqx.is_array)))
print(f"Encoder: (1, 28, 28) -> {z_teste.shape}  (bottleneck 2D)")
print(f"Classificador: (1, 28, 28) -> {logits_teste.shape}  (logits)")
print(f"Parametros totais: {n_params:,}")

# %%
# Preparar dados no formato CNN e definir funcoes de treino

# Formato CNN: (N, 1, 28, 28) como JAX arrays
X_train_cnn = jnp.array(X_train[:, None])
y_train_jnp = jnp.array(y_train)

# Conjunto de avaliacao completo (10k) — carregado em batches para plots
X_eval_cnn = jnp.array(X_eval[:, None])
y_eval_jnp = jnp.array(y_eval)

N_TRAIN = len(X_train)
BATCH_SUP = 128


def perda_ce(modelo, x_batch, y_batch):
    """Entropia cruzada (cross-entropy)."""
    logits = jax.vmap(modelo)(x_batch)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()


otimizador_sup = optax.adam(learning_rate=2e-3)
opt_state_sup = otimizador_sup.init(eqx.filter(classificador, eqx.is_array))


@eqx.filter_jit
def passo_sup(modelo, opt_state, x, y):
    """Um passo de treino supervisionado."""
    perda, grads = eqx.filter_value_and_grad(perda_ce)(modelo, x, y)
    updates, opt_state = otimizador_sup.update(grads, opt_state, modelo)
    modelo = eqx.apply_updates(modelo, updates)
    return modelo, opt_state, perda


print(f"Treino   : {N_TRAIN:,} imagens")
print(f"Avaliacao: {len(X_eval):,} imagens (scatter plots)")
print(f"Batch: {BATCH_SUP}   Otimizador: Adam (lr=2e-3)")

# %%
# Treino supervisionado
N_EPOCAS_SUP = 30
historico_sup = []
chave_sup = jax.random.PRNGKey(1)

print(f"Treinando classificador ({N_EPOCAS_SUP} epocas)...")

for epoca in range(1, N_EPOCAS_SUP + 1):
    chave_sup, k_perm = jax.random.split(chave_sup)
    perm = jax.random.permutation(k_perm, N_TRAIN)

    for i in range(0, N_TRAIN, BATCH_SUP):
        idx = perm[i:i + BATCH_SUP]
        classificador, opt_state_sup, _ = passo_sup(
            classificador, opt_state_sup, X_train_cnn[idx], y_train_jnp[idx]
        )

    # Perda no treino
    p_tr = float(perda_ce(classificador, X_train_cnn, y_train_jnp))
    historico_sup.append((epoca, p_tr))
    if epoca % 10 == 0 or epoca == 1:
        print(f"  Epoca {epoca:2d}  treino={p_tr:.4f}")

print("Treino concluido.")

# %%
# Curvas de aprendizado
epocas, l_tr = zip(*historico_sup)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(epocas, l_tr, "-o", ms=3, lw=1.5, color="#2980b9", label="treino")
ax.set_xlabel("Epoca")
ax.set_ylabel("Perda (cross-entropy)")
ax.set_title("Curvas de aprendizado — classificador com bottleneck 2D")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# FIGURA-CHAVE: scatter do espaco latente 2D (10 000 pontos do teste)

def obter_embeddings(encoder, X_cnn, batch_sz=512):
    """Calcula embeddings em lotes. X_cnn: JAX array (N, 1, 28, 28)."""
    partes = []
    for i in range(0, len(X_cnn), batch_sz):
        partes.append(np.array(jax.vmap(encoder)(X_cnn[i:i + batch_sz])))
    return np.concatenate(partes)


Z_sup = obter_embeddings(classificador.encoder, X_eval_cnn)

fig, ax = plt.subplots(figsize=(7, 6))
for c in range(10):
    mask = y_eval == c
    ax.scatter(Z_sup[mask, 0], Z_sup[mask, 1],
               s=5, color=CORES[c], alpha=0.5,
               edgecolors="none", label=str(c))
ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
ax.set_title("Espaco latente 2D — classificador supervisionado\n(10 000 pontos do teste)", fontsize=12)
ax.legend(title="Digito", fontsize=8, markerscale=2.5, framealpha=0.8)
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.show()

print(f"Scatter com {len(Z_sup):,} pontos — cada ponto e uma imagem do teste.")
print("As classes se separam — o bottleneck aprendeu coordenadas uteis!")

# %%
# Quais classes ficam proximas? Distancia entre centroides

centroides = np.array([Z_sup[y_eval == c].mean(axis=0) for c in range(10)])
dist_centroides = np.linalg.norm(centroides[:, None] - centroides[None, :], axis=-1)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(dist_centroides, cmap="YlOrRd_r", interpolation="nearest")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Digito"); ax.set_ylabel("Digito")
ax.set_title("Distancia entre centroides no espaco 2D")
plt.colorbar(im, ax=ax, fraction=0.046)
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{dist_centroides[i, j]:.1f}",
                ha="center", va="center", fontsize=7,
                color="white" if dist_centroides[i, j] < dist_centroides.max() * 0.5 else "black")
plt.tight_layout()
plt.show()

# Encontrar os pares mais proximos
pares = []
for i in range(10):
    for j in range(i + 1, 10):
        pares.append((dist_centroides[i, j], i, j))
pares.sort()
print("Pares de digitos mais proximos no espaco latente:")
for d, i, j in pares[:4]:
    print(f"  {i} e {j}: distancia = {d:.2f}")
print("\nFaz sentido? Digitos visualmente parecidos ficam proximos!")

# %% [markdown]
# **O que aprendemos:** o encoder encontrou coordenadas 2D onde classes
# similares ficam proximas. A estrutura emergiu do treino supervisionado
# — os rotulos disseram quem deve ficar perto de quem.
#
# **Pergunta para a Parte 4:** e possivel obter essa estrutura
# **sem usar nenhum rotulo**?

# %% [markdown]
# ---
# ## Parte 3 — Augmentacoes como declaracoes de invariancia
#
# *Referencia: L3B1 §3.3 — Augmentacoes como declaracoes de invariancia*
#
# Na aprendizagem contrastiva auto-supervisionada, os **pares positivos**
# vem de **augmentacoes**: duas versoes distorcidas da mesma imagem sao
# declaradas como "o mesmo objeto."
#
# Cada augmentacao e uma afirmacao:
# > "Esta transformacao **nao deve mudar** a identidade do objeto."
#
# Ou, na linguagem da fisica: cada augmentacao declara uma **simetria**
# que o encoder deve respeitar.

# %% [markdown]
# ### Por que augmentacoes fortes sao essenciais
#
# A literatura mostra que augmentacoes **fracas** (so ruido + pequena
# translacao) permitem que o encoder aprenda **atalhos** (*shortcuts*):
# em vez de capturar a forma global do digito, a rede explora a posicao
# absoluta ou a tinta total para distinguir pares positivos dos negativos,
# sem jamais aprender semantica.
#
# O artigo fundacional de **Simard, Steinkraus & Platt (2003, ICDAR)**
# mostrou que **deformacoes elasticas** — que emulam a variabilidade
# natural da escrita manual — sao a augmentacao mais poderosa para MNIST.
# Mais recentemente, **SimCLR (Chen et al. 2020, ICML)** demonstrou que
# a **composicao agressiva** de multiplas transformacoes e o fator
# critico: nenhuma transformacao isolada basta — e a composicao que
# forca o encoder a representar informacao de alto nivel.
#
# Nossa pipeline combina: deformacao elastica + transformacao afim
# (rotacao, cisalhamento, escala) + blur + ruido + cutout.

# %%
# Pipeline de augmentacoes FORTE para MNIST
from scipy.ndimage import gaussian_filter, map_coordinates, affine_transform as scipy_affine


def deformacao_elastica(img, rng, alpha=6.0, sigma=3.5):
    """Deformacao elastica (Simard et al. 2003).

    Gera campos de deslocamento aleatorios suavizados por um filtro
    gaussiano, depois reamostra a imagem com interpolacao bilinear.
    alpha: amplitude do deslocamento.
    sigma: largura do filtro (controla suavidade).
    """
    shape = img.shape  # (28, 28)
    # Campos de deslocamento aleatorios
    dx = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    # Coordenadas originais + deslocamento
    y_coords, x_coords = np.mgrid[0:shape[0], 0:shape[1]]
    coords = [y_coords + dy, x_coords + dx]
    return map_coordinates(img, coords, order=1, mode="constant", cval=0.0).astype(np.float32)


def transformacao_afim(img, rng, max_angulo=25, max_shear=0.15,
                       scale_range=(0.8, 1.2), max_trans=3):
    """Transformacao afim: rotacao + cisalhamento + escala independente + translacao.

    Segue Ciresan et al. (2010, arXiv:1003.0358) para MNIST.
    max_angulo: rotacao maxima em graus (limitado a ~25 para nao confundir 6/9).
    """
    h, w = img.shape
    cy, cx = h / 2.0, w / 2.0

    # Parametros aleatorios
    angulo = rng.uniform(-max_angulo, max_angulo) * np.pi / 180.0
    shear = rng.uniform(-max_shear, max_shear)
    sx = rng.uniform(*scale_range)
    sy = rng.uniform(*scale_range)
    tx = rng.uniform(-max_trans, max_trans)
    ty = rng.uniform(-max_trans, max_trans)

    # Matriz afim: escala * rotacao * cisalhamento
    cos_a, sin_a = np.cos(angulo), np.sin(angulo)
    # Rotacao + shear
    R = np.array([[cos_a, -sin_a],
                  [sin_a, cos_a]])
    S = np.array([[1.0 / sx, shear],
                  [0.0, 1.0 / sy]])
    M = S @ R  # Combinada (inversa para map de destino -> fonte)

    # Offset para manter centralizado
    offset = np.array([cy, cx]) - M @ np.array([cy + ty, cx + tx])

    return scipy_affine(img, M, offset=offset, order=1,
                        mode="constant", cval=0.0).astype(np.float32)


def blur_gaussiano(img, rng, sigma_max=1.2):
    """Blur gaussiano com sigma aleatorio."""
    sigma = rng.uniform(0.3, sigma_max)
    return gaussian_filter(img, sigma).astype(np.float32)


def cutout(img, rng, tamanho_max=8):
    """Apaga um patch retangular aleatorio (random erasing)."""
    img_out = img.copy()
    h, w = img.shape
    th = rng.integers(3, tamanho_max + 1)
    tw = rng.integers(3, tamanho_max + 1)
    y0 = rng.integers(0, max(1, h - th))
    x0 = rng.integers(0, max(1, w - tw))
    img_out[y0:y0 + th, x0:x0 + tw] = 0.0
    return img_out


def augmentar_imagem(img, rng):
    """Pipeline composta FORTE para MNIST (Simard 2003 + SimCLR 2020).

    Aplica sequencialmente:
    1. Deformacao elastica (sempre)
    2. Transformacao afim: rotacao +-25, shear, escala, translacao (sempre)
    3. Blur gaussiano (50% de chance)
    4. Ruido aditivo (sempre)
    5. Cutout / random erasing (50% de chance)
    """
    # 1. Deformacao elastica
    img = deformacao_elastica(img, rng, alpha=6.0, sigma=3.5)

    # 2. Transformacao afim
    img = transformacao_afim(img, rng, max_angulo=25, max_shear=0.15,
                            scale_range=(0.8, 1.2), max_trans=3)

    # 3. Blur gaussiano (probabilidade 50%)
    if rng.random() < 0.5:
        img = blur_gaussiano(img, rng, sigma_max=1.2)

    # 4. Ruido gaussiano aditivo
    img = img + rng.normal(0, 0.08, img.shape).astype(np.float32)

    # 5. Cutout (probabilidade 50%)
    if rng.random() < 0.5:
        img = cutout(img, rng, tamanho_max=8)

    return np.clip(img, 0, 1).astype(np.float32)


def augmentar_batch(X_batch, rng):
    """Augmenta um batch inteiro. X_batch: (B, 28, 28)."""
    return np.stack([augmentar_imagem(x, rng) for x in X_batch])


print("Pipeline de augmentacoes FORTE implementada:")
print("  1. Deformacao elastica (Simard et al. 2003)")
print("  2. Afim: rotacao +-25, shear, escala x/y, translacao (Ciresan et al. 2010)")
print("  3. Blur gaussiano (50%)")
print("  4. Ruido aditivo")
print("  5. Cutout / random erasing (50%)")

# %%
# FIGURA: grade de MUITAS vistas augmentadas de um unico digito

rng_aug_viz = np.random.default_rng(7)
idx_demo_aug = np.where(y_train == 3)[0][2]
img_demo = X_train[idx_demo_aug]

n_linhas, n_colunas = 5, 8  # 40 vistas
fig, axes = plt.subplots(n_linhas, n_colunas, figsize=(13, 8.5))
fig.suptitle('Um "3" original e 39 vistas augmentadas — pipeline composta forte',
             fontsize=13, y=0.98)

# Primeiro slot: original
axes[0, 0].imshow(img_demo, cmap="gray_r", interpolation="nearest")
axes[0, 0].set_title("Original", fontsize=8, fontweight="bold", color="#2980b9")

# Vistas augmentadas
for k in range(1, n_linhas * n_colunas):
    row, col = divmod(k, n_colunas)
    vista = augmentar_imagem(img_demo, rng_aug_viz)
    axes[row, col].imshow(vista, cmap="gray_r", interpolation="nearest")

for ax_row in axes:
    for ax in ax_row:
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("As distorcoes sao FORTES: elastica + afim + blur + ruido + cutout.")
print("O digito continua legivel, mas atalhos (posicao, tinta total) sao destruidos.")

# %%
# FIGURA: par positivo vs pares negativos
rng_pn = np.random.default_rng(42)
# Escolher imagens de digitos diferentes
imgs_pn = []
labs_pn = []
for dig in [3, 7, 0, 5]:
    i = np.where(y_train == dig)[0][0]
    imgs_pn.append(X_train[i])
    labs_pn.append(dig)

fig = plt.figure(figsize=(14, 5.5))
fig.suptitle("Par positivo vs pares negativos", fontsize=14, y=0.98)

# Par positivo: ancora + duas vistas
ancora = imgs_pn[0]
vista_a = augmentar_imagem(ancora, rng_pn)
vista_b = augmentar_imagem(ancora, rng_pn)

ax_anc = fig.add_subplot(2, 4, 1)
ax_v1 = fig.add_subplot(2, 4, 2)
ax_v2 = fig.add_subplot(2, 4, 3)
ax_txt1 = fig.add_subplot(2, 4, 4)

ax_anc.imshow(ancora, cmap="gray_r"); ax_anc.set_title("Ancora", fontsize=10)
ax_v1.imshow(vista_a, cmap="gray_r"); ax_v1.set_title("Vista 1", fontsize=10)
ax_v2.imshow(vista_b, cmap="gray_r"); ax_v2.set_title("Vista 2", fontsize=10)
ax_txt1.axis("off")
ax_txt1.text(0.0, 0.5, "PAR POSITIVO\n(mesmo objeto)", fontsize=12,
             color="#2ecc71", fontweight="bold", va="center", ha="left")

for ax in [ax_v1, ax_v2]:
    for spine in ax.spines.values():
        spine.set_edgecolor("#2ecc71"); spine.set_linewidth(3)

# Pares negativos: vista 1 da ancora vs vistas de outros digitos
ax_ref = fig.add_subplot(2, 4, 5)
ax_ref.imshow(vista_a, cmap="gray_r"); ax_ref.set_title("Vista 1", fontsize=10)

for k, (img, lab) in enumerate(zip(imgs_pn[1:], labs_pn[1:]), 6):
    ax_neg = fig.add_subplot(2, 4, k)
    v = augmentar_imagem(img, rng_pn)
    ax_neg.imshow(v, cmap="gray_r")
    ax_neg.set_title(f'"{lab}" augm.', fontsize=10)
    for spine in ax_neg.spines.values():
        spine.set_edgecolor("#e74c3c"); spine.set_linewidth(3)

for ax in fig.axes:
    ax.set_xticks([]); ax.set_yticks([])

# Adicionar texto descritivo
fig.text(0.75, 0.22, "PARES NEGATIVOS\n(objetos diferentes)", fontsize=12,
         color="#e74c3c", fontweight="bold", va="center", ha="center")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# %%
# Exemplo cautelar: augmentacao agressiva demais pode destruir o rotulo!
idx_6 = np.where(y_train == 6)[0][0]
img_6 = X_train[idx_6]
img_6_rot180 = img_6[::-1, ::-1]  # rotacao de 180 graus

idx_9 = np.where(y_train == 9)[0][0]
img_9 = X_train[idx_9]

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
fig.suptitle("Cuidado: rotacao de 180 graus transforma 6 em 9!", fontsize=12)
axes[0].imshow(img_6, cmap="gray_r"); axes[0].set_title('"6" original')
axes[1].imshow(img_6_rot180, cmap="gray_r"); axes[1].set_title('"6" rotacionado 180')
axes[2].imshow(img_9, cmap="gray_r"); axes[2].set_title('"9" real')
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

print("A escolha de augmentacoes e parte do design do modelo.")
print("Nossas rotacoes sao limitadas a +-25 graus — uma decisao de")
print("invariancia ESPECIFICA DO DOMINIO que preserva a distincao 6/9.")
print("Rotacoes maiores destruiriam a classe — uma invariancia ERRADA.")

# %% [markdown]
# **Resumo da Parte 3:** cada augmentacao declara uma invariancia.
# A deformacao elastica emula a variabilidade natural da caligrafia.
# Rotacao e escala declaram que orientacao/tamanho moderados sao
# irrelevantes. Cutout forca o uso da forma global (nao um pedaco).
#
# A **composicao forte** e essencial (Chen et al. 2020):
# augmentacoes fracas deixam atalhos faceis para a rede;
# augmentacoes compostas forcam o encoder a representar a
# **estrutura semantica** do digito.
#
# **O design das augmentacoes e parte do design do modelo.**
# Uma augmentacao mal escolhida planta invariancias erradas.

# %% [markdown]
# ---
# ## Parte 4 — Treinar com InfoNCE e recuperar a estrutura SEM rotulos
#
# *Referencia: L3B1 §4.4 — InfoNCE e a temperatura de Boltzmann*
#
# A perda InfoNCE e:
#
# $$\mathcal{L}_\text{InfoNCE} = -\log \frac{\exp\!\bigl(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau\bigr)}{\displaystyle\sum_{k \neq i} \exp\!\bigl(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau\bigr)}$$
#
# onde $\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$
# e a similaridade cosseno e $\tau > 0$ e a **temperatura**.

# %% [markdown]
# ### Arquitetura SimCLR: representacao $h$ e cabeca de projecao $z$
#
# Na pratica (Chen et al. 2020, SimCLR), a InfoNCE **nao** e aplicada
# diretamente sobre um embedding 2D — isso seria uma restricao severa
# demais. O protocolo padrao usa dois estagios:
#
# 1. **Encoder** $f_\theta$: imagem $\to$ representacao $\mathbf{h}$
#    (dimensao moderada, e.g. 64D)
# 2. **Cabeca de projecao** $g_\phi$: $\mathbf{h} \to \mathbf{z}$
#    (MLP pequeno; $\mathbf{z}$ e normalizado na esfera unitaria)
#
# A perda InfoNCE e computada sobre $\mathbf{z}$, mas **apos o treino
# descartamos a cabeca** e usamos $\mathbf{h}$ como representacao final.
#
# **Por que?** A cabeca absorve informacao especifica da tarefa pretexto
# (invariancias das augmentacoes), enquanto $\mathbf{h}$ retem mais
# informacao geral — e por isso transfere melhor para tarefas downstream
# (Chen et al. 2020, Sec. 4.2; ver tambem L3B1 §4.7).

# %%
# Encoder SSL reforçado: 3 camadas conv (R5 — mais capacidade para a tarefa contrastiva)

class EncoderSSL(eqx.Module):
    """CNN encoder reforcado para a tarefa contrastiva.

    Tres camadas convolucionais + FC -> representacao h.
    Mais capacidade que o Encoder da Parte 2, mas ainda leve para CPU.
    """
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc_out: eqx.nn.Linear

    def __init__(self, dim_h, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.conv1 = eqx.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, key=k2)
        self.conv3 = eqx.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, key=k3)
        # Apos conv3: (128, 4, 4) -> flatten -> 2048
        self.fc1 = eqx.nn.Linear(128 * 4 * 4, 128, key=k4)
        self.fc_out = eqx.nn.Linear(128, dim_h, key=k5)

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))    # (32, 14, 14)
        x = jax.nn.relu(self.conv2(x))    # (64, 7, 7)
        x = jax.nn.relu(self.conv3(x))    # (128, 4, 4)
        x = x.reshape(-1)                 # (2048,)
        x = jax.nn.relu(self.fc1(x))      # (128,)
        return self.fc_out(x)             # (dim_h,)


class CabecaProjecao(eqx.Module):
    """MLP de projecao: h -> z (normalizado L2).

    Chen et al. 2020 mostram que um MLP nao-linear aqui
    melhora substancialmente a qualidade da representacao.
    """
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim_h, dim_z, key):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(dim_h, dim_h, key=k1)
        self.fc2 = eqx.nn.Linear(dim_h, dim_z, key=k2)

    def __call__(self, h):
        z = jax.nn.relu(self.fc1(h))
        z = self.fc2(z)
        # Normalizacao L2 -> esfera unitaria
        return z / (jnp.linalg.norm(z) + 1e-8)


class ModeloContrastivo(eqx.Module):
    """Encoder + cabeca de projecao (SimCLR-style)."""
    encoder: EncoderSSL
    head: CabecaProjecao

    def __init__(self, dim_h, dim_z, key):
        k1, k2 = jax.random.split(key)
        self.encoder = EncoderSSL(dim_h, key=k1)
        self.head = CabecaProjecao(dim_h, dim_z, key=k2)

    def __call__(self, x):
        h = self.encoder(x)       # representacao (dim_h,)
        z = self.head(h)          # projecao normalizada (dim_z,)
        return z

    def representacao(self, x):
        """Retorna h (para avaliacao downstream — sem a cabeca)."""
        return self.encoder(x)


# Hiperparametros
DIM_H = 64   # dimensao da representacao
DIM_Z = 64   # dimensao da projecao (onde InfoNCE atua)

KEY, k_ssl = jax.random.split(KEY)
modelo_ssl = ModeloContrastivo(DIM_H, DIM_Z, key=k_ssl)

# Teste de shape
x_teste = jnp.zeros((1, 28, 28))
h_teste = modelo_ssl.representacao(x_teste)
z_teste = modelo_ssl(x_teste)
n_params_ssl = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(modelo_ssl, eqx.is_array)))
print(f"Encoder SSL: (1, 28, 28) -> h {h_teste.shape}  (representacao)")
print(f"Modelo completo: (1, 28, 28) -> z {z_teste.shape}  (projecao, norma L2 = {float(jnp.linalg.norm(z_teste)):.3f})")
print(f"Parametros totais: {n_params_ssl:,}")
print()
print("Apos o treino, DESCARTAMOS a cabeca de projecao.")
print("Usamos h (64D) como representacao — ela retem mais informacao geral.")

# %% [markdown]
# ### Regime de treino: batch grande = mais negativos
#
# *Referencia: L3B1 §4.5 — O papel do numero de negativos*
#
# A qualidade da InfoNCE escala com o **numero de negativos**:
# cada batch de $N$ amostras fornece $2(N-1)$ negativos por ancora.
# Com batch pequeno a tarefa e facil demais e o encoder nao
# precisa aprender representacoes finas.
#
# Por isso SimCLR usa batches de 4096-8192 (na GPU); **MoCo** e
# bancos de memoria foram inventados para ter muitos negativos
# sem precisar de batches tao grandes (L3B1 §4.5).
#
# Aqui, no nosso regime didatico em CPU, usamos batch=256
# (fornecendo $2 \times 255 = 510$ negativos por ancora) e
# compensamos com mais epocas.

# %%
# Perda InfoNCE — agora opera sobre o modelo completo (encoder + head)

TAU = 0.15  # temperatura (SimCLR usa ~0.07-0.1; 0.15 e estavel para nosso regime)
BATCH_SSL = 256  # batch grande -> mais negativos


def perda_infonce(modelo, x_vista1, x_vista2):
    """
    InfoNCE sobre a projecao z (L3B1 §4.4).

    x_vista1, x_vista2: duas vistas augmentadas do mesmo batch.
    Shape: (B, 1, 28, 28).
    """
    # -- Projecoes z (ja normalizadas L2 dentro do modelo) --
    z_i = jax.vmap(modelo)(x_vista1)                            # (B, dim_z)
    z_j = jax.vmap(modelo)(x_vista2)                            # (B, dim_z)
    B = z_i.shape[0]

    # -- Todas as representacoes: 2B embeddings --
    z = jnp.concatenate([z_i, z_j], axis=0)                     # (2B, dim_z)

    # -- Similaridade cosseno (z ja normalizado) --
    sim = z @ z.T                                                # (2B, 2B)

    # -- Logits = sim / tau --
    logits = sim / TAU                                           # (2B, 2B)

    # -- Excluir auto-similaridade: diagonal -> -inf --
    logits = jnp.where(~jnp.eye(2 * B, dtype=bool), logits, -1e9)

    # -- Denominador: "funcao de particao" --
    log_denom = jax.nn.logsumexp(logits, axis=1)                 # (2B,)

    # -- Numerador: sim(z_i, z_j)/tau para o par positivo --
    idx_pos = jnp.concatenate([jnp.arange(B, 2 * B), jnp.arange(B)])
    log_num = sim[jnp.arange(2 * B), idx_pos] / TAU              # (2B,)

    # -- InfoNCE: -E[ log(numerador / denominador) ] --
    return -jnp.mean(log_num - log_denom)


print(f"Perda InfoNCE definida.")
print(f"  Temperatura tau = {TAU} (SimCLR range)")
print(f"  Batch = {BATCH_SSL} -> 2x(B-1) = {2*(BATCH_SSL-1)} negativos por ancora")
print("  Nenhum rotulo de classe sera usado no treino!")

# %%
# Treino contrastivo (mais epocas que o supervisionado — InfoNCE converge mais devagar)

otimizador_ssl = optax.adam(learning_rate=1e-3)
opt_state_ssl = otimizador_ssl.init(eqx.filter(modelo_ssl, eqx.is_array))


@eqx.filter_jit
def passo_ssl(modelo, opt_state, x_v1, x_v2):
    """Um passo de treino contrastivo (InfoNCE)."""
    perda, grads = eqx.filter_value_and_grad(perda_infonce)(modelo, x_v1, x_v2)
    updates, opt_state = otimizador_ssl.update(grads, opt_state, modelo)
    modelo = eqx.apply_updates(modelo, updates)
    return modelo, opt_state, perda


N_EPOCAS_SSL = 80
historico_ssl = []
rng_ssl = np.random.default_rng(123)

print(f"Treinando com InfoNCE ({N_EPOCAS_SSL} epocas, SEM rotulos)...")
print(f"  Batch={BATCH_SSL}, tau={TAU}, lr=1e-3, dim_h={DIM_H}, dim_z={DIM_Z}")

for epoca in range(1, N_EPOCAS_SSL + 1):
    perm = rng_ssl.permutation(N_TRAIN)
    perdas_epoca = []

    for i in range(0, N_TRAIN, BATCH_SSL):
        idx = perm[i:i + BATCH_SSL]
        if len(idx) < 8:
            continue  # batch muito pequeno para InfoNCE
        X_batch_np = X_train[idx]
        # Duas vistas augmentadas (numpy) -> converter para JAX
        v1 = jnp.array(augmentar_batch(X_batch_np, rng_ssl)[:, None])
        v2 = jnp.array(augmentar_batch(X_batch_np, rng_ssl)[:, None])
        modelo_ssl, opt_state_ssl, perda = passo_ssl(
            modelo_ssl, opt_state_ssl, v1, v2
        )
        perdas_epoca.append(float(perda))

    media = np.mean(perdas_epoca)
    historico_ssl.append((epoca, media))
    if epoca % 20 == 0 or epoca == 1:
        print(f"  Epoca {epoca:2d}  perda InfoNCE = {media:.4f}")

print("Treino contrastivo concluido.")

# %%
# Curva de perda InfoNCE
ep_ssl, l_ssl = zip(*historico_ssl)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(ep_ssl, l_ssl, "-o", ms=2, lw=1.5, color="#8e44ad")
ax.set_xlabel("Epoca")
ax.set_ylabel("Perda InfoNCE")
ax.set_title(f"Treino contrastivo — SEM rotulos (tau={TAU}, batch={BATCH_SSL})")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualizar a representacao $h$ com t-SNE
#
# *Referencia: L3B1 §6 — Projecoes: visualizar espacos de alta dimensao*
#
# A representacao $\mathbf{h}$ tem 64 dimensoes — nao podemos plota-la
# diretamente. Usamos **t-SNE** para projetar em 2D e inspecionar se
# ha estrutura de clusters.
#
# **Advertencias importantes** (L3B1 §6.3):
# - Distancias **entre** clusters no mapa t-SNE sao largamente sem
#   significado — a compressao nao-linear distorce separacoes inter-grupo.
# - Tamanhos e densidades de clusters nao sao fieis.
# - O mapa muda com a perplexidade e a semente aleatoria.
#
# Usamos o t-SNE para **detectar se existe estrutura** (clusters
# separados por classe), nao como medicao quantitativa.

# %%
# Extrair representacoes h (64D) do conjunto de avaliacao (10k pontos)
from sklearn.manifold import TSNE

def obter_representacao_h(modelo, X_cnn, batch_sz=512):
    """Calcula representacao h (antes da cabeca) em lotes."""
    partes = []
    for i in range(0, len(X_cnn), batch_sz):
        partes.append(np.array(jax.vmap(modelo.representacao)(X_cnn[i:i + batch_sz])))
    return np.concatenate(partes)


H_eval = obter_representacao_h(modelo_ssl, X_eval_cnn)
print(f"Representacao h: shape = {H_eval.shape}")
print(f"  (10 000 pontos x 64 dimensoes — a cabeca de projecao foi descartada)")

# %%
# t-SNE com duas perplexidades diferentes (mostra sensibilidade ao parametro)
print("Computando t-SNE (pode demorar ~30s em CPU)...")

tsne_30 = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
H_tsne_30 = tsne_30.fit_transform(H_eval)

tsne_50 = TSNE(n_components=2, perplexity=50, random_state=42, init="pca")
H_tsne_50 = tsne_50.fit_transform(H_eval)

print("t-SNE concluido.")

# %%
# FIGURA CLIMAX: t-SNE da representacao h (duas perplexidades)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Representacao h (64D) projetada com t-SNE — InfoNCE SEM rotulos\n"
             "(10 000 pontos do teste, coloridos por classe)", fontsize=13)

for c in range(10):
    mask = y_eval == c
    ax1.scatter(H_tsne_30[mask, 0], H_tsne_30[mask, 1],
                s=4, color=CORES[c], alpha=0.5, edgecolors="none", label=str(c))
    ax2.scatter(H_tsne_50[mask, 0], H_tsne_50[mask, 1],
                s=4, color=CORES[c], alpha=0.5, edgecolors="none", label=str(c))

ax1.set_title("Perplexidade = 30", fontsize=11)
ax2.set_title("Perplexidade = 50", fontsize=11)
for ax in (ax1, ax2):
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(title="Digito\n(nao visto!)", fontsize=7, markerscale=2.5, framealpha=0.8)
    ax.grid(True, alpha=0.1)
plt.tight_layout()
plt.show()

print("O encoder NUNCA viu rotulos — mas a representacao h")
print("mostra agrupamentos claros por classe no t-SNE!")
print()
print("NOTA: distancias ENTRE clusters nao sao significativas (L3B1 §6.3).")
print("O t-SNE preserva vizinhancas locais, nao geometria global.")

# %%
# Sonda linear: medida QUANTITATIVA da qualidade da representacao h
from sklearn.linear_model import LogisticRegression

# Representacoes de treino (para ajustar a sonda)
H_train = obter_representacao_h(modelo_ssl, X_train_cnn)

# Regressao logistica sobre h CONGELADO (nao treina o encoder)
sonda = LogisticRegression(max_iter=2000, random_state=42)
sonda.fit(H_train, y_train)
acc_ssl = sonda.score(H_eval, y_eval)

# Para comparacao: acuracia do classificador supervisionado da Parte 2
preds_sup = []
for i in range(0, len(X_eval_cnn), 512):
    logits_batch = jax.vmap(classificador)(X_eval_cnn[i:i + 512])
    preds_sup.append(np.array(jnp.argmax(logits_batch, axis=-1)))
preds_sup = np.concatenate(preds_sup)
acc_sup = float(np.mean(preds_sup == y_eval))

print("=== Avaliacao da representacao ===")
print(f"  Supervisionado (Parte 2, bottleneck 2D):  acuracia = {acc_sup:.1%}")
print(f"  InfoNCE + sonda linear (h, 64D):          acuracia = {acc_ssl:.1%}")
print(f"  Acaso (10 classes):                        acuracia = 10.0%")
print()
print(f"A sonda linear sobre h congelado atinge {acc_ssl:.1%}!")
print("O encoder aprendeu estrutura rica SEM nenhum rotulo de classe.")

# %% [markdown]
# ### Comparacao visual: supervisionado 2D vs InfoNCE t-SNE
#
# A Parte 2 usou rotulos para forcar a estrutura em 2D.
# A Parte 4 descobriu estrutura **sem rotulos**, numa representacao
# 64D que projetamos com t-SNE para visualizar.

# %%
# Lado a lado: supervisionado 2D vs InfoNCE t-SNE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Supervisionado (2D) vs Auto-supervisionado (64D, t-SNE) — 10 000 pontos", fontsize=13)

for c in range(10):
    mask = y_eval == c
    ax1.scatter(Z_sup[mask, 0], Z_sup[mask, 1],
                s=4, color=CORES[c], alpha=0.5, edgecolors="none", label=str(c))
    ax2.scatter(H_tsne_30[mask, 0], H_tsne_30[mask, 1],
                s=4, color=CORES[c], alpha=0.5, edgecolors="none", label=str(c))

ax1.set_title("Parte 2: COM rotulos\n(classificador supervisionado, 2D direto)", fontsize=11)
ax2.set_title("Parte 4: SEM rotulos\n(InfoNCE, h 64D projetado via t-SNE)", fontsize=11)
for ax in (ax1, ax2):
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(title="Digito", fontsize=7, markerscale=2.5, framealpha=0.8)
    ax.grid(True, alpha=0.1)
plt.tight_layout()
plt.show()

print("Ambos revelam estrutura de clusters — mas foram obtidos de formas muito diferentes:")
print("  - Esquerda: rotulos disseram quem e quem (supervisionado)")
print("  - Direita: so augmentacoes declararam 'estas vistas sao do mesmo objeto'")

# %% [markdown]
# **Nota honesta** (no espirito da L3B1):
#
# - **A cabeca de projecao e essencial.** Aplicar InfoNCE direto em 2D
#   (como seria a abordagem ingenua) produz representacoes pobres
#   porque 2 dimensoes nao oferecem capacidade suficiente para a
#   tarefa contrastiva. A separacao encoder/cabeca (Chen et al. 2020)
#   resolve isso — a cabeca absorve as invariancias e a representacao
#   $\mathbf{h}$ retem informacao geral.
#
# - **O modelo nunca viu rotulos.** A unica informacao que guiou o
#   treino foram as augmentacoes — "estas duas vistas sao do mesmo
#   objeto." Que estrutura emerja dessas declaracoes e notavel.
#
# - **A temperatura $\tau$** controla a nitidez dos contrastes
#   (L3B1 §4.4): $\tau$ baixo foca nos negativos mais dificeis;
#   $\tau$ alto suaviza. Experimente mudar `TAU` e re-treinar!
#
# - **Batch grande importa** (L3B1 §4.5): mais negativos = tarefa
#   mais dificil = representacoes mais finas. SimCLR e MoCo foram
#   desenhados exatamente para maximizar o numero de negativos.

# %% [markdown]
# ---
# ## Fechamento
#
# | Parte | O que fizemos | A mensagem |
# |-------|---------------|-----------|
# | 1 | Distancias de pixels | Coordenadas brutas mentem |
# | 2 | CNN $\to$ 2D com rotulos | Encoder cria espaco com estrutura |
# | 3 | Augmentacoes | Cada uma e uma invariancia declarada |
# | 4 | InfoNCE sem rotulos | Estrutura emerge sem supervisao |
#
# A transicao 1 $\to$ 2 $\to$ 4 e o arco central: saimos de um espaco
# onde distancia nao significa nada (pixels) para um espaco onde
# distancia reflete similaridade — primeiro com rotulos, depois sem.
#
# **Ingredientes criticos da Parte 4:**
# - Encoder com capacidade suficiente (3 conv layers, h 64D)
# - Cabeca de projecao descartavel (InfoNCE atua sobre z, nao h)
# - Augmentacoes fortes (deformacao elastica + afim + cutout)
# - Batch grande (510 negativos por ancora)
# - Temperatura adequada ($\tau = 0.15$)
#
# **Ponte para o Dia 4:** amanha veremos este mesmo maquinario
# aplicado a dados cientificos reais. O encoder aprende um espaco
# onde halos de materia escura ou quasares similares ficam proximos,
# e clustering nesse espaco revela estrutura que nenhum parametro
# analitico capturava.

# %% [markdown]
# ## Para casa
#
# **Exercicio 1 — Efeito da temperatura**
#
# Mude `TAU` para 0.05, 0.15 e 0.5. Re-treine o encoder InfoNCE
# para cada valor e compare os t-SNEs. Para qual temperatura
# os clusters sao mais nitidos? (Dica: L3B1 §4.4)
#
# ---
#
# **Exercicio 2 — Dimensao da representacao**
#
# Troque `DIM_H` para 16 e depois para 128. Re-treine e avalie
# com a sonda linear. A acuracia melhora com mais dimensoes?
# Ha rendimentos decrescentes?
#
# ---
#
# **Exercicio 3 — Quais augmentacoes importam?**
#
# Remova uma augmentacao por vez (so ruido, so translacao, so rotacao)
# e re-treine. Qual augmentacao contribui mais para a qualidade?
#
# ---
#
# **Exercicio 4 (desafio) — Sem cabeca de projecao**
#
# Remova a `CabecaProjecao` e aplique InfoNCE diretamente sobre $h$.
# Compare a acuracia da sonda linear. A separacao encoder/cabeca
# realmente ajuda? (Spoiler: Chen et al. 2020 dizem que sim.)
