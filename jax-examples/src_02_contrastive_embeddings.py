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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/02_contrastive_embeddings.ipynb)
#
# # 🟢 Notebook 02 — Embeddings Contrastivos: Esculpindo Espaços
# ### Em três atos — de partículas a dígitos a halos
# **I Escola de Inverno do IFUSP — Bloco L03_B02**
#
# > **Modo de uso:** demo guiada pelo instrutor; você recebe o notebook depois.
# > Objetivo: entender aprendizado contrastivo vivenciando-o em três escalas —
# > como um potencial físico (Ato 1), como um encoder de imagens (Ato 2),
# > e como uma ferramenta de descoberta sem rótulos (Ato 3).
#
# ---
#
# ### 🗺️ Mapa do Curso
#
# | Dia | Bloco | Tema |
# |-----|-------|------|
# | Ter. 21/07 | L01_B01 | ML e Física: o mapa do território |
# | Ter. 21/07 | L01_B02 | A caixa de ferramentas |
# | Qua. 22/07 | L02_B01 | Domain shift: teoria |
# | Qua. 22/07 | L02_B02 | Mão na massa I — quebrar e consertar um classificador |
# | Qui. 23/07 | L03_B01 | Aprendizado contrastivo: teoria |
# | **Qui. 23/07** | **L03_B02 ← você está aqui** | **Mão na massa II — esculpindo embeddings** |
# | Sex. 24/07 | L04_B01 | Halos de matéria escura com segmentação de instâncias |
# | Sex. 24/07 | L04_B02 | J-PAS: domain adaptation com quasares reais |

# %% [markdown]
# ## 🟢 A Tese: Perda Contrastiva = Potencial de Interação
#
# Este notebook é uma peça em **três atos**, todos conectados pela mesma ideia:
#
# > *Uma boa função de perda contrastiva é um potencial de interação:
# > ela atrai objetos semelhantes e repele objetos diferentes.*
#
# ### Os três atos:
#
# **Ato 1 — Sandbox de partículas** (sem rede neural):
# 200 "partículas" 2D, com rótulos de grupo. Otimizamos as posições
# diretamente. O resultado parece simulação de dinâmica molecular.
#
# **Ato 2 — Encoder real** (dados MNIST):
# As "partículas" do Ato 1 são substituídas pelo output de um encoder MLP.
# A **perda é a mesma** — a única diferença é que os pontos agora vêm de
# imagens reais. O encoder *aprende* a representar as imagens.
#
# **Ato 3 — Colher o espaço** (clustering + projeções):
# Com o espaço de embeddings organizado, aplicamos k-means para recuperar
# as classes **sem usar rótulos na inferência**. Isso É segmentação de
# instâncias. Depois, inspecionamos o espaço latente com t-SNE e UMAP.
#
# > **PRETRAINED = True** (padrão): se os assets pré-computados estiverem
# > disponíveis, são carregados. Se não estiverem (execução limpa no Colab),
# > o notebook baixa o MNIST e computa tudo do zero automaticamente.

# %%
# ── Setup: importações e configuração global ──────────────────────────────────
import os
import gzip
import struct
import pickle
import urllib.request
import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE

# Colab: instale umap-learn se necessário (guarda local já tem as dependências)
try:
    import umap
    UMAP_DISPONIVEL = True
except ImportError:
    UMAP_DISPONIVEL = False

# ── Flag global: True = carrega checkpoints pré-computados quando disponíveis
PRETRAINED = True

# ── Semente de reprodutibilidade
SEMENTE = 42
CHAVE   = jax.random.PRNGKey(SEMENTE)

# ── Caminho dos assets (gitignored — nunca versionado)
try:
    ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
except NameError:  # Jupyter/Colab: __file__ não existe; usa caminho relativo ao CWD
    ASSETS = pathlib.Path("../assets")

# Garante que o diretório assets/ existe
ASSETS.mkdir(parents=True, exist_ok=True)

# ── Hiperparâmetros do sandbox (Ato 1)
N_GRUPOS      = 5      # grupos de partículas
N_POR_GRUPO   = 40     # partículas por grupo
DELTA_PULL    = 0.5    # raio de atração (pontos devem estar a < δ_pull do centro)
DELTA_PUSH    = 1.5    # margem de repulsão (centros devem estar a > δ_push)
LAMBDA_REG    = 0.01   # peso do regularizador (âncora dos centros à origem)
ETA_SANDBOX   = 0.05   # passo do gradiente descendente no sandbox
N_PASSOS      = 800    # passos de otimização para o sandbox

# ── Hiperparâmetros do encoder MNIST (Ato 2)
N_CLASSES     = 10
DIM_2D        = 2
DIM_16D       = 16
DP_PULL_ENC   = 0.3
DP_PUSH_ENC   = 2.5
LAMBDA_ENC    = 0.001
LR_ENCODER    = 2e-3
BATCH_SZ      = 500
EPOCAS_2D     = 250

# ── Paleta de cores (10 classes MNIST)
CORES_10 = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#c0392b", "#27ae60",
]

# ── Paleta de cores (5 grupos sandbox)
CORES_5 = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

# ── Estilo global
plt.rcParams.update({
    "figure.dpi"    : 110,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
})

print(f"JAX versão : {jax.__version__}")
print(f"Dispositivo: {jax.devices()[0]}")
print(f"PRETRAINED : {PRETRAINED}")
print(f"Assets em  : {ASSETS.resolve()}")
print(f"UMAP       : {'disponível' if UMAP_DISPONIVEL else 'não instalado (use o Colab)'}")

# %% [markdown]
# ---
# ## 🟢 ATO 1 — Sandbox de Partículas (Sem Rede Neural)
#
# Começamos **sem nenhuma rede neural**. Temos 200 pontos 2D, divididos em
# 5 grupos (rótulos de instância). Queremos aprender um arranjo espacial onde:
#
# - Membros do **mesmo grupo se atraem** (ficam próximos do centro do grupo).
# - Grupos **diferentes se repelem** (centros de grupos distintos ficam afastados).
# - Os centros ficam **ancorados** perto da origem (regularizador).
#
# ### A perda de Weinberger (discriminativa)
#
# $$L = \underbrace{L_\text{pull}}_{\text{atração}} + \underbrace{L_\text{push}}_{\text{repulsão}} + \lambda \underbrace{L_\text{reg}}_{\text{âncora}}$$
#
# $$L_\text{pull} = \frac{1}{N}\sum_i \bigl[\max\!\bigl(\|x_i - \mu_{c(i)}\| - \delta_\text{pull},\,0\bigr)\bigr]^2$$
#
# $$L_\text{push} = \frac{1}{\binom{K}{2}}\sum_{c \neq c'} \bigl[\max\!\bigl(\delta_\text{push} - \|\mu_c - \mu_{c'}\|,\,0\bigr)\bigr]^2$$
#
# onde $\mu_c$ é o **centro do grupo** $c$ (média das posições dos membros).
#
# > **Chave:** $L_\text{pull}$ puxa pontos para dentro de uma bola de raio
# > $\delta_\text{pull}$ ao redor do seu centro; $L_\text{push}$ empurra
# > pares de centros para fora de uma margem $\delta_\text{push}$.

# %%
# ── ATO 1.1 — Gerar partículas 2D ─────────────────────────────────────────────
#
# Grupos começam MISTURADOS perto da origem (com um pequeno deslocamento de
# simetria), para que a separação seja visível durante a otimização.

def gerar_particulas(n_grupos, n_por_grupo, chave):
    """
    Posições iniciais: grupos sobrepostos perto da origem com pequeno
    deslocamento circular (0.4 de raio) para quebrar a simetria.
    """
    angulos  = jnp.linspace(0, 2 * jnp.pi, n_grupos, endpoint=False)
    centros  = 0.4 * jnp.stack([jnp.cos(angulos), jnp.sin(angulos)], axis=1)
    pts_list = []
    for c in range(n_grupos):
        chave, kc = jax.random.split(chave)
        ruido = jax.random.normal(kc, (n_por_grupo, 2)) * 0.9
        pts_list.append(ruido + centros[c])
    posicoes = jnp.concatenate(pts_list, axis=0)
    rotulos  = jnp.repeat(jnp.arange(n_grupos), n_por_grupo)
    return np.array(posicoes, dtype=np.float32), np.array(rotulos, dtype=np.int32)


# Padrão "gera se ausente": carrega de assets/ ou gera inline
_path_init = ASSETS / "nb2_sandbox_initial.npz"
if _path_init.exists():
    _d = np.load(_path_init)
    x_init_np  = _d["positions"]
    rotulos_np = _d["labels"]
    print("Posições iniciais carregadas de assets/.")
else:
    CHAVE, kp = jax.random.split(CHAVE)
    x_init_np, rotulos_np = gerar_particulas(N_GRUPOS, N_POR_GRUPO, kp)
    np.savez(str(_path_init), positions=x_init_np, labels=rotulos_np)
    print("Posições iniciais geradas e salvas em assets/.")

print(f"Partículas: {x_init_np.shape}  rótulos: {rotulos_np.shape}")
print(f"Grupos    : {N_GRUPOS}  Partículas/grupo: {N_POR_GRUPO}")

# Plota estado inicial
fig, ax = plt.subplots(figsize=(5, 5))
for c in range(N_GRUPOS):
    mask = rotulos_np == c
    ax.scatter(x_init_np[mask, 0], x_init_np[mask, 1],
               s=20, color=CORES_5[c], alpha=0.8, edgecolors="none",
               label=f"Grupo {c}")
ax.set_title("Estado inicial — grupos misturados", fontsize=12)
ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
plt.tight_layout(); plt.show()
print("Grupos começam sobrepostos — não há estrutura clara ainda.")

# %% [markdown]
# ## 🟢 ATO 1.2 — A Perda Contrastiva em JAX
#
# Implementamos as três perdas em JAX puro. Pontos-chave:
#
# 1. **Centros de cluster** = média das posições por grupo (diferenciável).
# 2. **`jax.grad`** calcula o gradiente da perda w.r.t. as posições.
# 3. A mesma função `weinberger_loss` será usada no Ato 2 — só mudamos
#    o que são os "pontos" (posições diretas → saída do encoder).

# %%
# ── ATO 1.2 — Funções de perda (JAX) ──────────────────────────────────────────

def fazer_weinberger_loss(n_classes):
    """
    Factory: retorna weinberger_loss com n_classes fixado como int Python
    (necessário para JIT — n_classes não pode ser valor rastreado pelo JAX).
    """
    _nc = int(n_classes)

    def _loss(x, rotulos, delta_pull, delta_push, lambda_reg):
        # Centros via agregação one-hot (compatível com JIT)
        one_hot  = jax.nn.one_hot(rotulos, _nc)              # (N, K)
        contagem = one_hot.sum(axis=0)                         # (K,)
        centros  = (one_hot.T @ x) / jnp.maximum(contagem[:, None], 1.0)

        # Pull: cada ponto dentro de delta_pull do centro do seu grupo
        c_por_pt = centros[rotulos]
        dist_pt  = jnp.sqrt(jnp.sum((x - c_por_pt)**2, axis=-1) + 1e-8)
        L_pull   = jnp.mean(jnp.maximum(dist_pt - delta_pull, 0.0)**2)

        # Push: cada par de centros distintos afastados por delta_push
        ci        = centros[:, None, :]
        cj        = centros[None, :, :]
        dist_cc   = jnp.sqrt(jnp.sum((ci - cj)**2, axis=-1) + 1e-8)
        viol_push = jnp.maximum(delta_push - dist_cc, 0.0)**2
        fora_diag = 1.0 - jnp.eye(_nc)
        L_push    = (jnp.sum(fora_diag * viol_push)
                     / jnp.maximum(fora_diag.sum(), 1.0))

        # Regularizador: centros próximos da origem
        L_reg = jnp.mean(jnp.sum(centros**2, axis=-1))

        return L_pull + L_push + lambda_reg * L_reg

    return _loss


# Cria perda específica para o sandbox (5 grupos)
perda_sandbox  = fazer_weinberger_loss(N_GRUPOS)
grad_sandbox   = jax.jit(jax.grad(perda_sandbox, argnums=0))

# Teste rápido de shapes
x_teste   = jnp.ones((N_GRUPOS * N_POR_GRUPO, 2))
r_teste   = jnp.repeat(jnp.arange(N_GRUPOS), N_POR_GRUPO)
l_teste   = perda_sandbox(x_teste, r_teste, DELTA_PULL, DELTA_PUSH, LAMBDA_REG)
print(f"Perda de teste (shapes OK): {float(l_teste):.4f}")
print(f"Gradiente shape: {grad_sandbox(x_teste, r_teste, DELTA_PULL, DELTA_PUSH, LAMBDA_REG).shape}")

# %% [markdown]
# ## 🟡 Poll — Ato 1: O que acontece com δ_push = 0?
#
# **Pense antes de rodar:**
#
# > Se removermos a repulsão ($\delta_\text{push} = 0$), o que acontece
# > quando rodamos a otimização?
# >
# > (a) Os grupos se **separam** mesmo assim, graças ao pull.
# > (b) Os grupos **colapsam** num único ponto — o mínimo trivial.
# > (c) Nada muda — a perda já é zero.
#
# **Levante a mão** com sua resposta antes de rodar a próxima célula.

# %%
# ── ATO 1.3 — Colapso trivial (δ_push = 0) ────────────────────────────────────
#
# Sem repulsão, o mínimo da perda é trivial: todos os grupos no mesmo ponto.
# A regularização puxa todos os centros para a origem → colapso.
#
# ATENÇÃO: para demonstrar o colapso usamos lambda_reg=10.0 (forte),
# que faz com que a regularização domine sobre o pull.
#
# Padrão "gera se ausente": carrega de assets/ se disponível, caso contrário
# computa inline (alguns segundos com JAX JIT) e salva para reutilização.

LAMBDA_COLAPSO = 10.0   # forte o suficiente para forçar colapso à origem

_path_col = ASSETS / "nb2_sandbox_collapsed.npz"
if PRETRAINED and _path_col.exists():
    _dc = np.load(_path_col)
    x_colapsado = _dc["positions"]
    print("Posições colapsadas carregadas dos assets.")
else:
    # Computa ao vivo — também é o fallback para execução limpa
    print("Rodando colapso ao vivo (δ_push=0, lambda_reg=10.0)...")
    x_col = jnp.array(x_init_np)
    lb_j  = jnp.array(rotulos_np)
    for passo in range(600):
        g = grad_sandbox(x_col, lb_j, DELTA_PULL, 0.0, LAMBDA_COLAPSO)
        x_col = x_col - ETA_SANDBOX * g
    x_colapsado = np.array(x_col)
    np.savez(str(_path_col), positions=x_colapsado, labels=rotulos_np)
    print("Colapso computado e salvo em assets/.")

# Verificação: centros devem estar próximos
centros_col = np.stack([x_colapsado[rotulos_np == c].mean(axis=0)
                         for c in range(N_GRUPOS)])
dists_col   = [np.linalg.norm(centros_col[i] - centros_col[j])
               for i in range(N_GRUPOS) for j in range(i+1, N_GRUPOS)]
print(f"\nDistância máx. entre centros: {max(dists_col):.3f}"
      f"  (< {DELTA_PULL} → todos no mesmo ponto!)")

# Figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Colapso trivial — sem repulsão (δ_push = 0)", fontsize=12)

for c in range(N_GRUPOS):
    mask = rotulos_np == c
    ax1.scatter(x_init_np[mask, 0], x_init_np[mask, 1],
                s=18, color=CORES_5[c], alpha=0.8, edgecolors="none",
                label=f"Grupo {c}")
ax1.set_title("Estado inicial"); ax1.set_xlabel("x₁"); ax1.set_ylabel("x₂")
ax1.legend(fontsize=7); ax1.grid(True, alpha=0.2)
ax1.set_xlim(-4, 4); ax1.set_ylim(-4, 4)

for c in range(N_GRUPOS):
    mask = rotulos_np == c
    ax2.scatter(x_colapsado[mask, 0], x_colapsado[mask, 1],
                s=18, color=CORES_5[c], alpha=0.8, edgecolors="none",
                label=f"Grupo {c}")
ax2.set_title("Após δ_push = 0 → colapso!"); ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")
ax2.legend(fontsize=7); ax2.grid(True, alpha=0.2)
ax2.set_xlim(-4, 4); ax2.set_ylim(-4, 4)

plt.tight_layout(); plt.show()
print("\n→ Resposta: (b) colapso trivial — sem repulsão, tudo converge para um só ponto.")
print("  O mínimo global de L_pull + L_reg (sem L_push) é todo mundo na origem.")

# %%
# ── ATO 1.4 — Relaxação com δ_push > 0 ──────────────────────────────────────
#
# Com a repulsão ativada, os grupos não podem colapsar — a perda os empurra
# para fora até atingirem um equilíbrio: grupos tight + centros afastados.
#
# Padrão "gera se ausente": carrega de assets/ se disponível, caso contrário
# computa inline e salva.

_path_fin = ASSETS / "nb2_sandbox_final.npz"
if PRETRAINED and _path_fin.exists():
    _df    = np.load(_path_fin)
    x_final = _df["positions"]
    snaps   = _df["snapshots"]   # (3, N, 2): inicial, meio, final
    print("Sandbox final carregado dos assets (δ_push > 0).")
else:
    # Otimização ao vivo
    print("Rodando sandbox ao vivo (δ_push=1.5, lambda_reg=0.01)...")
    x_sep      = jnp.array(x_init_np)
    lb_j       = jnp.array(rotulos_np)
    snaps_list = [np.array(x_sep)]
    mid_step   = N_PASSOS // 2
    for passo in range(N_PASSOS):
        g = grad_sandbox(x_sep, lb_j, DELTA_PULL, DELTA_PUSH, LAMBDA_REG)
        x_sep = x_sep - ETA_SANDBOX * g
        if passo == mid_step - 1:
            snaps_list.append(np.array(x_sep))
    x_final = np.array(x_sep)
    snaps_list.append(x_final)
    snaps = np.stack(snaps_list)
    np.savez(str(_path_fin), positions=x_final, labels=rotulos_np, snapshots=snaps)
    print("Sandbox computado e salvo em assets/.")

# Filmstrip: initial / mid / final — sempre plota diretamente dos dados
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Sandbox pull/push — δ_push = 1.5 (repulsão ativada)", fontsize=12)
titulos = ["Estado inicial\n(misturado)",
           "Intermediário\n(começa a separar)",
           "Estado final\n(grupos distintos)"]
for ax, snap, titulo in zip(axes, snaps, titulos):
    lim_plot = max(abs(snap).max() * 1.2, 3.5)
    for c in range(N_GRUPOS):
        mask = rotulos_np == c
        ax.scatter(snap[mask, 0], snap[mask, 1],
                   s=20, color=CORES_5[c], alpha=0.85,
                   edgecolors="none", label=f"Grupo {c}")
    # Centros com X
    ctrs = np.stack([snap[rotulos_np == c].mean(0) for c in range(N_GRUPOS)])
    for c in range(N_GRUPOS):
        ax.scatter(ctrs[c, 0], ctrs[c, 1], s=80, color=CORES_5[c],
                   marker="X", edgecolors="k", linewidths=0.8, zorder=5)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_xlim(-lim_plot, lim_plot); ax.set_ylim(-lim_plot, lim_plot)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.show()

# Métricas finais
centros_fin = np.stack([x_final[rotulos_np == c].mean(axis=0) for c in range(N_GRUPOS)])
dists_fin   = [np.linalg.norm(centros_fin[i] - centros_fin[j])
               for i in range(N_GRUPOS) for j in range(i+1, N_GRUPOS)]
print(f"\nDistância mínima entre centros (final): {min(dists_fin):.3f}")
print(f"δ_push configurado: {DELTA_PUSH}")
print("→ Grupos separados! A repulsão cria uma geometria com estrutura.")

# %% [markdown]
# ## 🟣 (Opcional) Versão com Temperatura — InfoNCE
#
# O potencial push-hard pode ser suavizado com um softmax de temperatura,
# dando origem à perda InfoNCE usada no SimCLR e no MoCo:
#
# $$L_\text{InfoNCE} = -\log \frac{e^{-d_+/T}}{\sum_{c' \neq c} e^{-d_{c'}/T}}$$
#
# onde $d_+ = \|\mu_c - \mu_{c(i)}\|$ é a distância positiva e o somatório
# percorre todos os outros centros.
#
# **Propriedade interessante:** para $T \to 0$, o softmax concentra toda a
# massa no negativo mais próximo → comportamento "hard-negative" = push.

# %%
# ── (Opcional) Perda InfoNCE com temperatura ──────────────────────────────────

def perda_infonce_sandbox(x, rotulos, temperatura=0.5):
    """
    Versão suave da repulsão: softmax sobre distâncias a centros negativos.
    n_classes fixado em N_GRUPOS via closure (necessário para JIT).
    """
    _nc = int(N_GRUPOS)
    one_hot = jax.nn.one_hot(rotulos, _nc)
    contagem = one_hot.sum(axis=0)
    centros  = (one_hot.T @ x) / jnp.maximum(contagem[:, None], 1.0)

    # Para cada amostra, distância ao centro positivo vs. centros negativos
    c_pos   = centros[rotulos]                                   # (N, 2)
    d_pos   = jnp.sqrt(jnp.sum((x - c_pos)**2, axis=-1) + 1e-8)

    # Distâncias a todos os centros (N, K)
    diffs   = x[:, None, :] - centros[None, :, :]               # (N, K, 2)
    d_all   = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)       # (N, K)

    # Log-probabilidade InfoNCE
    logits  = -d_all / temperatura                               # (N, K)
    log_sum = jax.nn.logsumexp(logits, axis=-1)                  # (N,)
    log_num = -d_pos / temperatura                               # (N,)

    return -jnp.mean(log_num - log_sum)


print("Perda InfoNCE definida.")
print("Para rodar: chame perda_infonce_sandbox(x, rotulos, temperatura=0.5)")
print("Experimento sugerido: compare T=0.1, T=0.5, T=2.0 com a versão pull/push.")

# %% [markdown]
# ---
# ## 🟢 ATO 2 — Encoder Real: MNIST (Dados Públicos)
#
# **A transição fundamental:**
#
# - **Ato 1:** os "pontos" $x_i$ eram posições 2D otimizadas *diretamente*.
# - **Ato 2:** os "pontos" $x_i$ são o **output de um encoder** MLP que
#   mapeia imagens MNIST (28×28 = 784 dimensões) para um espaço 2D.
#
# A **perda é idêntica** à do Ato 1. A única diferença:
# em vez de otimizar as posições diretamente, otimizamos os **pesos do encoder**
# para que seus outputs satisfaçam as restrições pull/push.
#
# **Rótulos como geradores de pares:** os rótulos de classe (0–9) definem
# quais imagens devem estar próximas (mesmo dígito = mesmo grupo).
#
# > A diferença entre Ato 1 e Ato 2 é a diferença entre simulação molecular
# > e aprendizado de representações — **a matemática é a mesma**.

# %%
# ── ATO 2.1 — Carregar subset MNIST ───────────────────────────────────────────
#
# Padrão "gera se ausente": se mnist_4k.npz não existe em assets/, baixa os
# arquivos brutos do MNIST (Google CVDF storage, ~12 MB) e cria um subset
# balanceado de 4000 treino + 1000 teste.


def _parse_mnist_gz(fpath, kind="images"):
    """Lê um arquivo IDX gzipado do MNIST."""
    with gzip.open(str(fpath), "rb") as f:
        raw = f.read()
    if kind == "images":
        _, n, rows, cols = struct.unpack(">IIII", raw[:16])
        data = np.frombuffer(raw[16:], dtype=np.uint8).reshape(n, rows * cols)
        return data.astype(np.float32) / 255.0
    else:
        _, n = struct.unpack(">II", raw[:8])
        return np.frombuffer(raw[8:], dtype=np.uint8).astype(np.int32)


def _baixar_e_preparar_mnist(assets_dir, n_train=4000, n_test=1000, seed=42):
    """
    Baixa os 4 arquivos gz do MNIST (se não estiverem em cache) e cria
    um subset balanceado. Salva em assets/mnist_4k.npz.
    """
    url_base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    fnames = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    raw_dir = assets_dir / "_mnist_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for key, fname in fnames.items():
        local = raw_dir / fname
        if not local.exists():
            print(f"  Baixando {fname} ...")
            urllib.request.urlretrieve(url_base + fname, str(local))
            print(f"  OK ({local.stat().st_size // 1024} KB)")
        else:
            print(f"  {fname} em cache.")
        paths[key] = local

    rng      = np.random.default_rng(seed)
    X_tr_all = _parse_mnist_gz(paths["train_images"], "images")
    y_tr_all = _parse_mnist_gz(paths["train_labels"], "labels")
    X_te_all = _parse_mnist_gz(paths["test_images"],  "images")
    y_te_all = _parse_mnist_gz(paths["test_labels"],  "labels")

    ntr = n_train // 10
    nte = n_test  // 10
    tr_idx, te_idx = [], []
    for c in range(10):
        idx = np.where(y_tr_all == c)[0]
        tr_idx.append(rng.choice(idx, ntr, replace=False))
        idx_te = np.where(y_te_all == c)[0]
        te_idx.append(rng.choice(idx_te, nte, replace=False))
    tr_idx = np.concatenate(tr_idx); rng.shuffle(tr_idx)
    te_idx = np.concatenate(te_idx); rng.shuffle(te_idx)

    out_path = assets_dir / "mnist_4k.npz"
    np.savez(str(out_path),
             X_train=X_tr_all[tr_idx], y_train=y_tr_all[tr_idx],
             X_test=X_te_all[te_idx],  y_test=y_te_all[te_idx])
    print(f"  Salvo: {out_path.name} ({out_path.stat().st_size // 1024} KB)")


def carregar_mnist(caminho):
    """Carrega subset MNIST do arquivo .npz. Retorna (X_train, y_train, X_test, y_test)."""
    d = np.load(caminho)
    return (d["X_train"].astype(np.float32),
            d["y_train"].astype(np.int32),
            d["X_test"].astype(np.float32),
            d["y_test"].astype(np.int32))


# Gera se ausente (suporta execução limpa no Colab sem assets pré-existentes)
_mnist_path = ASSETS / "mnist_4k.npz"
if not _mnist_path.exists():
    print("mnist_4k.npz não encontrado — baixando MNIST e criando subset...")
    _baixar_e_preparar_mnist(ASSETS)
    print("MNIST pronto.")

X_train, y_train, X_test, y_test = carregar_mnist(_mnist_path)

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")
print(f"Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Amostras por classe (treino): {[int((y_train == c).sum()) for c in range(N_CLASSES)]}")

# Grade de exemplos 5x10
fig, axes = plt.subplots(5, 10, figsize=(12, 6))
fig.suptitle("Subset MNIST — 5 amostras por dígito (0–9)", fontsize=12)
for c in range(10):
    idx_c = np.where(y_test == c)[0][:5]
    for row, idx in enumerate(idx_c):
        ax = axes[row, c]
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray_r", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(str(c), fontsize=10, color=CORES_10[c], fontweight="bold")
plt.tight_layout(); plt.show()
print("Cada coluna = um dígito. Note os parecidos: 4/9, 3/8, 5/6...")

# %% [markdown]
# ## 🟣 (Opcional) Variante Galaxy10 — Dados Astronômicos

# %%
# ── (Opcional) Galaxy10 ────────────────────────────────────────────────────────

galaxy10_path = ASSETS / "galaxy10_1k.npz"
if galaxy10_path.exists():
    d_gal = np.load(galaxy10_path)
    X_gal = d_gal["X"].astype(np.float32)
    y_gal = d_gal["y"].astype(np.int32)
    print(f"Galaxy10: X={X_gal.shape}, y={y_gal.shape}")
    print("Para usar com o encoder: substitua X_train/X_test por X_gal.")
else:
    print("Galaxy10 não disponível neste ambiente.")
    print("Os dados astronômicos requerem download separado (~50 MB).")
    print("Dica: execute no Colab com GPU para a variante Galaxy10.")

# %% [markdown]
# ## 🟢 ATO 2.2 — Encoder MLP: 784 → 256 → 64 → 2
#
# A arquitetura do encoder mapeia uma imagem MNIST achatada para um ponto 2D:
#
# | Camada | Entrada | Saída | Ativação |
# |--------|---------|-------|----------|
# | FC 1   | 784     | 256   | ReLU     |
# | FC 2   | 256     | 64    | ReLU     |
# | FC 3   | 64      | 2     | linear   |
#
# Usamos ReLU (em vez de tanh) pois é mais adequado para dados de imagem.
# A saída linear permite que o espaço latente escale livremente.

# %%
# ── ATO 2.2 — Definição do encoder MLP ───────────────────────────────────────

def iniciar_encoder(tamanhos_camadas, chave):
    """
    Inicialização He-normal (para ReLU).
    Retorna lista de tuplas (W, b), uma por camada.
    (Mesmo padrão de params do Notebook 00 e 01.)
    """
    params = []
    for i in range(len(tamanhos_camadas) - 1):
        chave, kw = jax.random.split(chave)
        entrada = tamanhos_camadas[i]
        saida   = tamanhos_camadas[i + 1]
        W = jax.random.normal(kw, (entrada, saida)) * np.sqrt(2.0 / entrada)
        b = jnp.zeros(saida)
        params.append((W, b))
    return params


def forward_encoder(params, x):
    """ReLU nas camadas ocultas, saída linear."""
    h = x
    for W, b in params[:-1]:
        h = jax.nn.relu(h @ W + b)
    W, b = params[-1]
    return h @ W + b


# Teste de shape: uma imagem → ponto 2D
CHAVE, ke = jax.random.split(CHAVE)
enc_teste_2d = iniciar_encoder([784, 256, 64, DIM_2D], ke)
x_img_teste  = jnp.ones((3, 784))
z_teste      = forward_encoder(enc_teste_2d, x_img_teste)
print(f"Encoder [784→256→64→2]: entrada {x_img_teste.shape} → embedding {z_teste.shape}")
print(f"Parâmetros totais: {sum(W.size + b.size for W, b in enc_teste_2d):,}")

# %% [markdown]
# ## 🟢 ATO 2.3 — A Mesma Perda, Agora com Encoder
#
# A função `perda_discriminativa` é **idêntica** em estrutura à `perda_sandbox`
# do Ato 1. A única diferença: $x_i = \text{encoder}(\text{imagem}_i)$
# em vez de ser uma posição diretamente otimizável.
#
# > "A perda não sabe que está olhando para pixels —
# > ela só vê pontos em $\mathbb{R}^2$ e empurra/puxa."

# %%
# ── ATO 2.3 — Perda discriminativa com encoder ───────────────────────────────

# Cria perda de Weinberger específica para 10 classes (MNIST)
# (n_classes=10 capturado como int Python → compatível com JIT)
perda_weinberger_10 = fazer_weinberger_loss(N_CLASSES)


def perda_discriminativa_2d(enc_params, X_batch, y_batch,
                             delta_pull, delta_push, lambda_reg):
    """
    Perda discriminativa para encoder 2D:
      x_i = encoder(imagem_i)  →  aplica weinberger_loss.
    Estrutura idêntica ao Ato 1 — só os pontos mudaram de origem.
    """
    x = forward_encoder(enc_params, X_batch)
    return perda_weinberger_10(x, y_batch, delta_pull, delta_push, lambda_reg)


grad_2d = jax.jit(jax.grad(perda_discriminativa_2d, argnums=0))

# Teste: gradiente com parâmetros aleatórios
CHAVE, ke2 = jax.random.split(CHAVE)
enc_teste2  = iniciar_encoder([784, 256, 64, DIM_2D], ke2)
Xb_teste    = jnp.ones((10, 784))
yb_teste    = jnp.arange(10, dtype=jnp.int32)
l_enc_teste = perda_discriminativa_2d(enc_teste2, Xb_teste, yb_teste,
                                      DP_PULL_ENC, DP_PUSH_ENC, LAMBDA_ENC)
print(f"Perda discriminativa de teste: {float(l_enc_teste):.4f}")
print("Gradiente calculado com sucesso (shapes consistentes).")
print("\n→ Esta perda é IDÊNTICA à do Ato 1 — só os pontos mudaram de origem.")

# %%
# ── ATO 2.4 — Adam (sem optax, mesmo estilo do NB00 e NB01) ─────────────────

def adam_init(params):
    """Inicializa momentos (m, v) zerados para todos os parâmetros."""
    m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    v = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params]
    return m, v


def adam_passo(params, grads, m, v, t,
               lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Um passo de Adam com correção de bias."""
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


def batch_estratificado(rng, X, y, tamanho_batch, n_cls=10):
    """Mini-batch estratificado: tamanho_batch // n_cls amostras por classe."""
    por_cls = tamanho_batch // n_cls
    idx = []
    for c in range(n_cls):
        c_idx = np.where(y == c)[0]
        idx.append(rng.choice(c_idx, por_cls, replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx)
    return jnp.array(X[idx]), jnp.array(y[idx])


print("Adam manual definido (sem optax).")
print(f"LR={LR_ENCODER}  batch={BATCH_SZ} (50 amostras/classe × 10 classes)")

# %%
# ── ATO 2.5 — Treino do encoder 2D ──────────────────────────────────────────
#
# Padrão "gera se ausente": se PRETRAINED=True e os checkpoints existirem,
# carrega. Caso contrário (execução limpa ou PRETRAINED=False), treina do zero
# e salva os checkpoints em assets/ para reutilização futura.

def carregar_pkl(fname):
    """Carrega parâmetros de encoder de arquivo pickle."""
    with open(str(ASSETS / fname), "rb") as f:
        return pickle.load(f)


def salvar_pkl(params, fname):
    """Salva parâmetros de encoder em arquivo pickle."""
    with open(str(ASSETS / fname), "wb") as f:
        pickle.dump([(np.array(W), np.array(b)) for W, b in params], f)


def obter_embeddings(enc, X):
    """Retorna embeddings como array numpy."""
    z = forward_encoder(enc, jnp.array(X, dtype=jnp.float32))
    return np.array(z)


_enc_late_path = ASSETS / "nb2_encoder_late.pkl"
if PRETRAINED and _enc_late_path.exists():
    enc_2d = carregar_pkl("nb2_encoder_late.pkl")
    print("Encoder 2D (época 250) carregado dos assets.")
else:
    # Treina do zero — também é o fallback para execução limpa no Colab
    print("Treinando encoder 2D do zero (pode levar ~20-40s no CPU)...")
    print(f"(PRETRAINED={PRETRAINED}, arquivo {'ausente' if not _enc_late_path.exists() else 'presente'})")
    rng_batch = np.random.default_rng(0)
    CHAVE, ke3 = jax.random.split(CHAVE)
    enc_2d   = iniciar_encoder([784, 256, 64, DIM_2D], ke3)

    # Salva epoch 0 (inicialização aleatória) para o filmstrip
    salvar_pkl(enc_2d, "nb2_encoder_epoch0.pkl")

    m_e, v_e = adam_init(enc_2d)
    for epoca in range(1, EPOCAS_2D + 1):
        Xb, yb = batch_estratificado(rng_batch, X_train, y_train, BATCH_SZ)
        g = grad_2d(enc_2d, Xb, yb, DP_PULL_ENC, DP_PUSH_ENC, LAMBDA_ENC)
        enc_2d, m_e, v_e = adam_passo(enc_2d, g, m_e, v_e, epoca, lr=LR_ENCODER)

        # Salva checkpoint cedo (época 20) para o filmstrip
        if epoca == 20:
            salvar_pkl(enc_2d, "nb2_encoder_early.pkl")

        if epoca % 50 == 0:
            Xj = jnp.array(X_train, dtype=jnp.float32)
            yj = jnp.array(y_train, dtype=jnp.int32)
            lv = float(perda_discriminativa_2d(enc_2d, Xj, yj,
                                               DP_PULL_ENC, DP_PUSH_ENC, LAMBDA_ENC))
            print(f"  Época {epoca:3d}  perda={lv:.4f}")

    # Salva checkpoint final
    salvar_pkl(enc_2d, "nb2_encoder_late.pkl")
    print("Treino concluído! Checkpoints salvos em assets/.")

# %% [markdown]
# ## 🟢 ATO 2.6 — Filmstrip: Evolução do Espaço de Embeddings
#
# Três momentos do treino, com os mesmos 1000 pontos de teste:
#
# - **Época 0:** caos — inicialização aleatória, sem estrutura.
# - **Época 20:** início do agrupamento — classes começam a se separar.
# - **Época 250:** classes em nuvens — o encoder aprendeu a representar dígitos.

# %%
# ── ATO 2.6 — Filmstrip de evolução do embedding ─────────────────────────────
#
# Tenta carregar os 3 checkpoints de assets/. Se a célula ATO 2.5 treinou
# do zero, os checkpoints foram salvos acima e estarão disponíveis.
# Fallback: usa o encoder atual como proxy para os estágios em falta.

try:
    enc_epoca0 = carregar_pkl("nb2_encoder_epoch0.pkl")
    enc_cedo   = carregar_pkl("nb2_encoder_early.pkl")
    enc_tarde  = carregar_pkl("nb2_encoder_late.pkl")
    print("Checkpoints das 3 épocas carregados.")
except FileNotFoundError:
    # Fallback: sem checkpoints de estágios anteriores
    enc_epoca0 = iniciar_encoder([784, 256, 64, DIM_2D], jax.random.PRNGKey(42))
    enc_cedo   = enc_2d
    enc_tarde  = enc_2d
    print("Fallback: usando encoder atual para épocas cedo/tarde.")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Evolução do Espaço de Embeddings — encoder 2D (MNIST)", fontsize=13)

for ax, (enc, titulo) in zip(axes, [
    (enc_epoca0, "Época 0 — caos\n(inicialização aleatória)"),
    (enc_cedo,   "Época 20 — início do agrupamento"),
    (enc_tarde,  "Época 250 — classes separadas"),
]):
    Z = obter_embeddings(enc, X_test)
    for c in range(N_CLASSES):
        mask = y_test == c
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   s=8, color=CORES_10[c], alpha=0.7,
                   edgecolors="none", label=str(c))
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.legend(fontsize=7, title="Dígito", markerscale=2)
    ax.grid(True, alpha=0.15)

plt.tight_layout(); plt.show()

Z_2d_teste = obter_embeddings(enc_tarde, X_test)
print(f"Embedding 2D do teste: {Z_2d_teste.shape}")

# %% [markdown]
# ## 🟡 Poll — Ato 2: Que Classes se Misturam?
#
# **Olhe o embedding da época 250.**
#
# > Quais pares de dígitos ficaram mais próximos no espaço latente?
# > Faz sentido visualmente? (dica: pense em **4/9** e **3/8**)
# >
# > **Levante a mão** com sua palpite antes de continuar.

# %% [markdown]
# ## 🟢 Resposta ao Poll: "Degenerescências Físicas" da Escrita
#
# Os dígitos **visualmente parecidos** (4 e 9, 3 e 8) formam **nuvens
# adjacentes** no espaço latente — o encoder descobriu que eles compartilham
# estrutura visual.
#
# São as *"degenerescências físicas"* da escrita: dois estados distintos (dois
# dígitos) com propriedades tão similares que qualquer representação eficiente
# os coloca próximos.
#
# > **Gancho para o Dia 4:** amanhã veremos que halos de matéria escura com
# > história de formação similar fazem o mesmo — seus embeddings se agrupam,
# > revelando sub-populações que nenhum parâmetro analítico capturava.

# %% [markdown]
# ---
# ## 🟢 ATO 3 — Colher o Espaço: Cluster + Projeções
#
# O embedding 2D já está **organizado**. Agora usamos esse espaço para:
#
# 1. **Clustering:** k-means recupera as 10 classes *sem usar rótulos na
#    inferência* — isso É segmentação de instâncias, em miniatura.
# 2. **Encoder 16D + t-SNE:** um encoder de maior dimensão, projetado com
#    t-SNE, revela estrutura mais sutil (variações dentro de cada classe).
# 3. **UMAP:** alternativa ao t-SNE com melhor preservação de estrutura global.

# %%
# ── ATO 3.1 — K-means no embedding 2D ────────────────────────────────────────
#
# Aplica k-means (K=10) ao embedding 2D do conjunto de teste.
# Calcula ARI (Adjusted Rand Index) entre clusters previstos e rótulos reais.
# ARI = 1: clusters perfeitos. ARI = 0: aleatório. ARI < 0: pior que aleatório.

km_2d   = KMeans(n_clusters=10, random_state=42, n_init=10)
y_pred  = km_2d.fit_predict(Z_2d_teste)
ari_2d  = adjusted_rand_score(y_test, y_pred)
print(f"ARI (k-means no embedding 2D): {ari_2d:.3f}")
print(f"→ {'BOM!' if ari_2d > 0.7 else 'Razoável'} — acima de 0.7 indica clustering útil.")
print()
print("ISSO É SEGMENTAÇÃO DE INSTÂNCIAS:")
print("  1. Treinar encoder com rótulos (fase de aprendizado)")
print("  2. Na inferência: embed → cluster  (sem usar rótulos!)")

# Figura: labels verdadeiros vs. k-means
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Clustering no embedding 2D — ARI = {ari_2d:.3f}", fontsize=12)

for c in range(N_CLASSES):
    mask = y_test == c
    ax1.scatter(Z_2d_teste[mask, 0], Z_2d_teste[mask, 1],
                s=8, color=CORES_10[c], alpha=0.7,
                edgecolors="none", label=str(c))
ax1.set_title("Colorido por rótulo real", fontsize=11)
ax1.set_xlabel("z₁"); ax1.set_ylabel("z₂")
ax1.legend(fontsize=7, title="Dígito real", markerscale=2)
ax1.grid(True, alpha=0.15)

for c in range(10):
    mask = y_pred == c
    ax2.scatter(Z_2d_teste[mask, 0], Z_2d_teste[mask, 1],
                s=8, color=CORES_10[c], alpha=0.7,
                edgecolors="none", label=str(c))
ax2.set_title("Colorido por cluster k-means\n(sem rótulos na inferência)", fontsize=11)
ax2.set_xlabel("z₁"); ax2.set_ylabel("z₂")
ax2.legend(fontsize=7, title="Cluster", markerscale=2)
ax2.grid(True, alpha=0.15)

plt.tight_layout(); plt.show()
print("\n→ Os clusters k-means recuperam os dígitos sem nunca ver os rótulos!")

# %%
# ── ATO 3.2 — Encoder 16D ─────────────────────────────────────────────────────
#
# Para t-SNE e UMAP, projetar 16D → 2D é mais informativo que 2D → 2D.
# O encoder 16D preserva mais estrutura interna das classes.
#
# Padrão "gera se ausente": carrega de assets/ se PRETRAINED=True e arquivo
# existir; caso contrário, treina do zero e salva.

_enc_16d_path = ASSETS / "nb2_encoder_16d_late.pkl"
if PRETRAINED and _enc_16d_path.exists():
    enc_16d = carregar_pkl("nb2_encoder_16d_late.pkl")
    print("Encoder 16D carregado dos assets.")
else:
    print("Treinando encoder 16D do zero (pode levar ~20-40s no CPU)...")

    def perda_discriminativa_16d(enc_params, X_batch, y_batch,
                                  delta_pull, delta_push, lambda_reg):
        x = forward_encoder(enc_params, X_batch)
        return perda_weinberger_10(x, y_batch, delta_pull, delta_push, lambda_reg)

    grad_16d  = jax.jit(jax.grad(perda_discriminativa_16d, argnums=0))
    rng_16    = np.random.default_rng(1)
    CHAVE, ke16 = jax.random.split(CHAVE)
    enc_16d   = iniciar_encoder([784, 256, 64, DIM_16D], ke16)
    m16, v16  = adam_init(enc_16d)

    for epoca in range(1, EPOCAS_2D + 1):
        Xb, yb = batch_estratificado(rng_16, X_train, y_train, BATCH_SZ)
        g = grad_16d(enc_16d, Xb, yb, DP_PULL_ENC, DP_PUSH_ENC, LAMBDA_ENC)
        enc_16d, m16, v16 = adam_passo(enc_16d, g, m16, v16, epoca, lr=LR_ENCODER)
        if epoca % 50 == 0:
            Xj = jnp.array(X_train, dtype=jnp.float32)
            yj = jnp.array(y_train, dtype=jnp.int32)
            lv = float(perda_discriminativa_16d(enc_16d, Xj, yj,
                                                DP_PULL_ENC, DP_PUSH_ENC, LAMBDA_ENC))
            print(f"  Época {epoca:3d}  perda={lv:.4f}")

    salvar_pkl(enc_16d, "nb2_encoder_16d_late.pkl")
    print("Treino 16D concluído! Checkpoint salvo em assets/.")

Z_16d_teste = obter_embeddings(enc_16d, X_test)
ari_16d     = adjusted_rand_score(y_test,
                  KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(Z_16d_teste))
print(f"ARI (k-means direto no embedding 16D): {ari_16d:.3f}")
print(f"Embedding 16D: {Z_16d_teste.shape}")

# %%
# ── ATO 3.3 — t-SNE com múltiplas perplexidades ───────────────────────────────
#
# t-SNE projeta o embedding 16D em 2D para visualização.
# AVISO: distâncias entre grupos NÃO têm significado em t-SNE!
# O mapa MUDA com a perplexidade — escolha diferente ≠ "mais correto".

# Subsample para no máximo 1000 pontos (requisito de velocidade)
rng_tsne = np.random.default_rng(7)
idx_ts   = rng_tsne.choice(len(X_test), min(1000, len(X_test)), replace=False)
Z_ts     = Z_16d_teste[idx_ts]
y_ts     = y_test[idx_ts]

PERPLEXIDADES = [5, 30, 100]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("t-SNE do Espaço 16D — a geometria muda com a perplexidade!",
             fontsize=13)

for ax, perp in zip(axes, PERPLEXIDADES):
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

plt.tight_layout(); plt.show()
print("\n⚠️  AVISO DAS CONSTELAÇÕES:")
print("   t-SNE preserva vizinhanças LOCAIS — distâncias entre grupos são artefatos.")
print("   Perplexidade baixa (5) = estrutura local; alta (100) = estrutura global.")
print("   Use sempre 3+ perplexidades antes de interpretar.")

# %%
# ── ATO 3.4 — UMAP (requer umap-learn) ────────────────────────────────────────
#
# UMAP preserva melhor a estrutura global que t-SNE, mas requer umap-learn.
# No Colab: !pip install umap-learn==0.5.7
# Localmente: variante não instalada neste ambiente.

if UMAP_DISPONIVEL:
    N_NEIGHBORS_LIST = [5, 15, 50]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("UMAP do Espaço 16D — n_neighbors afeta escala de estrutura",
                 fontsize=13)
    for ax, nn in zip(axes, N_NEIGHBORS_LIST):
        redutor = umap.UMAP(n_components=2, n_neighbors=nn,
                             random_state=42, min_dist=0.1)
        Z_umap = redutor.fit_transform(Z_ts)
        for c in range(N_CLASSES):
            mask = y_ts == c
            ax.scatter(Z_umap[mask, 0], Z_umap[mask, 1],
                       s=8, color=CORES_10[c], alpha=0.7,
                       edgecolors="none", label=str(c))
        ax.set_title(f"n_neighbors = {nn}", fontsize=11)
        ax.set_xlabel("UMAP₁"); ax.set_ylabel("UMAP₂")
        ax.legend(fontsize=7, title="Dígito", markerscale=2)
        ax.grid(True, alpha=0.15)
    plt.tight_layout(); plt.show()
    print("UMAP: maior n_neighbors = mais estrutura global preservada.")
else:
    print("umap-learn não instalado.")
    print("No Colab, execute: !pip install umap-learn==0.5.7")
    _umap_png = ASSETS / "nb2_fig_umap.png"
    if _umap_png.exists():
        display(Image(str(_umap_png)))
    else:
        print("(figura de referência não disponível neste ambiente)")

# %% [markdown]
# ## 🟢 Resumo: Embed-Then-Cluster
#
# ### O que aprendemos neste notebook
#
# | Ato | O que fizemos | A ideia central |
# |-----|---------------|-----------------|
# | 1 — Sandbox | Posições 2D otimizadas com pull/push | Perda contrastiva = potencial físico |
# | 2 — MNIST   | Encoder MLP com a MESMA perda | Rede aprende coordenadas úteis |
# | 3 — Colheita | K-means + t-SNE + UMAP | Espaço organizado → clustering sem rótulos |
#
# ### O pipeline *embed-then-cluster*
#
# ```
# imagem → encoder → embedding → cluster → instância
# ```
#
# > Um bom espaço de embeddings transforma um problema *sem função de perda*
# > (separar instâncias) num problema de clustering — que tem soluções clássicas.
#
# ### Ponte para o Dia 4
#
# > Amanhã veremos **esta máquina exata** aplicada a halos de matéria escura:
# > o encoder é treinado com perda discriminativa nos campos de densidade
# > de N-body simulations, e o clustering do espaço latente revela
# > sub-populações de halos com história de formação compartilhada.
# > A física é nova — a matemática é esta que você acabou de escrever.

# %% [markdown]
# ## 🟣 (Opcional) mini-SimCLR: Pares por Aumentação (Sem Rótulos)
#
# Até aqui usamos **rótulos** para definir pares positivos.
# O SimCLR dispensa rótulos: dois *augments* da mesma imagem = par positivo.
# Cada augmentação é uma declaração de invariância.

# %%
# ── (Opcional) mini-SimCLR ─────────────────────────────────────────────────────

def aumentar_mnist(x, chave, sigma_ruido=0.1, max_translacao=2):
    """
    Aumentação simples para MNIST:
    - Ruído gaussiano (sigma_ruido)
    - Translação aleatória (max_translacao pixels) com roll
    """
    chave, k1, k2 = jax.random.split(chave, 3)
    # Ruído gaussiano
    x_aug = x + jax.random.normal(k1, x.shape) * sigma_ruido
    x_aug = jnp.clip(x_aug, 0.0, 1.0)
    # Translação: remodela para 28x28, rola, reachata
    x_2d  = x_aug.reshape(-1, 28, 28)
    tx    = int(jax.random.randint(k2, (), -max_translacao, max_translacao + 1))
    ty_k, _ = jax.random.split(k2)
    ty    = int(jax.random.randint(ty_k, (), -max_translacao, max_translacao + 1))
    x_2d  = jnp.roll(x_2d, tx, axis=2)
    x_2d  = jnp.roll(x_2d, ty, axis=1)
    return x_2d.reshape(-1, 784)


print("Aumentação definida: ruído gaussiano + translação aleatória.")
print()
print("Pipeline mini-SimCLR (para rodar em casa):")
print("  1. Para cada imagem x: gerar x_aug1 = augmentar(x),  x_aug2 = augmentar(x)")
print("  2. Pares positivos: (encoder(x_aug1), encoder(x_aug2))")
print("  3. Pares negativos: outras imagens no batch")
print("  4. Perda InfoNCE sobre esses pares")
print()
print("A chave: NENHUM rótulo de classe é usado!")
print("Cada augmentação é uma declaração: 'estas duas visões são do mesmo objeto'.")
print()
print("Referência: Chen et al. 2020 — 'A Simple Framework for Contrastive Learning'")

# %% [markdown]
# ## 🟡 Para Casa
#
# ### Exercícios para explorar no seu próprio ritmo
#
# ---
#
# **Exercício 1 — Geometria da repulsão (Ato 1)**
#
# No Ato 1, mude `DELTA_PUSH` para diferentes valores: `[0.5, 1.0, 1.5, 2.5]`.
# Para cada valor:
# - Execute a relaxação (célula do Ato 1.4 com `PRETRAINED=False`).
# - Plote o estado final.
# - Meça a distância mínima entre centros de grupo.
#
# *Pergunta:* existe um valor de `DELTA_PUSH` além do qual os grupos param
# de separar ainda mais? O que isso implica sobre o equilíbrio do potencial?
#
# ---
#
# **Exercício 2 — DBSCAN em vez de k-means (Ato 3)**
#
# Substitua o k-means no Ato 3 por `sklearn.cluster.DBSCAN` com parâmetros
# `eps=0.5, min_samples=5`.
# - Compare o ARI do DBSCAN vs. k-means.
# - Quantos clusters o DBSCAN encontra? Ele detecta ruído (pontos isolados)?
#
# *Pergunta:* o que a diferença entre os dois resultados diz sobre a
# geometria do espaço de embeddings?
#
# ---
#
# **Exercício 3 — Variante Galaxy10 (com GPU no Colab)**
#
# Se tiver acesso a GPU no Colab:
# 1. Baixe o dataset Galaxy10-DECaLS (disponível via `astroNN` ou HDF5 direto).
# 2. Substitua MNIST pelo Galaxy10 no encoder 2D.
# 3. Execute o pipeline completo: embed → cluster → t-SNE.
#
# *Pergunta:* quais tipos morfológicos de galáxias ficam próximos no espaço
# latente? Faz sentido astrofisicamente?
#
# *Dica:* `pip install astroNN` no Colab; a interface é idêntica à do MNIST
# (X de shape (N, 784) normalizado para [0,1]).
