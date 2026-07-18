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
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/01_domain_shift_toy_v2.ipynb)
#
# # Notebook 01 — Domain Shift: Quebrar e Consertar um Classificador
# ### Treinar, avaliar, quebrar com shift, consertar com transfer learning
# **I Escola de Inverno do IFUSP — Bloco L2B2**
#
# > **Modo de uso:** demo guiada pelo instrutor; vocês recebem o notebook depois.
# > Ao final, teremos vivenciado todo o ciclo do domain shift num universo 2D
# > de brinquedo: treinar um classificador, avalia-lo com ferramentas ricas,
# > observar a degradacao quando o dominio muda, e conserta-lo com fine-tuning.

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
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

print(f"JAX versao  : {jax.__version__}")
print(f"Dispositivo : {jax.devices()[0]}")

# %% [markdown]
# ---
# ## Bloco 1 — Gerador de dados 2D (dominio fonte)
#
# Criamos um problema de classificacao 2D com **3 classes desbalanceadas**:
#
# | Classe | Fracao | Descricao |
# |--------|--------|-----------|
# | 0 (majoritaria) | 50% | Nuvem larga, alongada |
# | 1 (intermediaria) | 30% | Nuvem compacta |
# | 2 (minoritaria) | 20% | Nuvem pequena e alongada |
#
# As classes se **sobrepoe parcialmente** — gerando erros reais de
# classificacao — e tem **covariancia diferente**, o que forca fronteiras
# de decisao nao-lineares.
#
# **Por que desbalancear?** Em problemas reais (e.g. classificacao de quasares),
# as classes raras sao as mais interessantes fisicamente e as mais faceis de
# ignorar durante o treino.

# %%
# Parametros do dominio FONTE
N_CLASSES = 3
NOMES_CLASSES = ["Classe 0", "Classe 1", "Classe 2"]
CORES_CLASSES = ["#e74c3c", "#3498db", "#2ecc71"]

# Centros e covariancias de cada classe no dominio fonte
CENTROS_FONTE = np.array([
    [-2.0, 0.0],   # classe 0
    [2.0, 0.0],    # classe 1
    [0.0, 2.8],    # classe 2
], dtype=np.float32)

COVARIANCIAS_FONTE = np.array([
    [[1.2, 0.5], [0.5, 1.0]],    # classe 0: alongada, inclinada
    [[0.8, -0.3], [-0.3, 0.8]],  # classe 1: ligeiramente inclinada
    [[0.5, 0.0], [0.0, 1.3]],    # classe 2: alongada verticalmente
], dtype=np.float32)

# Proporcoes de cada classe
FRACOES = np.array([0.50, 0.30, 0.20])

# %%
def gerar_dados(n_total, centros, covariancias, fracoes, chave):
    """Gera mistura de Gaussianas 2D com classes desbalanceadas.

    Retorna X (n_total, 2) e y (n_total,) como arrays JAX.
    """
    X_list, y_list = [], []
    for c in range(len(fracoes)):
        n_c = int(n_total * fracoes[c])
        chave, k = jax.random.split(chave)
        # Amostra de normal padrao e transforma com Cholesky
        L = jnp.linalg.cholesky(jnp.array(covariancias[c]))
        z = jax.random.normal(k, (n_c, 2))
        pts = z @ L.T + centros[c]
        X_list.append(pts)
        y_list.append(jnp.full(n_c, c, dtype=jnp.int32))
    return jnp.concatenate(X_list), jnp.concatenate(y_list)


# Gerar dados fonte: treino + validacao
N_TREINO = 800
N_VAL = 200

key_src_tr, key_src_val, KEY = jax.random.split(KEY, 3)
X_treino, y_treino = gerar_dados(N_TREINO, CENTROS_FONTE, COVARIANCIAS_FONTE,
                                 FRACOES, key_src_tr)
X_val, y_val = gerar_dados(N_VAL, CENTROS_FONTE, COVARIANCIAS_FONTE,
                           FRACOES, key_src_val)

print(f"Treino: {X_treino.shape[0]} pontos")
print(f"Validacao: {X_val.shape[0]} pontos")
for c in range(N_CLASSES):
    n_tr = int((y_treino == c).sum())
    n_va = int((y_val == c).sum())
    print(f"  {NOMES_CLASSES[c]}: {n_tr} treino, {n_va} val")

# %%
fig, ax = plt.subplots(figsize=(6.5, 5))
for c in range(N_CLASSES):
    mask = y_treino == c
    ax.scatter(X_treino[mask, 0], X_treino[mask, 1],
               s=15, color=CORES_CLASSES[c], alpha=0.6,
               edgecolors="none", label=f"{NOMES_CLASSES[c]} ({int(FRACOES[c]*100)}%)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Dominio fonte — dados de treino")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Bloco 2 — Treinar um classificador (Equinox + Optax)
#
# Usamos a mesma receita do Notebook 00:
#
# - **Equinox** define o modelo como modulo.
# - **Optax** fornece o otimizador (Adam).
#
# A arquitetura: MLP `[2 -> 32 -> 32 -> 3]` com ativacao `tanh` nas
# camadas ocultas e saida linear (logits para 3 classes).

# %%
class Classificador(eqx.Module):
    """MLP classificador com camadas de tamanho variavel."""
    layers: list

    def __init__(self, camadas, key):
        self.layers = []
        for i in range(len(camadas) - 1):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(camadas[i], camadas[i + 1], key=subkey))

    def __call__(self, x):
        """Forward pass: tanh nas ocultas, linear na saida."""
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


# Criar modelo
CAMADAS = [2, 32, 32, N_CLASSES]
modelo = Classificador(CAMADAS, key=jax.random.PRNGKey(0))

# Teste rapido
y_teste = jax.vmap(modelo)(X_treino[:3])
print(f"Entrada: shape {X_treino[:3].shape}")
print(f"Saida (logits): shape {y_teste.shape}")
n_params = sum(jnp.size(p) for p in jax.tree_util.tree_leaves(modelo))
print(f"Total de parametros: {n_params}")

# %% [markdown]
# ### Entropia cruzada ponderada
#
# Sem pesos, o modelo pode ignorar a classe minoritaria (20%) e ainda ter
# acuracia alta. Pesos inversamente proporcionais a frequencia forcam o
# modelo a levar as classes raras a serio.

# %%
# Pesos de classe: inverso da frequencia, normalizados
pesos_classe = jnp.array(1.0 / FRACOES)
pesos_classe = pesos_classe / pesos_classe.sum() * N_CLASSES
print(f"Pesos de classe: {pesos_classe}")


def perda_ce_ponderada(modelo, x_batch, y_batch):
    """Entropia cruzada ponderada por classe."""
    logits = jax.vmap(modelo)(x_batch)
    log_p = jax.nn.log_softmax(logits, axis=-1)
    N = y_batch.shape[0]
    log_py = log_p[jnp.arange(N), y_batch]
    pesos = pesos_classe[y_batch]
    return -jnp.mean(pesos * log_py)

# %%
# Otimizador e passo de treino
otimizador = optax.adam(learning_rate=3e-3)
opt_state = otimizador.init(eqx.filter(modelo, eqx.is_array))


@eqx.filter_jit
def passo_treino(modelo, estado, x, y):
    """Um passo: gradiente + atualizacao Adam."""
    perda, grads = eqx.filter_value_and_grad(perda_ce_ponderada)(modelo, x, y)
    atualizacoes, estado = otimizador.update(grads, estado, modelo)
    modelo = eqx.apply_updates(modelo, atualizacoes)
    return modelo, estado, perda

# %%
# Loop de treino com mini-batches
N_EPOCAS = 1500
BATCH_SIZE = 64

historico_treino = []
historico_val = []
chave_treino = jax.random.PRNGKey(1)

print(f"Treinando {N_EPOCAS} epocas — Adam (lr=3e-3, batch={BATCH_SIZE})")

for epoca in range(1, N_EPOCAS + 1):
    chave_treino, chave_perm = jax.random.split(chave_treino)
    perm = jax.random.permutation(chave_perm, X_treino.shape[0])
    X_emb = X_treino[perm]
    y_emb = y_treino[perm]

    for i in range(0, X_treino.shape[0], BATCH_SIZE):
        x_b = X_emb[i:i + BATCH_SIZE]
        y_b = y_emb[i:i + BATCH_SIZE]
        modelo, opt_state, _ = passo_treino(modelo, opt_state, x_b, y_b)

    if epoca % 50 == 0 or epoca == 1:
        p_tr = float(perda_ce_ponderada(modelo, X_treino, y_treino))
        p_va = float(perda_ce_ponderada(modelo, X_val, y_val))
        historico_treino.append((epoca, p_tr))
        historico_val.append((epoca, p_va))
        if epoca <= 1 or epoca % 300 == 0:
            print(f"  Epoca {epoca:4d}  treino={p_tr:.4f}  val={p_va:.4f}")

print("Treino concluido.")

# %%
# Curvas de aprendizado
ep_tr, l_tr = zip(*historico_treino)
ep_va, l_va = zip(*historico_val)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(ep_tr, l_tr, "-", lw=1.5, color="#2980b9", label="treino")
ax.plot(ep_va, l_va, "-", lw=1.5, color="#e74c3c", label="validacao")
ax.set_xlabel("Epoca")
ax.set_ylabel("Perda (CE ponderada)")
ax.set_title("Curvas de aprendizado — modelo fonte")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Bloco 3 — Avaliacao: visualizacoes e metricas
#
# O modelo treinou. Agora precisamos **entender** o que ele aprendeu.
# Vamos construir 4 ferramentas de avaliacao, cada uma revelando um
# aspecto diferente do classificador.

# %% [markdown]
# ### 3a) Mapa de probabilidade
#
# Avaliamos o modelo numa malha densa cobrindo o espaco 2D. Para cada
# ponto da malha, calculamos $P(\text{classe } k \mid x)$. Isso mostra
# exatamente o que o modelo **acredita** em cada regiao — inclusive onde
# ele esta confiante mas errado.

# %%
def criar_malha(xlim=(-6, 6), ylim=(-4, 7), passo=0.10):
    """Cria malha 2D para avaliacao densa."""
    xx, yy = np.meshgrid(
        np.arange(xlim[0], xlim[1], passo),
        np.arange(ylim[0], ylim[1], passo),
    )
    return xx, yy


def probabilidades_malha(modelo, xx, yy):
    """Calcula softmax do modelo em todos os pontos da malha."""
    pts = jnp.array(np.c_[xx.ravel(), yy.ravel()], dtype=jnp.float32)
    logits = jax.vmap(modelo)(pts)
    probs = jax.nn.softmax(logits, axis=-1)
    return np.array(probs)


xx, yy = criar_malha()
probs_malha = probabilidades_malha(modelo, xx, yy)

# %%
# Mapa de probabilidade — uma subpanel por classe
fig, axes = plt.subplots(1, N_CLASSES, figsize=(14, 4))
fig.suptitle("Mapa de probabilidade — $P$(classe $k$ | $x$)", fontsize=13)

for c in range(N_CLASSES):
    P_c = probs_malha[:, c].reshape(xx.shape)
    im = axes[c].pcolormesh(xx, yy, P_c, cmap="RdBu_r", shading="auto",
                            vmin=0, vmax=1)
    # Sobrepor pontos de validacao
    mask = y_val == c
    axes[c].scatter(X_val[mask, 0], X_val[mask, 1],
                    s=12, color=CORES_CLASSES[c], edgecolors="k",
                    linewidths=0.3, zorder=3)
    axes[c].set_title(NOMES_CLASSES[c])
    axes[c].set_xlabel("$x_1$")
    if c == 0:
        axes[c].set_ylabel("$x_2$")
    plt.colorbar(im, ax=axes[c], fraction=0.046)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3b) Mapa de decisao (regioes de classe)
#
# Aplicamos $\arg\max$ nas probabilidades: cada ponto da malha recebe
# a classe de maior probabilidade. Isso mostra as **fronteiras de decisao**
# aprendidas pelo modelo.

# %%
def plotar_mapa_decisao(ax, modelo, X_overlay, y_overlay, titulo,
                        xx, yy, probs=None, alpha_fundo=0.35):
    """Plota regioes de decisao com pontos sobrepostos."""
    if probs is None:
        probs = probabilidades_malha(modelo, xx, yy)
    preds = probs.argmax(axis=1).reshape(xx.shape)

    cmap_bg = mcolors.ListedColormap(
        [mcolors.to_rgba(c, alpha_fundo) for c in CORES_CLASSES])
    ax.pcolormesh(xx, yy, preds, cmap=cmap_bg, shading="auto",
                  vmin=0, vmax=N_CLASSES - 1)

    for c in range(N_CLASSES):
        mask = y_overlay == c
        ax.scatter(np.array(X_overlay[mask, 0]),
                   np.array(X_overlay[mask, 1]),
                   s=15, color=CORES_CLASSES[c], edgecolors="k",
                   linewidths=0.3, zorder=3, label=NOMES_CLASSES[c])

    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


fig, ax = plt.subplots(figsize=(6.5, 5))
plotar_mapa_decisao(ax, modelo, X_val, y_val,
                    "Mapa de decisao — validacao fonte", xx, yy,
                    probs=probs_malha)
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3c) Matriz de confusao e metricas por classe
#
# A **acuracia** global pode ser enganosa com classes desbalanceadas:
# se o modelo prever sempre "Classe 0", ja tera 50% de acerto!
#
# Metricas por classe revelam a verdade:
#
# | Metrica | O que mede |
# |---------|-----------|
# | **TPR** (recall) | Dos exemplos reais da classe, quantos o modelo acertou? |
# | **PPV** (precision) | Das predicoes nessa classe, quantas estavam corretas? |
# | **F1** | Media harmonica de TPR e PPV — resume ambos num numero |

# %%
def metricas_classificacao(y_true, y_pred, n_classes=N_CLASSES):
    """Calcula matriz de confusao, TPR, PPV e F1 por classe.

    Retorna: cm (n,n), tpr (n,), ppv (n,), f1 (n,), acc (float)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1

    tpr = np.zeros(n_classes)
    ppv = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tpr[c] = tp / max(tp + fn, 1)
        ppv[c] = tp / max(tp + fp, 1)
        if tpr[c] + ppv[c] > 0:
            f1[c] = 2 * tpr[c] * ppv[c] / (tpr[c] + ppv[c])

    acc = np.trace(cm) / max(cm.sum(), 1)
    return cm, tpr, ppv, f1, acc


def plotar_matriz_confusao(ax, cm, titulo):
    """Plota a matriz de confusao como heatmap."""
    ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([f"C{c}" for c in range(N_CLASSES)], fontsize=9)
    ax.set_yticklabels([f"C{c}" for c in range(N_CLASSES)], fontsize=9)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            cor = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=cor, fontsize=10)

# %%
# Predicoes no conjunto de validacao fonte
preds_val_fonte = np.array(jnp.argmax(jax.vmap(modelo)(X_val), axis=-1))
cm_fonte, tpr_fonte, ppv_fonte, f1_fonte, acc_fonte = metricas_classificacao(
    np.array(y_val), preds_val_fonte)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plotar_matriz_confusao(ax1, cm_fonte, "Matriz de confusao — val. fonte")

# Tabela de metricas como barplot
x_pos = np.arange(N_CLASSES)
largura = 0.25
ax2.bar(x_pos - largura, tpr_fonte, largura, label="TPR (recall)",
        color="#3498db")
ax2.bar(x_pos, ppv_fonte, largura, label="PPV (precision)",
        color="#e74c3c")
ax2.bar(x_pos + largura, f1_fonte, largura, label="F1",
        color="#2ecc71")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(NOMES_CLASSES, fontsize=9)
ax2.set_ylim(0, 1.1)
ax2.set_ylabel("Score")
ax2.set_title("Metricas por classe — val. fonte")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

print(f"Acuracia global: {acc_fonte:.3f}")
print(f"Macro-F1       : {f1_fonte.mean():.3f}")
for c in range(N_CLASSES):
    print(f"  {NOMES_CLASSES[c]}: TPR={tpr_fonte[c]:.3f}  "
          f"PPV={ppv_fonte[c]:.3f}  F1={f1_fonte[c]:.3f}")

# %% [markdown]
# ### 3d) Curva ROC e AUC (por classe, one-vs-rest)
#
# A curva ROC mostra o trade-off entre **taxa de verdadeiro positivo**
# (TPR) e **taxa de falso positivo** (FPR) ao variar o limiar de decisao.
#
# - **AUC = 1.0**: separacao perfeita.
# - **AUC = 0.5**: classificador aleatorio (a diagonal).
#
# A AUC captura o que a acuracia nao captura: quao bem o modelo *ordena*
# as probabilidades, independentemente do limiar escolhido.

# %%
def calcular_roc_auc(y_true, probs, n_classes=N_CLASSES):
    """Calcula curva ROC e AUC por classe (one-vs-rest).

    Trata o caso degenerado (classe ausente no conjunto) retornando None.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    resultados = []

    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        # Caso degenerado: todos da mesma classe
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            resultados.append({"fpr": None, "tpr": None, "auc": None})
            continue

        scores = probs[:, c]
        # Ordenar por score decrescente
        ordem = np.argsort(-scores)
        y_sorted = y_bin[ordem]
        scores_sorted = scores[ordem]

        # Calcular TPR e FPR para cada limiar unico
        n_pos = y_bin.sum()
        n_neg = len(y_bin) - n_pos

        tpr_list = [0.0]
        fpr_list = [0.0]
        tp_acum = 0
        fp_acum = 0

        # Limiares unicos (pontos onde o score muda)
        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp_acum += 1
            else:
                fp_acum += 1
            # Registrar ponto se proximo score e diferente ou e o ultimo
            if i == len(y_sorted) - 1 or scores_sorted[i] != scores_sorted[i + 1]:
                tpr_list.append(tp_acum / n_pos)
                fpr_list.append(fp_acum / n_neg)

        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)

        # AUC via regra do trapezio
        auc_val = float(np.trapezoid(tpr_arr, fpr_arr))
        resultados.append({"fpr": fpr_arr, "tpr": tpr_arr, "auc": auc_val})

    return resultados

# %%
# ROC/AUC para o modelo fonte na validacao
probs_val_fonte = np.array(jax.nn.softmax(jax.vmap(modelo)(X_val), axis=-1))
roc_fonte = calcular_roc_auc(np.array(y_val), probs_val_fonte)

fig, ax = plt.subplots(figsize=(5.5, 5))
for c in range(N_CLASSES):
    r = roc_fonte[c]
    if r["auc"] is not None:
        ax.plot(r["fpr"], r["tpr"], lw=2, color=CORES_CLASSES[c],
                label=f'{NOMES_CLASSES[c]} (AUC={r["auc"]:.3f})')

ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Aleatorio (0.5)")
ax.set_xlabel("Taxa de Falso Positivo (FPR)")
ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR)")
ax.set_title("Curvas ROC — validacao fonte (one-vs-rest)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

for c in range(N_CLASSES):
    auc_val = roc_fonte[c]["auc"]
    auc_str = f"{auc_val:.3f}" if auc_val is not None else "N/A"
    print(f"  {NOMES_CLASSES[c]}: AUC = {auc_str}")

# %% [markdown]
# ---
# ## Bloco 4 — Introduzindo o domain shift
#
# Agora geramos dados de um **dominio alvo** (target) onde as distribuicoes
# mudaram. Duas classes foram deslocadas:
#
# - **Classe 1**: centro deslocou de (2, 0) para (2, 2) — shift vertical.
# - **Classe 2**: centro deslocou de (0, 2.8) para (-1.5, 2.8) — shift horizontal.
#
# A Classe 0 (majoritaria) **nao mudou** — o modelo ainda acerta nela,
# mascarando o problema na acuracia global.

# %%
# Parametros do dominio ALVO — classes 1 e 2 deslocadas
CENTROS_ALVO = np.array([
    [-2.0, 0.0],    # classe 0 — mesma posicao
    [2.0, 2.0],     # classe 1 — deslocada +2 em y
    [-1.5, 2.8],    # classe 2 — deslocada -1.5 em x
], dtype=np.float32)

# Covariancias sao as mesmas (covariate shift, nao concept shift)
COVARIANCIAS_ALVO = COVARIANCIAS_FONTE.copy()

# Gerar dados do alvo
N_ALVO = 600
key_tgt, KEY = jax.random.split(KEY)
X_alvo, y_alvo = gerar_dados(N_ALVO, CENTROS_ALVO, COVARIANCIAS_ALVO,
                             FRACOES, key_tgt)

print(f"Dados do alvo: {X_alvo.shape[0]} pontos")
for c in range(N_CLASSES):
    n_c = int((y_alvo == c).sum())
    print(f"  {NOMES_CLASSES[c]}: {n_c} pontos")

# %%
# Figura: fonte vs. alvo lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Domain shift: fonte vs. alvo", fontsize=13)

for c in range(N_CLASSES):
    mask = y_treino == c
    ax1.scatter(X_treino[mask, 0], X_treino[mask, 1],
                s=15, color=CORES_CLASSES[c], alpha=0.6,
                edgecolors="none", label=NOMES_CLASSES[c])
ax1.set_title("Fonte (Source)")
ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-6, 6); ax1.set_ylim(-4, 7)

for c in range(N_CLASSES):
    mask = y_alvo == c
    ax2.scatter(X_alvo[mask, 0], X_alvo[mask, 1],
                s=15, color=CORES_CLASSES[c], alpha=0.6,
                edgecolors="none", label=NOMES_CLASSES[c])
    # Indicar deslocamento com seta
    if c > 0:
        dx = CENTROS_ALVO[c, 0] - CENTROS_FONTE[c, 0]
        dy = CENTROS_ALVO[c, 1] - CENTROS_FONTE[c, 1]
        ax2.annotate("", xy=CENTROS_ALVO[c], xytext=CENTROS_FONTE[c],
                     arrowprops=dict(arrowstyle="->", lw=2,
                                     color=CORES_CLASSES[c]))

ax2.set_title("Alvo (Target) — classes 1 e 2 deslocadas")
ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-6, 6); ax2.set_ylim(-4, 7)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Bloco 5 — Quanto o modelo se degrada no alvo?
#
# Aplicamos o modelo fonte (sem nenhuma modificacao) aos dados do alvo.
# Vamos reutilizar as 4 ferramentas do Bloco 3 para medir a degradacao.

# %%
# Predicoes do modelo fonte no alvo
preds_alvo = np.array(jnp.argmax(jax.vmap(modelo)(X_alvo), axis=-1))
cm_alvo, tpr_alvo, ppv_alvo, f1_alvo, acc_alvo = metricas_classificacao(
    np.array(y_alvo), preds_alvo)

print("=== Modelo fonte aplicado ao ALVO ===")
print(f"Acuracia: {acc_alvo:.3f}  (fonte: {acc_fonte:.3f})")
print(f"Macro-F1: {f1_alvo.mean():.3f}  (fonte: {f1_fonte.mean():.3f})")
print()
for c in range(N_CLASSES):
    print(f"  {NOMES_CLASSES[c]}: TPR={tpr_alvo[c]:.3f} (fonte: {tpr_fonte[c]:.3f})  "
          f"F1={f1_alvo[c]:.3f} (fonte: {f1_fonte[c]:.3f})")

# %%
# Mapas de decisao: fonte vs. alvo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Modelo fonte — aplicado a fonte vs. aplicado ao alvo", fontsize=13)

plotar_mapa_decisao(ax1, modelo, X_val, y_val,
                    f"Fonte (acc={acc_fonte:.2f}, F1={f1_fonte.mean():.2f})",
                    xx, yy, probs=probs_malha)
ax1.legend(fontsize=8)

plotar_mapa_decisao(ax2, modelo, X_alvo, y_alvo,
                    f"Alvo (acc={acc_alvo:.2f}, F1={f1_alvo.mean():.2f})",
                    xx, yy, probs=probs_malha)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% [markdown]
# As fronteiras de decisao **nao mudaram** (o modelo e o mesmo), mas os
# dados do alvo se deslocaram para regioes que o modelo atribui a outras
# classes. O modelo esta **confiante e errado** — o pior cenario possivel.

# %%
# Matrizes de confusao comparadas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plotar_matriz_confusao(ax1, cm_fonte, "Confusao — val. fonte")
plotar_matriz_confusao(ax2, cm_alvo, "Confusao — alvo (degradado)")
# Trocar colormap do segundo para vermelho
ax2.images[0].set_cmap("Reds")
plt.tight_layout()
plt.show()

# %%
# ROC/AUC no alvo
probs_alvo_full = np.array(jax.nn.softmax(jax.vmap(modelo)(X_alvo), axis=-1))
roc_alvo = calcular_roc_auc(np.array(y_alvo), probs_alvo_full)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Curvas ROC — fonte vs. alvo", fontsize=13)

for c in range(N_CLASSES):
    r = roc_fonte[c]
    if r["auc"] is not None:
        ax1.plot(r["fpr"], r["tpr"], lw=2, color=CORES_CLASSES[c],
                 label=f'{NOMES_CLASSES[c]} ({r["auc"]:.3f})')
ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax1.set_title("Fonte")
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax1.set_aspect("equal")

for c in range(N_CLASSES):
    r = roc_alvo[c]
    if r["auc"] is not None:
        ax2.plot(r["fpr"], r["tpr"], lw=2, color=CORES_CLASSES[c],
                 label=f'{NOMES_CLASSES[c]} ({r["auc"]:.3f})')
ax2.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax2.set_title("Alvo (degradado)")
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.set_aspect("equal")

plt.tight_layout()
plt.show()

print("\nResumo AUC:")
print(f"{'Classe':<12} {'Fonte':>8} {'Alvo':>8} {'Delta':>8}")
for c in range(N_CLASSES):
    a_f = roc_fonte[c]["auc"] if roc_fonte[c]["auc"] is not None else 0
    a_a = roc_alvo[c]["auc"] if roc_alvo[c]["auc"] is not None else 0
    print(f"{NOMES_CLASSES[c]:<12} {a_f:>8.3f} {a_a:>8.3f} {a_a - a_f:>+8.3f}")

# %% [markdown]
# ---
# ## Bloco 6 — Transfer learning supervisionado (fine-tuning)
#
# Temos um modelo que funciona bem na fonte mas falha no alvo.
# A solucao mais simples: **fine-tuning** — re-treinar o modelo usando
# uma pequena amostra rotulada do dominio alvo.
#
# Concretamente:
# - Partimos do modelo ja treinado na fonte (todos os pesos).
# - Re-treinamos com um learning rate menor e poucos dados do alvo.
# - Isso e o **supervised transfer learning** apresentado na teoria (L2B1).
#
# Usamos apenas **K = 60 rotulos do alvo** (10% dos dados do alvo).

# %%
# Amostrar K rotulos do alvo (estratificado por classe)
K_ROTULOS = 60


def amostrar_k_rotulos(X, y, K, chave):
    """Amostra K pontos rotulados, estratificado por classe."""
    k_por_classe = max(2, K // N_CLASSES)
    idx_list = []
    for c in range(N_CLASSES):
        idx_c = jnp.where(y == c)[0]
        chave, k = jax.random.split(chave)
        escolhidos = jax.random.permutation(k, idx_c.shape[0])[:k_por_classe]
        idx_list.append(idx_c[escolhidos])
    idx = jnp.concatenate(idx_list)
    return X[idx], y[idx]


key_ft, KEY = jax.random.split(KEY)
X_ft, y_ft = amostrar_k_rotulos(X_alvo, y_alvo, K_ROTULOS, key_ft)
print(f"Amostra para fine-tuning: {X_ft.shape[0]} pontos do alvo")
for c in range(N_CLASSES):
    print(f"  {NOMES_CLASSES[c]}: {int((y_ft == c).sum())} pontos")

# %%
# Fine-tuning: copiar modelo fonte e re-treinar com lr menor
import copy

# Copiar o modelo fonte (deep copy dos parametros)
modelo_ft = jax.tree.map(lambda x: x, modelo)

# Learning rate menor para fine-tuning (nao destruir o que ja aprendeu)
otimizador_ft = optax.adam(learning_rate=1e-3)
opt_state_ft = otimizador_ft.init(eqx.filter(modelo_ft, eqx.is_array))

N_EPOCAS_FT = 800
historico_ft = []
chave_ft = jax.random.PRNGKey(7)


@eqx.filter_jit
def passo_ft(modelo, estado, x, y):
    """Passo de fine-tuning."""
    perda, grads = eqx.filter_value_and_grad(perda_ce_ponderada)(modelo, x, y)
    atualizacoes, estado = otimizador_ft.update(grads, estado, modelo)
    modelo = eqx.apply_updates(modelo, atualizacoes)
    return modelo, estado, perda


print(f"Fine-tuning: {N_EPOCAS_FT} epocas, lr=1e-3, K={X_ft.shape[0]} rotulos")

for epoca in range(1, N_EPOCAS_FT + 1):
    chave_ft, chave_perm = jax.random.split(chave_ft)
    perm = jax.random.permutation(chave_perm, X_ft.shape[0])
    modelo_ft, opt_state_ft, perda = passo_ft(modelo_ft, opt_state_ft,
                                               X_ft[perm], y_ft[perm])

    if epoca % 25 == 0 or epoca == 1:
        p = float(perda_ce_ponderada(modelo_ft, X_ft, y_ft))
        historico_ft.append((epoca, p))
        if epoca <= 1 or epoca % 200 == 0:
            print(f"  Epoca {epoca:4d}  perda={p:.4f}")

print("Fine-tuning concluido.")

# %%
# Curva de perda do fine-tuning
ep_ft, l_ft = zip(*historico_ft)

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(ep_ft, l_ft, "-", lw=1.5, color="#8e44ad")
ax.set_xlabel("Epoca")
ax.set_ylabel("Perda (CE ponderada)")
ax.set_title(f"Fine-tuning — perda nos {X_ft.shape[0]} rotulos do alvo")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Bloco 7 — Comparacao final
#
# Tres cenarios, lado a lado:
#
# | Cenario | Modelo | Dados avaliados |
# |---------|--------|-----------------|
# | (A) Baseline | fonte | validacao fonte |
# | (B) Degradacao | fonte | alvo |
# | (C) Reparacao | fine-tuned | alvo |
#
# Se o fine-tuning funcionou, (C) deve se aproximar de (A).

# %%
# Metricas do modelo fine-tuned no alvo
preds_ft = np.array(jnp.argmax(jax.vmap(modelo_ft)(X_alvo), axis=-1))
cm_ft, tpr_ft, ppv_ft, f1_ft, acc_ft = metricas_classificacao(
    np.array(y_alvo), preds_ft)

probs_ft_full = np.array(jax.nn.softmax(jax.vmap(modelo_ft)(X_alvo), axis=-1))
roc_ft = calcular_roc_auc(np.array(y_alvo), probs_ft_full)

# %%
# Figura-sintese: 3 mapas de decisao
probs_malha_ft = probabilidades_malha(modelo_ft, xx, yy)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Comparacao final — tres cenarios", fontsize=14)

plotar_mapa_decisao(axes[0], modelo, X_val, y_val,
                    f"(A) Fonte->Fonte\nacc={acc_fonte:.2f}  F1={f1_fonte.mean():.2f}",
                    xx, yy, probs=probs_malha)
axes[0].legend(fontsize=7, loc="upper right")

plotar_mapa_decisao(axes[1], modelo, X_alvo, y_alvo,
                    f"(B) Fonte->Alvo\nacc={acc_alvo:.2f}  F1={f1_alvo.mean():.2f}",
                    xx, yy, probs=probs_malha)
axes[1].legend(fontsize=7, loc="upper right")

plotar_mapa_decisao(axes[2], modelo_ft, X_alvo, y_alvo,
                    f"(C) Fine-tuned->Alvo\nacc={acc_ft:.2f}  F1={f1_ft.mean():.2f}",
                    xx, yy, probs=probs_malha_ft)
axes[2].legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.show()

# %%
# Tabela-resumo de metricas
print("=" * 62)
print(f"{'Metrica':<14} {'(A) Fonte':>12} {'(B) Alvo':>12} {'(C) Fine-tuned':>14}")
print("=" * 62)
print(f"{'Acuracia':<14} {acc_fonte:>12.3f} {acc_alvo:>12.3f} {acc_ft:>14.3f}")
print(f"{'Macro-F1':<14} {f1_fonte.mean():>12.3f} {f1_alvo.mean():>12.3f} {f1_ft.mean():>14.3f}")
print("-" * 62)
for c in range(N_CLASSES):
    a_f = roc_fonte[c]["auc"] if roc_fonte[c]["auc"] is not None else 0
    a_a = roc_alvo[c]["auc"] if roc_alvo[c]["auc"] is not None else 0
    a_ft = roc_ft[c]["auc"] if roc_ft[c]["auc"] is not None else 0
    print(f"AUC {NOMES_CLASSES[c]:<9} {a_f:>12.3f} {a_a:>12.3f} {a_ft:>14.3f}")
print("-" * 62)
for c in range(N_CLASSES):
    print(f"F1  {NOMES_CLASSES[c]:<9} {f1_fonte[c]:>12.3f} {f1_alvo[c]:>12.3f} {f1_ft[c]:>14.3f}")
print("=" * 62)

# %%
# Barplot comparativo de F1 por classe nos 3 cenarios
fig, ax = plt.subplots(figsize=(8, 4.5))
x_pos = np.arange(N_CLASSES)
largura = 0.25

ax.bar(x_pos - largura, f1_fonte, largura, label="(A) Fonte->Fonte",
       color="#3498db")
ax.bar(x_pos, f1_alvo, largura, label="(B) Fonte->Alvo",
       color="#e74c3c")
ax.bar(x_pos + largura, f1_ft, largura, label="(C) Fine-tuned->Alvo",
       color="#2ecc71")

ax.set_xticks(x_pos)
ax.set_xticklabels(NOMES_CLASSES)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1-score")
ax.set_title("F1 por classe — comparacao dos 3 cenarios")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Takeaway
#
# > **Domain shift degrada em silencio; fine-tuning com poucos rotulos conserta.**
#
# O que aprendemos:
#
# 1. Um modelo pode ter **alta confianca e estar errado** quando o dominio muda.
#    Acuracia alta na fonte **nao garante** bom desempenho no alvo.
#
# 2. **Metricas por classe** (TPR, F1, AUC) revelam degradacoes que a
#    acuracia global esconde — especialmente em classes minoritarias.
#
# 3. **Fine-tuning supervisionado** com poucos rotulos do alvo e suficiente
#    para recuperar grande parte do desempenho. Nao precisamos re-treinar
#    do zero: o conhecimento aprendido na fonte transfere.
#
# Na proxima aula (L3), veremos como **aprendizado contrastivo** constroi
# representacoes que sao naturalmente mais robustas a mudancas de dominio.

# %% [markdown]
# ---
# ## Para casa
#
# **Exercicio 1 — Magnitude do shift**
#
# Altere `CENTROS_ALVO` para deslocar as classes por apenas 0.5 unidades
# (em vez de 2.0). Repita a avaliacao. Para qual magnitude de shift o
# modelo fonte comeca a ser aceitavel sem fine-tuning?
#
# **Exercicio 2 — Orcamento de rotulos**
#
# Varie `K_ROTULOS` entre 10 e 200 e plote o Macro-F1 do modelo fine-tuned
# em funcao de K. A partir de qual K o modelo praticamente iguala o baseline?
#
# **Exercicio 3 — Quais camadas adaptar?**
#
# No fine-tuning, todos os pesos do modelo foram atualizados. Tente
# congelar a primeira camada (usando `eqx.tree_at` para zerar seus
# gradientes) e treinar apenas as camadas 2 e 3. O resultado melhora ou
# piora? Por que?
