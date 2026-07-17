# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: WinterSchool
#     language: python
#     name: WinterSchool
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/00_caixa_de_ferramentas_v2.ipynb)
#
# # Notebook 00 — A Caixa de Ferramentas
# ### JAX, redes neurais do zero e com bibliotecas
# **I Escola de Inverno do IFUSP — Bloco L1B2**
#
# > **Modo de uso:** demo guiada pelo instrutor; vocês recebem o notebook depois.
# > Ao final, vocês terão construído e treinado uma rede neural para regressão
# > — primeiro na mão (JAX puro), depois com Equinox + Optax.

# %% [markdown]
# ## O que é este ambiente?
#
# **Google Colab** roda Python num servidor remoto, direto do navegador.
# O documento é um *notebook Jupyter*:
#
# - **Células de código** (fundo cinza): `Shift+Enter` para executar.
# - **Células de texto** (como esta): explicações em Markdown.
#
# Cada célula executada mantém o estado: variáveis ficam disponíveis nas seguintes.

# %%
# Instalação de pacotes (só no Colab — localmente já estão instalados)
import subprocess, sys
try:
    import google.colab  # noqa: F401
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "jax", "jaxlib", "equinox", "optax", "matplotlib"])
except ImportError:
    pass

# %%
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Semente de reprodutibilidade
SEED = 42
KEY = jax.random.PRNGKey(SEED)

# Estilo dos gráficos
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

print(f"JAX versão  : {jax.__version__}")
print(f"Dispositivo : {jax.devices()[0]}")

# %% [markdown]
# ---
# ## 1. Backbone matemático em JAX — arrays, shapes, broadcasting
#
# Toda rede neural, no fundo, é uma sequência de **multiplicações de matrizes**
# seguidas de funções não-lineares. Por isso, o primeiro passo é dominar arrays
# e suas operações. Fazemos tudo com `jax.numpy` (abreviado `jnp`).

# %% [markdown]
# ### Criando arrays

# %%
# Criando arrays com jax.numpy
vetor = jnp.array([1.0, 2.0, 3.0])
print("vetor       :", vetor)
print("vetor.shape :", vetor.shape)    # (3,) — um vetor com 3 elementos
print("vetor.dtype :", vetor.dtype)    # float32 — padrão do JAX
print()

# Funções utilitárias
zeros = jnp.zeros(5)
uns   = jnp.ones((2, 3))              # matriz 2x3 de uns
grade = jnp.linspace(0, 10, 5)        # 5 pontos igualmente espaçados
print("zeros       :", zeros)
print("uns         :\n", uns)
print("grade       :", grade)

# %% [markdown]
# ### Broadcasting — operações entre arrays de tamanhos diferentes
#
# Quando operamos um escalar com um vetor, o JAX "estica" automaticamente
# o escalar para ter o mesmo tamanho. Essa regra se generaliza para
# dimensões maiores — é o **broadcasting**.

# %%
# Regra 1: escalar + vetor — o escalar é "esticado"
T_celsius = jnp.array([20.1, 21.5, 19.8, 22.3])
T_kelvin  = T_celsius + 273.15   # 273.15 (shape ()) → esticado para shape (4,)

print("T_celsius.shape :", T_celsius.shape)
print("273.15 (escalar) é esticado para (4,)")
print("T_kelvin        :", T_kelvin)

# %%
# Regra 2: duas dimensões compatíveis — ambas são esticadas
coluna = jnp.array([[1.0],
                     [2.0],
                     [3.0]])          # shape (3, 1)
linha  = jnp.array([[10, 20, 30, 40]])  # shape (1, 4)

print("coluna.shape :", coluna.shape, " — 3 linhas, 1 coluna")
print("linha.shape  :", linha.shape, " — 1 linha, 4 colunas")
print()

resultado = coluna + linha   # (3,1) + (1,4) → (3,4)
print("coluna + linha → shape", resultado.shape)
print(resultado)

# %% [markdown]
# ### Produtos de matrizes — a operação central do ML
#
# Uma camada de rede neural faz essencialmente: $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$.
# Vamos ver as várias formas de multiplicar matrizes em JAX.

# %%
M = jnp.array([[1.0, 2.0],
               [3.0, 4.0]])
v = jnp.array([1.0, 0.5])

# Produto matriz-vetor
print("M @ v              :", M @ v)
print("jnp.dot(M, v)      :", jnp.dot(M, v))
print("einsum('ij,j->i')  :", jnp.einsum('ij,j->i', M, v))
print()

# Produto de matrizes
N = jnp.array([[0.5, 0.0],
               [0.0, 2.0]])
print("M @ N (produto de matrizes):\n", M @ N)
print()

# CUIDADO: * é elemento a elemento, NÃO é produto de matrizes!
print("M * N (elemento a elemento — diferente de M @ N!):\n", M * N)

# %% [markdown]
# ---
# ## 2. CPU vs GPU — por que ML ficou viável
#
# A GPU (*Runtime > Change runtime type* no Colab) tem **milhares de
# processadores simples em paralelo**, ideais para multiplicar matrizes grandes.
# É isso que torna o treino de redes neurais possível.

# %%
TAMANHO = 2000  # matriz 2000 x 2000

A_mat = jax.random.normal(jax.random.PRNGKey(0), (TAMANHO, TAMANHO))
B_mat = jax.random.normal(jax.random.PRNGKey(1), (TAMANHO, TAMANHO))

# Primeira chamada compila o kernel — descartamos
_ = (A_mat @ B_mat).block_until_ready()

# Agora medimos
t0 = time.perf_counter()
_ = (A_mat @ B_mat).block_until_ready()
dt = time.perf_counter() - t0

dispositivo = jax.devices()[0].device_kind
print(f"Multiplicação {TAMANHO}x{TAMANHO}")
print(f"  Dispositivo : {dispositivo}")
print(f"  Tempo       : {dt*1e3:.1f} ms")
print()
print("No Colab com GPU, este mesmo cálculo leva <5 ms — 50-100x mais rápido!")

# %% [markdown]
# ---
# ## 3. Matplotlib — nosso primeiro gráfico
#
# Vamos plotar uma **senoide amortecida** — a função que usaremos
# como alvo ao longo de todo o notebook:
#
# $$y = A\,\sin\!\left(\frac{2\pi x}{\lambda}\right)\,e^{-x/\tau}$$

# %%
def senoide_amortecida(x, A=1.5, lam=2.0, tau=6.0):
    """Senoide amortecida — a função-alvo do nosso exercício."""
    return A * jnp.sin(2.0 * jnp.pi * x / lam) * jnp.exp(-x / tau)


x_plot = jnp.linspace(0, 4 * jnp.pi, 300)
y_plot = senoide_amortecida(x_plot)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(x_plot, y_plot, lw=2, color="#2980b9",
        label=r"$y = A\sin(2\pi x/\lambda)\,e^{-x/\tau}$")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Senoide amortecida — a função que vamos aprender a ajustar")
ax.legend()
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 🟣 (Opcional) Diferenciação automática — `jax.grad`
#
# > Esta seção é opcional. Execute se houver tempo.
#
# O JAX calcula **derivadas exatas** de qualquer função Python com `jax.grad`.
# É isto que torna o treino de redes neurais possível.

# %%
# Exemplo simples: f(x) = x²  →  f'(x) = 2x
def f(x):
    return x ** 2

df_dx = jax.grad(f)   # df_dx é uma FUNÇÃO Python!

print("f(x) = x²")
print(f"f'(3.0) = {df_dx(3.0):.1f}   (esperado: 2 x 3 = 6.0)")
print(f"f'(5.0) = {df_dx(5.0):.1f}   (esperado: 2 x 5 = 10.0)")

# %%
# Derivada da senoide amortecida em relação a x
# jax.vmap aplica a função a cada elemento do array automaticamente
df_senoide = jax.vmap(jax.grad(senoide_amortecida))

x_ad = jnp.linspace(0.05, 4.0 * jnp.pi, 200)
y_ad = senoide_amortecida(x_ad)
dy_ad = df_senoide(x_ad)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(x_ad, y_ad, lw=2, color="#2980b9", label=r"$f(x)$")
ax.plot(x_ad, dy_ad, lw=2, color="#e74c3c", label=r"$f'(x) = \mathrm{d}f/\mathrm{d}x$")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("x")
ax.set_title("Função e derivada — calculadas automaticamente pelo JAX")
ax.legend()
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()

print("jax.grad calculou a derivada EXATA — sem fórmula analítica!")
print("É isto que usaremos para treinar redes neurais:")
print("  derivar a perda em relação aos PESOS do modelo.")

# %% [markdown]
# ---
# ## 🟣 (Opcional) `jax.jit` e `jax.vmap`
#
# > Seção opcional. Estes são dois "superpoderes" do JAX:
# >
# > - `jax.jit` — compila a função para XLA, acelerando 10-100x.
# > - `jax.vmap` — vetoriza automaticamente sobre um eixo de batch.

# %%
# ── jax.jit: compilação para XLA ─────────────────────────────────────────────
def somar_quadrados(x):
    return jnp.sum(x ** 2)

somar_jit = jax.jit(somar_quadrados)

v_grande = jnp.ones(10_000_000)
_ = somar_jit(v_grande).block_until_ready()   # primeiro call compila

t0 = time.perf_counter()
_ = somar_quadrados(v_grande).block_until_ready()
dt_sem = time.perf_counter() - t0

t0 = time.perf_counter()
_ = somar_jit(v_grande).block_until_ready()
dt_com = time.perf_counter() - t0

print(f"Sem jit : {dt_sem*1e3:.2f} ms")
print(f"Com jit : {dt_com*1e3:.2f} ms")
print(f"Aceleração: {dt_sem/max(dt_com, 1e-9):.1f}x")

# ── jax.vmap: vetorização automática ─────────────────────────────────────────
print()

def distancia(a, b):
    return jnp.sqrt(jnp.sum((a - b) ** 2))

dist_batch = jax.vmap(distancia, in_axes=(0, 0))

pts_a = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
pts_b = jnp.array([[1.0, 1.0], [4.0, 5.0], [2.0, 3.0]])
print("Distâncias (vmap):", dist_batch(pts_a, pts_b))

# %% [markdown]
# ---
# ## Exercício Central: Regressão com Rede Neural
#
# Vamos construir e treinar uma **rede neural completamente conexa (FCNN)**
# para ajustar a senoide amortecida com ruído gaussiano:
#
# $$y = A \sin\!\left(\frac{2\pi x}{\lambda}\right) e^{-x/\tau} + \varepsilon,
#    \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$
#
# Parâmetros: $A = 1.5$, $\lambda = 2$, $\tau = 6$, $\sigma = 0.15$.
#
# **Faremos o exercício duas vezes:**
#
# | | Bloco A — do zero | Bloco B — com bibliotecas |
# |---|---|---|
# | Modelo | matrizes explícitas | `eqx.nn.MLP` (Equinox) |
# | Forward | `h = tanh(x @ W + b)` | `modelo(x)` |
# | Otimizador | SGD+momentum na mão | `optax.adam` (Optax) |
#
# Ao final, demonstramos **sobreajuste** com uma rede grande demais.

# %% [markdown]
# ### Gerar os dados

# %%
# ── Parâmetros ────────────────────────────────────────────────────────────────
N_DADOS  = 200
SIGMA_EP = 0.15
X_MAX    = 4.0 * jnp.pi

# ── Gerar dados ──────────────────────────────────────────────────────────────
key_dados, k_x, k_ruido = jax.random.split(KEY, 3)

x_dados      = jnp.sort(jax.random.uniform(k_x, (N_DADOS,),
                                            minval=0.0, maxval=float(X_MAX)))
y_verdadeiro = senoide_amortecida(x_dados)
ruido        = jax.random.normal(k_ruido, (N_DADOS,)) * SIGMA_EP
y_ruidoso    = y_verdadeiro + ruido

# ── Normalizar x para [-1, 1] (essencial para ativações tanh) ────────────────
def normalizar_x(x):
    """Mapeia [0, X_MAX] para [-1, 1]."""
    return 2.0 * x / X_MAX - 1.0

x_norm = normalizar_x(x_dados)

# ── Preparar inputs para a rede: shape (N, 1) ────────────────────────────────
x_in = x_norm.reshape(-1, 1)
y_in = y_ruidoso

# ── Grid denso para plotar ───────────────────────────────────────────────────
x_grade = jnp.linspace(0.0, float(X_MAX), 500)
xg_in   = normalizar_x(x_grade).reshape(-1, 1)
y_grade = senoide_amortecida(x_grade)

print(f"Dados: {N_DADOS} pontos, x em [0, 4pi]")
print(f"Input normalizado: x_in.shape = {x_in.shape}")

# %%
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.scatter(x_dados, y_ruidoso, s=12, alpha=0.5, color="#aaaaaa",
           label=f"dados ruidosos (N={N_DADOS})", zorder=2)
ax.plot(x_grade, y_grade, "-", lw=2, color="#2980b9",
        label="função verdadeira", zorder=3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Dados de treino: senoide amortecida + ruído gaussiano")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Piso do ruído (sigma² = {SIGMA_EP**2:.4f})"
      " — nenhum modelo deveria ter perda menor.")

# %% [markdown]
# ---
# ## Bloco A — FCNN do zero
#
# Vamos construir tudo explicitamente: pesos, forward pass, perda, gradiente,
# e o loop de treino.

# %% [markdown]
# ### Inicialização dos pesos
#
# Os parâmetros são uma lista de tuplas $(W, b)$, uma por camada.
# Usamos inicialização de **Glorot**: escala proporcional ao tamanho das camadas.

# %%
def init_params(camadas, chave):
    """Cria pesos (W, b) para cada camada com inicialização de Glorot."""
    params = []
    for i in range(len(camadas) - 1):
        chave, k = jax.random.split(chave)
        n_in, n_out = camadas[i], camadas[i + 1]
        escala = jnp.sqrt(6.0 / (n_in + n_out))
        W = jax.random.uniform(k, (n_in, n_out), minval=-escala, maxval=escala)
        b = jnp.zeros(n_out)
        params.append((W, b))
    return params


# Rede [1 -> 32 -> 32 -> 1]
CAMADAS = [1, 32, 32, 1]
params_a = init_params(CAMADAS, jax.random.PRNGKey(0))

print("Rede", CAMADAS)
for i, (W, b) in enumerate(params_a):
    print(f"  Camada {i}: W {W.shape}, b {b.shape}")
total = sum(W.size + b.size for W, b in params_a)
print(f"  Total: {total} parâmetros")

# %% [markdown]
# ### Forward pass
#
# Cada camada oculta aplica: $\mathbf{h} = \tanh(\mathbf{h}\,\mathbf{W} + \mathbf{b})$.
# A camada de saída é linear (sem ativação).

# %%
def forward(params, x):
    """Forward pass: tanh nas ocultas, linear na saída."""
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W_out, b_out = params[-1]
    return h @ W_out + b_out


# Teste rápido
y_teste = forward(params_a, x_in[:5])
print(f"Input  : shape {x_in[:5].shape}")
print(f"Output : shape {y_teste.shape}")
print(f"Predições iniciais (devem ser pequenas): {y_teste.squeeze()}")

# %% [markdown]
# ### Função de perda — MSE
#
# $$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$
#
# É esta função que `jax.grad` vai diferenciar.

# %%
def perda_mse(params, x_batch, y_batch):
    """Erro quadrático médio."""
    y_pred = forward(params, x_batch).squeeze(-1)
    return jnp.mean((y_pred - y_batch) ** 2)


perda_ini = float(perda_mse(params_a, x_in, y_in))
var_y     = float(jnp.var(y_in))
print(f"Perda inicial (modelo aleatório) : {perda_ini:.4f}")
print(f"Variância de y_ruidoso           : {var_y:.4f}")
print("(Esperado: perda próxima da variância quando o modelo prevê ~0)")

# %% [markdown]
# ### Treino — SGD com momentum e mini-batches
#
# A cada passo, calculamos o gradiente num mini-batch e atualizamos os pesos.
# Usamos **momentum** — uma "memória" dos gradientes anteriores que suaviza
# o caminho e acelera a convergência:
#
# $$\mathbf{v} \leftarrow \beta\,\mathbf{v} + \nabla\mathcal{L}$$
# $$\mathbf{W} \leftarrow \mathbf{W} - \eta\,\mathbf{v}$$
#
# Sem momentum ($\beta = 0$), o SGD puro converge muito lentamente para
# redes com muitas camadas.

# %%
N_EPOCAS_A = 2000
LR_A       = 0.05      # taxa de aprendizado
MOMENTUM   = 0.9       # coeficiente de momentum
BATCH_SIZE = 50

EPOCAS_FOTO = [0, 300, 800, 2000]  # snapshots para a "figura-troféu"

# Função de gradiente compilada (jit acelera o cálculo)
grad_fn = jax.jit(jax.grad(perda_mse))

# Inicializar velocidade (mesma estrutura dos params, tudo zero)
vel_a = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in params_a]

# Guardar snapshots e histórico de perda
fotos_a     = {0: list(params_a)}   # snapshot antes do treino
historico_a = []

chave_treino = jax.random.PRNGKey(1)

print(f"Treinando {N_EPOCAS_A} épocas — SGD+momentum"
      f" (lr={LR_A}, mom={MOMENTUM}, batch={BATCH_SIZE})")
print(f"{'Época':>8}  {'Perda':>10}")
print("-" * 22)

t0 = time.perf_counter()
for epoca in range(1, N_EPOCAS_A + 1):
    # Embaralhar
    chave_treino, chave_perm = jax.random.split(chave_treino)
    perm  = jax.random.permutation(chave_perm, N_DADOS)
    x_emb = x_in[perm]
    y_emb = y_in[perm]

    # Mini-batches
    for i in range(0, N_DADOS, BATCH_SIZE):
        x_b = x_emb[i:i + BATCH_SIZE]
        y_b = y_emb[i:i + BATCH_SIZE]
        grads = grad_fn(params_a, x_b, y_b)

        # Atualização com momentum — escrita explicitamente
        vel_a = [(MOMENTUM * vW + dW, MOMENTUM * vb + db)
                 for (vW, vb), (dW, db) in zip(vel_a, grads)]
        params_a = [(W - LR_A * vW, b - LR_A * vb)
                    for (W, b), (vW, vb) in zip(params_a, vel_a)]

    # Snapshot
    if epoca in EPOCAS_FOTO:
        fotos_a[epoca] = list(params_a)

    # Monitorar
    if epoca % 100 == 0 or epoca == 1:
        perda = float(perda_mse(params_a, x_in, y_in))
        historico_a.append((epoca, perda))
        if epoca <= 1 or epoca % 500 == 0:
            print(f"{epoca:>8}  {perda:>10.6f}")

dt_a = time.perf_counter() - t0
perda_final_a = float(perda_mse(params_a, x_in, y_in))
print(f"\nConcluído em {dt_a:.1f}s")
print(f"Perda final   : {perda_final_a:.6f}")
print(f"Piso (sigma²) : {SIGMA_EP**2:.4f}")

# %%
# Curva de aprendizado
epocas_h, perdas_h = zip(*historico_a)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(epocas_h, perdas_h, "-o", ms=3, color="#2980b9")
ax.axhline(SIGMA_EP**2, color="red", ls="--", lw=1,
           label=f"piso do ruído sigma² = {SIGMA_EP**2:.4f}")
ax.set_xlabel("Época")
ax.set_ylabel("Perda (MSE)")
ax.set_title("Bloco A — Curva de aprendizado (SGD + momentum)")
ax.legend()
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Figura-troféu — o modelo aprendendo época a época

# %%
fig, axes_trophy = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
fig.suptitle("Bloco A — Progressão do treino (SGD + momentum, do zero)",
             fontsize=13, fontweight="bold")

for ax, ep in zip(axes_trophy, EPOCAS_FOTO):
    p_snap   = fotos_a[ep]
    y_pred_s = forward(p_snap, xg_in).squeeze(-1)
    perda_s  = float(perda_mse(p_snap, x_in, y_in))

    ax.scatter(x_dados, y_ruidoso, s=8, alpha=0.4, color="#aaaaaa")
    ax.plot(x_grade, y_grade, "--", lw=1.5, color="#2980b9", label="verdadeira")
    ax.plot(x_grade, y_pred_s, "-",  lw=2.0, color="#e74c3c", label="modelo")
    ax.set_title(f"Época {ep}\n(perda = {perda_s:.4f})", fontsize=10)
    ax.set_xlabel("x")
    ax.set_xlim(0, float(X_MAX))
    ax.set_ylim(-2.2, 2.2)
    ax.grid(True, alpha=0.3)

axes_trophy[0].set_ylabel("y")
axes_trophy[-1].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Bloco B — Mesmo exercício com Equinox + Optax
#
# Agora reproduzimos o Bloco A usando duas bibliotecas do ecossistema JAX:
#
# - **Equinox** — define modelos como módulos (encapsula `init_params` + `forward`)
# - **Optax** — fornece otimizadores prontos (Adam, SGD com momentum, etc.)
#
# | Bloco A (do zero) | Bloco B (bibliotecas) |
# |---|---|
# | `init_params(camadas, chave)` | `eqx.nn.MLP(...)` |
# | `forward(params, x)` | `jax.vmap(modelo)(x)` |
# | `jax.grad` + SGD+momentum manual | `eqx.filter_value_and_grad` + `optax.adam` |

# %% [markdown]
# ### Definir o modelo

# %%
import equinox as eqx
import optax

# Uma linha substitui init_params + forward!
modelo_b = eqx.nn.MLP(
    in_size=1, out_size=1,
    width_size=32, depth=2,        # [1, 32, 32, 1] — mesma arquitetura
    activation=jnp.tanh,
    key=jax.random.PRNGKey(10),
)

# Teste — jax.vmap aplica o modelo a cada amostra do batch
y_teste_b = jax.vmap(modelo_b)(x_in[:5])
print(f"Input  : shape {x_in[:5].shape}")
print(f"Output : shape {y_teste_b.shape}")
print(f"Predições iniciais: {y_teste_b.squeeze()}")

# %% [markdown]
# ### Perda, otimizador e passo de treino

# %%
def perda_eqx(modelo, x_batch, y_batch):
    """MSE — versão para modelo Equinox."""
    y_pred = jax.vmap(modelo)(x_batch).squeeze(-1)
    return jnp.mean((y_pred - y_batch) ** 2)


# Otimizador Adam — substitui o SGD+momentum manual
otimizador_b = optax.adam(learning_rate=5e-3)
opt_state_b  = otimizador_b.init(eqx.filter(modelo_b, eqx.is_array))


@eqx.filter_jit
def passo_b(modelo, estado, x, y):
    """Um passo de treino: gradiente + atualização Adam."""
    perda, grads = eqx.filter_value_and_grad(perda_eqx)(modelo, x, y)
    atualizacoes, estado = otimizador_b.update(grads, estado, modelo)
    modelo = eqx.apply_updates(modelo, atualizacoes)
    return modelo, estado, perda

# %% [markdown]
# ### Loop de treino

# %%
N_EPOCAS_B  = 800
historico_b = []
chave_b     = jax.random.PRNGKey(2)

print(f"Treinando {N_EPOCAS_B} épocas — Adam (lr=0.005, batch={BATCH_SIZE})")

t0 = time.perf_counter()
for epoca in range(1, N_EPOCAS_B + 1):
    chave_b, chave_perm = jax.random.split(chave_b)
    perm  = jax.random.permutation(chave_perm, N_DADOS)
    x_emb = x_in[perm]
    y_emb = y_in[perm]

    for i in range(0, N_DADOS, BATCH_SIZE):
        x_b = x_emb[i:i + BATCH_SIZE]
        y_b = y_emb[i:i + BATCH_SIZE]
        modelo_b, opt_state_b, _ = passo_b(modelo_b, opt_state_b, x_b, y_b)

    if epoca % 100 == 0 or epoca == 1:
        perda = float(perda_eqx(modelo_b, x_in, y_in))
        historico_b.append((epoca, perda))
        if epoca <= 1 or epoca % 200 == 0:
            print(f"  Época {epoca:4d}  perda = {perda:.6f}")

dt_b = time.perf_counter() - t0
perda_final_b = float(perda_eqx(modelo_b, x_in, y_in))
print(f"\nConcluído em {dt_b:.1f}s")
print(f"Perda final : {perda_final_b:.6f}")

# %% [markdown]
# ### Comparação: Bloco A vs Bloco B

# %%
y_pred_a_plot = forward(params_a, xg_in).squeeze(-1)
y_pred_b_plot = jax.vmap(modelo_b)(xg_in).squeeze(-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
fig.suptitle("Do zero vs. Equinox + Optax", fontsize=13, fontweight="bold")

for ax, y_pred, titulo, cor in [
    (ax1, y_pred_a_plot,
     f"Bloco A — SGD+momentum\nperda = {perda_final_a:.4f}", "#e74c3c"),
    (ax2, y_pred_b_plot,
     f"Bloco B — Adam + Equinox\nperda = {perda_final_b:.4f}", "#27ae60"),
]:
    ax.scatter(x_dados, y_ruidoso, s=8, alpha=0.4, color="#aaaaaa")
    ax.plot(x_grade, y_grade, "--", lw=1.5, color="#2980b9", label="verdadeira")
    ax.plot(x_grade, y_pred, "-", lw=2, color=cor, label="modelo")
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("x")
    ax.set_xlim(0, float(X_MAX))
    ax.set_ylim(-2.2, 2.2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

ax1.set_ylabel("y")
plt.tight_layout()
plt.show()

print("Ambos aprendem a mesma função — as bibliotecas simplificam o código,")
print("não mudam a matemática subjacente!")

# %% [markdown]
# ---
# ## 🟡 Pergunta-relâmpago — o que acontece com uma rede maior?
#
# > Imagine que trocamos a rede `[1, 32, 32, 1]` por uma rede muito maior,
# > `[1, 128, 128, 128, 1]`, e treinamos por 3 000 épocas.
# >
# > **O que acontece com o ajuste?**
# >
# > Opções:
# > - (A) Fica muito melhor — mais parâmetros = melhor modelo
# > - (B) Piora — gradientes muito pequenos
# > - (C) Memoriza o ruído — sobreajuste

# %% [markdown]
# ### Sobreajuste (*overfitting*)

# %%
# Rede grande com Equinox
modelo_of = eqx.nn.MLP(
    in_size=1, out_size=1,
    width_size=128, depth=3,       # [1, 128, 128, 128, 1]
    activation=jnp.tanh,
    key=jax.random.PRNGKey(99),
)

otimizador_of = optax.adam(learning_rate=3e-3)
opt_state_of  = otimizador_of.init(eqx.filter(modelo_of, eqx.is_array))


@eqx.filter_jit
def passo_of(modelo, estado, x, y):
    perda, grads = eqx.filter_value_and_grad(perda_eqx)(modelo, x, y)
    atualizacoes, estado = otimizador_of.update(grads, estado, modelo)
    modelo = eqx.apply_updates(modelo, atualizacoes)
    return modelo, estado, perda


N_EPOCAS_OF = 3000
print(f"Treinando rede grande [1, 128, 128, 128, 1] por {N_EPOCAS_OF} épocas...")

t0 = time.perf_counter()
for epoca in range(1, N_EPOCAS_OF + 1):
    modelo_of, opt_state_of, perda_of = passo_of(
        modelo_of, opt_state_of, x_in, y_in)
    if epoca % 500 == 0:
        print(f"  Época {epoca:5d}  perda = {float(perda_of):.6f}")

dt_of = time.perf_counter() - t0
perda_final_of = float(perda_eqx(modelo_of, x_in, y_in))
print(f"\nConcluído em {dt_of:.1f}s")
print(f"Perda final            : {perda_final_of:.6f}")
print(f"Piso do ruído (sigma²) : {SIGMA_EP**2:.4f}")
if perda_final_of < SIGMA_EP**2:
    print("Sobreajuste confirmado: perda < piso do ruído!")

# %%
y_pred_bom = jax.vmap(modelo_b)(xg_in).squeeze(-1)
y_pred_of  = jax.vmap(modelo_of)(xg_in).squeeze(-1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
fig.suptitle("Generalização vs. Sobreajuste (overfitting)",
             fontsize=13, fontweight="bold")

# Rede adequada
ax1.scatter(x_dados, y_ruidoso, s=8, alpha=0.4, color="#aaaaaa",
            label="dados ruidosos")
ax1.plot(x_grade, y_grade, "--", lw=1.5, color="#2980b9",
         label="função verdadeira")
ax1.plot(x_grade, y_pred_bom, "-", lw=2, color="#27ae60",
         label="[1, 32, 32, 1]")
ax1.set_title(f"Rede adequada — generaliza\n"
              f"perda = {perda_final_b:.4f}", fontsize=10)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_xlim(0, float(X_MAX))
ax1.set_ylim(-2.2, 2.2)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Rede grande (sobreajuste)
ax2.scatter(x_dados, y_ruidoso, s=8, alpha=0.4, color="#aaaaaa",
            label="dados ruidosos")
ax2.plot(x_grade, y_grade, "--", lw=1.5, color="#2980b9",
         label="função verdadeira")
ax2.plot(x_grade, y_pred_of, "-", lw=2, color="#e67e22",
         label="[1, 128, 128, 128, 1]")
ax2.set_title(f"Rede grande — sobreajuste!\n"
              f"perda = {perda_final_of:.4f} < sigma²={SIGMA_EP**2:.4f}",
              fontsize=10)
ax2.set_xlabel("x")
ax2.set_xlim(0, float(X_MAX))
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("A rede grande memorizou o ruído — não generalizou.")
print("Na quarta-feira, veremos o que acontece quando a distribuição")
print("de TESTE é diferente da distribuição de TREINO.")

# %% [markdown]
# ---
# ## Mapa de vocabulário
#
# | O que fizemos | Jargão | Significado |
# |---|---|---|
# | Função alvo ruidosa | **tarefa** (task) | O que queremos que a rede aprenda |
# | `init_params` / `eqx.nn.MLP` | **modelo** | A função parametrizada |
# | `forward(params, x)` / `model(x)` | **inferência** | Aplicar o modelo a dados |
# | `perda_mse` | **função de perda** | Mede o erro do modelo |
# | `jax.grad(perda_mse)` | **gradiente** | Direção de subida mais íngreme |
# | `W -= lr * vel` / `optax.adam` | **otimizador** | Regra de atualização dos pesos |
# | Mini-batch | **SGD** | Gradiente estocástico (subconjunto dos dados) |
# | Modelo que generaliza | **generalização** | O objetivo real |
# | Modelo que memoriza ruído | **sobreajuste** (overfitting) | O que queremos evitar |
# | Equinox | **framework de modelos** | Define redes como módulos |
# | Optax | **biblioteca de otimizadores** | Fornece Adam, SGD, etc. |
#
# > **Takeaway:** *Treinar uma rede = descer o gradiente de uma função de perda.*
# > As bibliotecas (Equinox, Optax) encapsulam a matemática, mas o fundamento é o mesmo.

# %% [markdown]
# ## 🟡 Para casa
#
# 1. **Outra função:** substitua a senoide por $|\sin(x)|$ ou $x^3 e^{-x}$.
#    Precisou mudar a arquitetura ou o learning rate?
#
# 2. **Curva de validação:** divida os 200 pontos em 160 treino + 40 validação.
#    Plote ambas as perdas ao longo das épocas. Onde começa o sobreajuste?
#
# 3. **(🟣 Desafio) SGD puro vs. momentum:**
#    Remova o momentum ($\beta = 0$) do Bloco A e compare a convergência.
#    Quanto mais lento é? Tente compensar aumentando a taxa de aprendizado.
