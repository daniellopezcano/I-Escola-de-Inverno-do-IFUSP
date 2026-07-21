# Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos

Repositório de materiais da **I Escola de Inverno do IFUSP** — um minicurso de 4 aulas que percorre uma única ideia: *redes neurais aprendem coordenadas*, o sistema de referência em que a distância passa a significar semelhança. Cada dia combina um bloco teórico (slides) com um bloco prático (notebook em JAX, rodável no Google Colab), do primeiro contato com JAX até dois estudos de caso reais em cosmologia observacional.

|                 |                                                                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Evento**      | I Escola de Inverno do IFUSP — ["Redes neurais: princípios e aplicações na Física"](https://portal.if.usp.br/pesquisa/pt-br/node/2745) |
| **Datas**       | 21–24 de julho de 2026                                                                                                                 |
| **Local**       | Instituto de Física da USP - Auditório Abrahão de Moraes.                                                                              |
| **Instrutor**   | Dr. Daniel López-Cano                                                                                                                  |
| **Formato**     | 4 dias · 2 blocos de 40 min por dia (1 teórico + 1 prático)                                                                            |
| **Repositório** | [github.com/daniellopezcano/I-Escola-de-Inverno-do-IFUSP](https://github.com/daniellopezcano/I-Escola-de-Inverno-do-IFUSP)             |

---
## Estrutura do repositório

| Caminho                                                          | Conteúdo                                                                                          |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [`course-materials/`](course-materials/)                         | 8 arquivos `LxBy.md` — o material de cada bloco (narrativa teórica ou link do notebook), em pt-BR |
| [`jax-examples/notebooks/`](jax-examples/notebooks/)             | os 3 notebooks Colab (JAX) dos blocos práticos                                                    |
| [`jax-examples/requirements.txt`](jax-examples/requirements.txt) | dependências para rodar os notebooks localmente                                                   |
| [`assets/references/`](assets/references/)                       | os dois artigos científicos por trás da Aula 4                                                    |
| [`assets/figures_and_materials/`](assets/figures_and_materials/) | figuras de apoio usadas na preparação das aulas                                                   |
| [`COMO_USAR.md`](COMO_USAR.md)                                   | guia prático de uso deste repositório                                                             |
| [`LICENSE`](LICENSE)                                             | licença MIT                                                                                       |
| [`CITATION.cff`](CITATION.cff)                                   | como citar este material                                                                          |

---
## Blocos

| Bloco                            | Tipo     | Slides / Colab                                                                                                                                                                                                                        |
| -------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [L1B1](course-materials/L1B1.md) | teoria   | [Slides](https://docs.google.com/presentation/d/1urJoVZ1Oeko21DEa6jq737MJcpetG1whUMFMDD05oq0/edit?usp=drive_link)                                                                                                                     |
| [L1B2](course-materials/L1B2.md) | notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L1B2_caixa_de_ferramentas.ipynb)  |
| [L2B1](course-materials/L2B1.md) | teoria   | [Slides](https://docs.google.com/presentation/d/13U9gfE1-IIt9lJbpV18IznFxp768jyux5gatREnji-w/edit?usp=sharing)                                                                                                                        |
| [L2B2](course-materials/L2B2.md) | notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L2B2_domain_shift_toy.ipynb)      |
| [L3B1](course-materials/L3B1.md) | teoria   | [Slides](https://docs.google.com/presentation/d/1N-hVLYVonRZqiuSYis2bbJG7moPyTj3N9Cf_e3LC1sI/edit?usp=sharing)                                                                                                                        |
| [L3B2](course-materials/L3B2.md) | notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L3B2_contrastive_embedding.ipynb) |
| [L4B1](course-materials/L4B1.md) | teoria   | [Slides](https://docs.google.com/presentation/d/1TO9SUE9_11Tf5H3PCXMgtE4I7jCwh4bhqmC3UIpp18M/edit?usp=sharing)                                                                                                                        |
| [L4B2](course-materials/L4B2.md) | teoria   | [Slides](https://docs.google.com/presentation/d/1cp1CZIwPkHq28DFxMSgwyQ2K2t7z7phDAu6fZ54meEM/edit?usp=sharing)                                                                                                                        |

Cada bloco teórico (`B1`) é uma narrativa compacta com o link do Google Slides; cada bloco prático (`B2`) é um notebook JAX que gera todos os dados e cálculos no próprio runtime do Colab. Veja [COMO_USAR.md](COMO_USAR.md) para o passo a passo.

---
## O arco dos quatro dias

O curso segue uma ideia central: **redes neurais aprendem coordenadas** — um _encoder_ encontra o sistema de referência onde objetos semelhantes ficam próximos. Os quatro dias desenvolvem essa ideia em ordem: o mapa do campo → mudança de domínio → geometria de representações → cosmologia e estudos de caso reais. Cada dia tem um bloco teórico (slides) e um bloco prático (notebook JAX no Colab).

### Dia 1 — O mapa e as ferramentas (21 de julho)

**[L1B1](course-materials/L1B1.md)** _teoria_ — Panorama de ML em cosmologia: o encontro entre o dilúvio de dados observacionais e a ascensão do aprendizado profundo. O que o ML faz e não faz em astrofísica, introduzindo espaço latente, adaptação de domínio e aprendizagem contrastiva como respostas às suas limitações.

**[L1B2](course-materials/L1B2.md)** _notebook_ [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L1B2_caixa_de_ferramentas.ipynb) — Fundamentos de JAX (arrays, broadcasting, `jax.grad`), CPU×GPU, e uma rede ajustando uma senoide ruidosa **do zero** e depois **com bibliotecas** (Equinox/Optax), linha a linha. Fecha com sobreajuste e extrapolação, plantando o tema do Dia 2.

### Dia 2 — Quando os dados mudam (22 de julho)

**[L2B1](course-materials/L2B1.md)** _teoria_ — Taxonomia da mudança de domínio (covariate, prior, concept shift), o perigo de erros com alta confiança sob mudança de distribuição, e o mapa de estratégias de mitigação até transferência de aprendizado / ajuste fino.

**[L2B2](course-materials/L2B2.md)** _notebook_ [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L2B2_domain_shift_toy.ipynb) — Classificador de brinquedo em 2D: regiões de decisão, matriz de confusão, ROC/AUC e calibração, com o domínio alvo gerado por rotação controlada.

### Dia 3 — Esculpindo representações (23 de julho)

**[L3B1](course-materials/L3B1.md)** _teoria_ — Por que espaços latentes dominam o ML atual; o princípio contrastivo (positivos/negativos declaram invariâncias); a perda **InfoNCE** em detalhe (temperatura, negativos, colapso); e o **t-SNE** com suas limitações honestas.

**[L3B2](course-materials/L3B2.md)** _notebook_ [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L3B2_contrastive_embedding.ipynb) — MNIST em quatro partes: pixels enganam, um encoder com gargalo de 2D organiza classes sozinho, augmentações apropriadas, e treino autossupervisionado com InfoNCE validado por t-SNE e sondagem linear.

### Dia 4 — Da cosmologia à fronteira (24 de julho)

**[L4B1](course-materials/L4B1.md)** _teoria_ — Da Relatividade Geral ao ΛCDM (matéria e energia escura); panorama comparativo das sondas observacionais (clustering, lenteamento, CMB, Lyman-α, ondas gravitacionais); lógica bayesiana de inferência e o papel dos simuladores quando a verossimilhança é implícita.

**[L4B2](course-materials/L4B2.md)** _teoria_ — Dois estudos de caso: (1) segmentação de halos de matéria escura via perda contrastiva tipo Weinberger (atração/repulsão), aplicando o princípio do Dia 3 à física; (2) classificação de objetos no J-PAS com adaptação de domínio semi-supervisionada (mocks → observações reais), incluindo a diferença entre simulação e observação.

---

## Licença e citação

Este material é distribuído sob a [licença MIT](LICENSE) — use, adapte e reutilize livremente, com atribuição.

Se este material for útil para sua aula, pesquisa ou apresentação, uma citação é bem-vinda:

> López-Cano, D. (2026). *Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos* — I Escola de Inverno do IFUSP. https://github.com/daniellopezcano/I-Escola-de-Inverno-do-IFUSP

Formato de citação também disponível em [`CITATION.cff`](CITATION.cff) (lido nativamente pelo botão "Cite this repository" do GitHub).

Os dois estudos de caso do Dia 4 têm publicações próprias, citáveis independentemente — ver as referências completas em [`course-materials/L4B2.md`](course-materials/L4B2.md#referências).
