# Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos

Repositório de materiais da **I Escola de Inverno do IFUSP** — um minicurso de 4 aulas que percorre uma única ideia: *redes neurais aprendem coordenadas*, o sistema de referência em que a distância passa a significar semelhança. Cada dia combina um bloco teórico (slides) com um bloco prático (notebook em JAX, rodável no Google Colab sem instalação), do primeiro contato com JAX até dois estudos de caso reais em cosmologia observacional.

| | |
|---|---|
| **Evento** | I Escola de Inverno do IFUSP — ["Redes neurais: princípios e aplicações na Física"](https://portal.if.usp.br/pesquisa/pt-br/node/2745) |
| **Datas** | 21–24 de julho de 2026 |
| **Local** | Instituto de Física da USP |
| **Público** | ~130 estudantes de graduação (últimos anos) em Física |
| **Instrutor** | Dr. Daniel López-Cano |
| **Formato** | 4 dias · 2 blocos de 40 min por dia (1 teórico + 1 prático) |
| **Repositório** | [github.com/daniellopezcano/I-Escola-de-Inverno-do-IFUSP](https://github.com/daniellopezcano/I-Escola-de-Inverno-do-IFUSP) |

---

## Comece por aqui

Novo por aqui? Leia **[COMO_USAR.md](COMO_USAR.md)** — como abrir o material teórico, rodar os notebooks no Colab sem instalar nada, e o que fazer se algo travar. Curto e prático.

---

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| [`course-materials/`](course-materials/) | 8 arquivos `LxBy.md` — o material de cada bloco (narrativa teórica ou link do notebook), em pt-BR |
| [`jax-examples/notebooks/`](jax-examples/notebooks/) | os 3 notebooks Colab (JAX) dos blocos práticos |
| [`jax-examples/requirements.txt`](jax-examples/requirements.txt) | dependências para rodar os notebooks localmente |
| [`assets/references/`](assets/references/) | os dois artigos científicos por trás da Aula 4 |
| [`assets/figures_and_materials/`](assets/figures_and_materials/) | figuras de apoio usadas na preparação das aulas |
| [`COMO_USAR.md`](COMO_USAR.md) | guia prático de uso deste repositório |
| [`LICENSE`](LICENSE) | licença MIT |
| [`CITATION.cff`](CITATION.cff) | como citar este material |

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

O curso segue uma única linha argumentativa: **redes neurais aprendem coordenadas**. Assim como a mecânica analítica troca variáveis emaranhadas por modos normais ou pelo referencial do centro de massa, um _encoder_ neural encontra o sistema de coordenadas em que objetos semelhantes ficam próximos e objetos distintos ficam distantes. Os quatro dias percorrem essa ideia em ordem: primeiro o mapa do campo e as ferramentas; depois o que acontece quando o mundo muda sob os pés do modelo; em seguida como esculpir deliberadamente essas coordenadas; e, por fim, tudo isso aplicado à cosmologia e a dois estudos de caso reais.

Cada dia tem dois blocos de 40 minutos: um **bloco teórico**, acompanhado de slides, e um **bloco prático**, guiado por um notebook em JAX que roda no Google Colab.

### Dia 1 — O mapa e as ferramentas (21 de julho)

O primeiro dia responde a uma pergunta anterior a todas as outras: _o que é este campo, e por que ele importa para um físico hoje?_

#### [L1B1](course-materials/L1B1.md) — Aprendizado de máquina e Física: o mapa do território · _teoria_

Duas linhas do tempo paralelas abrem o curso. De um lado, o dilúvio de dados em cosmologia: 2dFGRS → SDSS-I/II → SDSS-III/IV → DESI DR1 → Rubin/LSST. Do outro, a ascensão do aprendizado de máquina: Deep Blue (1997) → ImageNet (2009) → AlexNet (2012) → GANs (2014) → Transformers (2017) → AlphaFold (2020) → ChatGPT (2022). As duas curvas se encontram exatamente onde este curso vive.

A seguir, um balanço honesto: o que o ML **faz** em astrofísica (emulação de simulações caras, reconhecimento de padrões, inferência baseada em simulações, problemas inversos, detecção de anomalias) e o que ele **não faz** (não extrapola de forma confiável, não certifica o próprio erro, herda os vieses dos dados, não modela causalidade e é difícil de interpretar). Os temas do curso surgem então como **respostas diretas a essas limitações**, unificados pela ideia de espaço latente — dados brutos → _encoder_ → representação, onde _distâncias cruas enganam e distâncias aprendidas revelam_. A **adaptação de domínio** alinha o treino à realidade; a **aprendizagem contrastiva** constrói o espaço em que distância significa semelhança.

#### [L1B2](course-materials/L1B2.md) — A caixa de ferramentas: Python, Colab e JAX · _notebook_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L1B2_caixa_de_ferramentas.ipynb) `jax-examples/notebooks/L1B2_caixa_de_ferramentas.ipynb`

Começamos pelo esqueleto matemático do ML, inteiramente em JAX: arrays, _shapes_, _broadcasting_ e as diferentes formas de multiplicar matrizes, tudo com prints explícitos para ver o que acontece. Só então medimos CPU × GPU — a diferença de velocidade que tornou o treino moderno viável — e passamos ao primeiro gráfico e às derivadas automáticas com `jax.grad`.

O exercício central aparece **duas vezes, lado a lado**: uma rede completamente conexa ajustando uma senoide amortecida ruidosa, primeiro _do zero_ (matrizes e vieses definidos à mão, ativações, perda MSE, retropropagação via `jax.grad` e passo de SGD escrito explicitamente) e depois _com bibliotecas_ (Equinox e Optax), mapeada uma-a-uma sobre a versão anterior — para que se veja exatamente quais linhas cada chamada de biblioteca substitui.

O fecho planta as sementes do Dia 2: uma célula de sobreajuste, a avaliação de um modelo pequeno e um grande sobre dados de teste novos, o desempenho fora do intervalo de treino (extrapolação) e o que acontece quando os dados vêm da mesma função com um parâmetro ligeiramente diferente.

### Dia 2 — Quando os dados mudam (22 de julho)

O segundo dia começa onde o primeiro terminou. O problema mais profundo não é a rede memorizar ruído: é os dados de aplicação virem de uma distribuição diferente da de treino.

#### [L2B1](course-materials/L2B1.md) — Mudança de domínio: quando o treino não é a prova · _teoria_

Todo o aprendizado supervisionado supõe, silenciosamente, que treino e teste vêm da mesma distribuição. A **mudança de domínio** é essa suposição falhando. Construímos a taxonomia com exemplos visuais — _covariate shift_ (a distribuição dos dados se desloca), _prior shift_ (as proporções entre classes mudam) e _concept shift_ (a própria regra muda) — e enquadramos o caso simulação → observação como a versão científica do problema.

Vem então o modo de falha mais perigoso do ML na ciência: sob mudança de domínio, modelos erram **com alta confiança**. Daí a distinção entre acurácia e calibração. O bloco fecha com o mapa das estratégias de mitigação, organizado pelo que se tem em mãos: aumento de dados e randomização de domínio; alinhamento de distribuições e reponderação; e, quando existem alguns rótulos do domínio alvo, **transferência de aprendizado e ajuste fino**.

#### [L2B2](course-materials/L2B2.md) — Quebrar, diagnosticar e consertar um classificador · _notebook_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L2B2_domain_shift_toy.ipynb) `jax-examples/notebooks/L2B2_domain_shift_toy.ipynb`

Um modelo de brinquedo em 2D onde tudo é visível, construído sobre o _benchmark_ clássico de adaptação de domínio.

Segue a caixa de ferramentas de avaliação, usada o dia inteiro: mapas de probabilidade sobre o espaço de atributos, regiões de decisão, matriz de confusão com TPR, PPV e F1 por classe, curvas ROC e AUC, calibração e mapas de erro. O domínio alvo é gerado por uma **rotação** — um único parâmetro que funciona como dial da severidade do desvio.

### Dia 3 — Esculpindo representações (23 de julho)

O terceiro dia coloca a geometria do espaço latente como protagonista. A pergunta de abertura é um quebra-cabeça: entre um dígito e a sua cópia deslocada alguns pixels, e o mesmo dígito comparado a outro diferente, qual par está mais próximo em distância euclidiana bruta? A resposta motiva tudo o que vem depois.

#### [L3B1](course-materials/L3B1.md) — Aprendizagem contrastiva: a geometria da similaridade · _teoria_

Primeiro, o que é um espaço latente e **por que ele domina o ML atual**: representações são a moeda reutilizável do aprendizado profundo — _encoders_ pré-treinados, transferência entre tarefas, busca por similaridade e recuperação de informação. É também o que torna possível a transferência do Dia 2.

Depois, o princípio contrastivo em uma frase: _escolha quais pares devem ficar perto (positivos) e quais devem ficar longe (negativos); a rede apenas geometriza a sua escolha_. Similaridade não se descobre — se declara. Os positivos vêm de rótulos ou de **augmentações**, e aqui está a ponte mais profunda com a física: escolher augmentações é declarar as **invariâncias** do problema (rotacionar diz que a orientação não importa; adicionar ruído diz que aquele nível de ruído é instrumental, não físico).

O centro matemático do bloco é a perda **InfoNCE**, tratada com profundidade: âncora, positivo e negativos; similaridade do cosseno sobre representações normalizadas; a leitura como problema de classificação (identificar o positivo entre os negativos, ou seja, um _softmax_ sobre similaridades); o papel da **temperatura** e sua leitura em mecânica estatística; o efeito do número de negativos; o problema do **colapso** e por que os negativos existem. Uma subseção mapeia os tipos conceituais de aprendizagem contrastiva e quando cada um se aplica, e outra compara perdas alternativas com prós e contras práticos.

Fecha o bloco o **t-SNE**, explicado de verdade — vizinhanças convertidas em probabilidades, o parâmetro de perplexidade, o núcleo de cauda pesada e o problema de aglomeração, e a divergência KL minimizada por gradiente — seguido de suas limitações honestas: distâncias entre grupos não têm significado, tamanhos e densidades não são fiéis, e a figura muda com a perplexidade e com a semente. A analogia é astronômica: constelações parecem grupos, mas suas estrelas estão a anos-luz umas das outras.

#### [L3B2](course-materials/L3B2.md) — Esculpindo um espaço de embeddings · _notebook_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/L3B2_contrastive_embedding.ipynb) `jax-examples/notebooks/L3B2_contrastive_embedding.ipynb`

Quatro partes sobre MNIST. Primeiro, a evidência de que **o espaço de pixels engana**: imagens do mesmo dígito podem estar mais distantes entre si do que de dígitos diferentes, e 784 dimensões não são interpretáveis.

Em seguida, um _encoder_ convolucional com **gargalo de 2 dimensões**, treinado como classificador comum: o espaço latente se organiza sozinho por classe, e dígitos visualmente próximos (4/9, 3/8) acabam vizinhos — estrutura emergente, e a primeira justificativa concreta para estudar espaços latentes.

A terceira parte é visual: augmentações apropriadas ao MNIST (deformações elásticas, recortes e redimensionamentos, transformações afins, ruído) mostradas em grade, junto com a ilustração explícita de um **par positivo** contra pares negativos — incluindo o alerta de que uma rotação exagerada transforma um 6 em 9 e destrói o rótulo.

Por fim, o treino **autossupervisionado com InfoNCE**: nenhum rótulo é usado, os positivos vêm das augmentações, e a estrutura por dígito reaparece no espaço aprendido — visualizada com t-SNE e verificada objetivamente por uma sondagem linear sobre a representação congelada.

### Dia 4 — Da cosmologia à fronteira (24 de julho)

O último dia inverte a perspectiva: em vez de partir do ML, parte-se da física. Primeiro o contexto cosmológico, depois os dois estudos de caso em que tudo o que foi construído nos três dias anteriores reaparece em escala real.

#### [L4B1](course-materials/L4B1.md) — Cosmologia: contexto físico e estado da arte · _teoria_

Um bloco deliberadamente **centrado na física**. Começa com o arco da Relatividade Geral até o ΛCDM: as equações de Einstein, o modelo FLRW erguido sobre homogeneidade e isotropia, as evidências de **matéria escura** (curvas de rotação) e de **energia escura** (diagrama de Hubble de supernovas), até a imagem moderna — CMB, idades escuras, reionização, formação de estruturas e expansão acelerada.

Segue um panorama das **sondas observacionais** que restringem a cosmologia — aglomeração de galáxias, lenteamento gravitacional fraco, radiação cósmica de fundo, floresta de Lyman-α e ondas gravitacionais — cada uma com o que observa, o que traça, em que época, o que informa e qual sua principal dificuldade. A mensagem é de complementaridade: épocas, escalas e sistemáticos diferentes, por isso combinados.

Depois, a lógica da inferência: parâmetros → modelo → observações, e o caminho de volta. Teorema de Bayes, MCMC com Metropolis–Hastings e — o ponto crucial — os **modelos implícitos**, em que não existe fórmula tratável para a verossimilhança, apenas um _prior_ e um simulador. É o que motiva a inferência baseada em simulações e o que torna as simulações indispensáveis. O bloco encerra explicando por que simulamos, o que são simulações de matéria escura não colisional e simulações hidrodinâmicas, e o compromisso inescapável entre volume, resolução e completude física.

#### [L4B2](course-materials/L4B2.md) — Simulações, ML e adaptação de domínio: dois estudos de caso · _teoria_

O compromisso que fechou o bloco anterior é o ponto de partida deste: simulações são caras, e o ML entra para aliviar o custo — emuladores de estatísticas de resumo, super-resolução condicionada a simulações baratas, emulação rápida de campos e correções aprendidas dentro de simulações de baixa resolução. Com a ressalva honesta de sempre: esses modelos herdam as hipóteses das simulações em que foram treinados.

O **primeiro estudo de caso** é o momento de reconhecimento do curso. Dado o campo de densidade inicial, quais partículas terminam no mesmo halo de matéria escura? O problema é formulado como segmentação panóptica e dividido em duas partes: _semântica_ ("é halo?", classificação com número fixo de classes) e _de instância_ ("qual halo?", número arbitrário de instâncias e rótulos invariantes por permutação — logo, sem perda diferenciável óbvia). A solução é uma perda contrastiva do tipo Weinberger, que projeta cada partícula num pseudo-espaço onde halos viram aglomerados, com um termo de **atração** ao centro da instância e um de **repulsão** entre centros. É exatamente o princípio do Dia 3, agora fazendo física.

O **segundo estudo de caso** é o experimento do Dia 2 em escala real. No J-PAS, dezenas de filtros estreitos produzem um "J-espectro" por objeto, e a tarefa é classificar galáxias, quasares de baixo e alto _redshift_ e estrelas. O domínio de origem são **mocks** construídos convolvendo espectros do DESI com as bandas do J-PAS (rótulos abundantes); o domínio alvo são as **observações reais**, cujos rótulos vêm apenas de um pequeno cruzamento com o DESI (rótulos escassos). Comparam-se três modelos — supervisionado apenas com observações, treinado em mocks e aplicado direto ao céu real, e o modelo **SSDA**, que reaproveita o anterior mantendo parte dos pesos fixa e readaptando o restante. As matrizes de confusão e as curvas ROC contam a história, com uma ressalva física: o corte em z ≈ 2,1 entre quasares de baixo e alto _redshift_ é arbitrário sobre uma variável contínua, e objetos próximos dele são ambíguos por natureza — distinguir "a física é difícil" de "o treino estava enviesado" é a habilidade mais transferível do curso.

Entre os dois, uma seção sobre o **espaço entre observações e simulações**: instrumentos, filtros, profundidade, PSF e funções de seleção diferentes fazem com que dois levantamentos do mesmo céu não sejam os mesmos dados; e simulações que não reproduzem exatamente as observações levam a inferências enviesadas — posteriores estreitas e erradas.

---

## Licença e citação

Este material é distribuído sob a [licença MIT](LICENSE) — use, adapte e reutilize livremente, com atribuição.

Se este material for útil para sua aula, pesquisa ou apresentação, uma citação é bem-vinda:

> López-Cano, D. (2026). *Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos* — I Escola de Inverno do IFUSP. https://github.com/daniellopezcano/I-Escola-de-Inverno-do-IFUSP

Formato de citação também disponível em [`CITATION.cff`](CITATION.cff) (lido nativamente pelo botão "Cite this repository" do GitHub).

Os dois estudos de caso do Dia 4 têm publicações próprias, citáveis independentemente — ver as referências completas em [`course-materials/L4B2.md`](course-materials/L4B2.md#referências).
