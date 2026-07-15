# Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos

**I Escola de Inverno do IFUSP** | 21–24 de julho de 2026 | Instituto de Física da USP
**Instrutor:** Daniel López Cano

> *Aprender é escolher coordenadas — e este curso ensina a escolhê-las, adaptá-las e desconfiar delas.*

---

## Como usar este material

Cada um dos oito arquivos de bloco nesta pasta é um guia duplo: a seção de estudante (em português) explica o conceito, mostra o raciocínio matemático e descreve o que será executado ao vivo; a seção `[!instructor]` (em inglês, colapsável no Obsidian) contém o plano de tempo, checklist de preparação e armadilhas comuns. Se você está usando o Obsidian, a navegação entre blocos acontece pelos wikilinks `[[L0X_B0Y]]` ao longo do texto — cada bloco aponta para os anteriores e posteriores. Se você está lendo no GitHub ou num editor de texto comum, os links aparecem como texto entre colchetes duplos. Os notebooks interativos ficam em `jax-examples/` e se abrem diretamente no Google Colab pelo botão de cada bloco prático. Uma boa forma de usar o material depois do curso é: leia o bloco teórico para rever o conceito, abra o notebook correspondente para experimentar o código, e siga os wikilinks para blocos adjacentes quando quiser aprofundar ou conectar com dias anteriores.

---

## O arco dos quatro dias

### Dia 1 — O mapa e as ferramentas (21 de julho)

O primeiro dia responde a uma pergunta antes de qualquer outra: *o que este campo realmente é, e por que ele importa para um físico em 2026?* Começamos com dois números reveladores — os 20 TB de imagens que o Observatório Rubin produzirá por noite e o Nobel de Física de outubro de 2024 concedido a Hopfield e Hinton — e estabelecemos a tese central do curso: redes neurais aprendem coordenadas. Assim como a mecânica analítica troca variáveis emaranhadas por modos normais ou centro de massa, um encoder neural encontra o sistema de coordenadas onde objetos similares ficam perto e objetos distintos ficam longe. No segundo bloco do dia, você vê essa ideia ganhar corpo em código JAX: construímos uma rede do zero, ajustamos uma senoide amortecida com ruído gaussiano via descida do gradiente explícita, e plantamos a primeira semente do Dia 2 ao observar o que acontece quando a rede é grande demais para os dados que tem.

### Dia 2 — Quando os dados mudam (22 de julho)

O segundo dia começa onde o primeiro terminou: a rede memorizou o ruído, mas o problema mais profundo não é memória — é quando os dados de teste vêm de uma distribuição completamente diferente da de treino. Isso tem um nome, *mudança de domínio*, e ocorre cronicamente na física e astrofísica toda vez que treinamos em simulações e aplicamos em observações reais. O bloco teórico constrói a taxonomia (covariate shift, prior shift, concept shift) e apresenta o modo de falha mais perigoso do ML científico: modelos que erram com 95% de confiança. Na sessão prática, você executa o ciclo completo num universo de brinquedo 2D — causa a falha, diagnostica o shift sem rótulos do alvo, e compara três estratégias de adaptação — culminando na curva que mostra quando o pré-treino vale ouro e quando não vale.

### Dia 3 — Esculpindo representações (23 de julho)

O terceiro dia faz da geometria do espaço latente o protagonista. A pergunta de abertura é um quebra-cabeça: entre um dígito "7" e o mesmo dígito deslocado 3 pixels, qual par está mais próximo em distância euclidiana bruta? A resposta contraintuitiva motiva tudo o que vem depois: aprendizagem contrastiva como um potencial de interação físico — molas que atraem pontos de mesma classe para o centroide, cargas que repelem centroides de classes diferentes. No bloco prático, você vê esse potencial em ação primeiro como dinâmica molecular pura (200 partículas se reorganizando sem nenhuma rede neural), depois como encoder treinando em dígitos MNIST, e por fim colhendo o espaço latente com k-means para fazer segmentação de instâncias sem usar rótulos na inferência — exatamente o pipeline que o Dia 4 aplicará em escala cosmológica.

### Dia 4 — Da teoria à fronteira (24 de julho)

O quarto dia aplica tudo que foi construído nos três dias anteriores a dois estudos de caso reais do departamento. No primeiro, você reconhece — não aprende do zero — a perda contrastiva do Dia 3 nas equações de um artigo publicado na *Astronomy & Astrophysics*: as "bolinhas coloridas" do sandbox são substituídas por partículas do universo primordial, e a pergunta é quais delas acabam no mesmo halo de matéria escura. No segundo estudo de caso, você vê o experimento de três regimes do Dia 2 reproduzido em escala real: um classificador treinado em espectros simulados do J-PAS degrada quando encontra o céu real, e a adaptação de encoder congela a cabeça e recalibra a percepção — exatamente como a analogia dos sotaques previu. O dia termina com o mapa do curso reaceso bloco a bloco, cada um acrescentando uma cláusula à sentença de síntese do curso.

---

## Mapa do curso — oito blocos

| Bloco | Título | Pergunta que o bloco responde | Tipo |
|-------|--------|-------------------------------|------|
| [[L01_B01]] | Aprendizado de máquina e Física: o mapa do território | O que é esse campo, por que importa para a física, e quais são as boas coordenadas? | Teoria / panorama |
| [[L01_B02]] | A caixa de ferramentas: Python, Colab e JAX na prática | Como construir e treinar uma rede do zero em JAX? | Demo guiada (NB0) |
| [[L02_B01]] | Mudança de domínio: quando o treino não é a prova | Por que modelos falham em dados de uma distribuição diferente da de treino? | Teoria |
| [[L02_B02]] | Mão na massa I: quebrar e consertar um classificador | Como diagnosticar e corrigir domain shift num experimento controlado 2D? | Laboratório (NB1) |
| [[L03_B01]] | Aprendizagem contrastiva: geometria da similaridade | Como projetar um espaço latente onde distância equivale a significado? | Teoria |
| [[L03_B02]] | Mão na massa II: esculpindo um espaço de embeddings | Como a perda contrastiva age como potencial físico e gera segmentação de instâncias? | Laboratório (NB2) |
| [[L04_B01]] | Estudo de caso I: prevendo a formação de halos com segmentação de instâncias | Como a perda do Dia 3 prevê quais partículas do universo formam cada halo? | Estudo de caso |
| [[L04_B02]] | Estudo de caso II: do mock ao céu — adaptação de domínio no J-PAS + encerramento | Como o experimento do Dia 2 opera com quasares reais em 54 filtros fotométricos? | Estudo de caso + encerramento |

---

## Notebooks interativos

Os três notebooks ficam em `jax-examples/` e executam em menos de 3 minutos no Google Colab com `PRETRAINED = True` (padrão). Cada um carrega checkpoints pré-computados e fallbacks estáticos para que a demo flua mesmo sem GPU ou com conexão instável.

---

### NB0 — A Caixa de Ferramentas

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/00_caixa_de_ferramentas.ipynb)

**Arquivo:** `jax-examples/00_caixa_de_ferramentas.ipynb` | **Bloco:** [[L01_B02]]

Constrói do zero uma rede neural completamente conexa em JAX e ajusta uma senoide amortecida via descida do gradiente explícita — sem nenhuma biblioteca de otimizador. A figura-troféu mostra o ajuste progressivo nas épocas 0, 200, 500 e 1000; a célula de sobreajuste demonstra o que acontece quando a rede é grande demais para os dados, plantando a semente conceitual do Dia 2.

---

### NB1 — Domain Shift: Quebrar e Consertar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/01_domain_shift_toy.ipynb)

**Arquivo:** `jax-examples/01_domain_shift_toy.ipynb` | **Bloco:** [[L02_B02]]

Quatro atos num universo gaussiano 2D: gera dois domínios com covariate shift visual, treina na fonte e observa a falha catastrófica no alvo (acurácia cai de 100% para 67,8%), diagnostica o shift sem rótulos, e compara zero-shot / somente alvo / SSDA numa varredura de K rótulos que mostra onde o pré-treino é valioso. O experimento exato que o estudo de caso do J-PAS no Dia 4 reproduz com quasares reais.

---

### NB2 — Embeddings Contrastivos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/02_contrastive_embeddings.ipynb)

**Arquivo:** `jax-examples/02_contrastive_embeddings.ipynb` | **Bloco:** [[L03_B02]]

Três atos de complexidade crescente: sandbox de partículas 2D relaxando sob potencial pull/push (sem rede neural), encoder MLP treinando em MNIST com a mesma perda discriminativa, e colheita do espaço latente com k-means (ARI = 0,743) e t-SNE em três perplexidades. O pipeline completo de *embed-then-cluster* que o estudo de caso de halos do Dia 4 realiza em escala cosmológica.

---

## Recursos para estudo autônomo

A lista completa de recursos curados — com orientações sobre quando e por que usar cada um — está em [[L01_B01]], na seção "Referências". Ali você encontra: a série de vídeos 3Blue1Brown para intuição visual, o curso Andrew Ng para fundamentos completos, os livros Nielsen e Prince (gratuitos online), os artigos de revisão Carleo et al. e Mehta et al. para a conexão física–ML, e links diretos para os repositórios dos dois estudos de caso do Dia 4 (`daniellopezcano/instance_halos` e `daniellopezcano/JPAS_Domain_Adaptation`).

---

## Como usar o vault do Obsidian

- **Estrutura dual dos blocos**: cada arquivo `.md` tem uma seção de estudante (em português) e um callout `[!instructor]` em inglês com o plano de tempo e notas de preparação. No Obsidian, o callout é colapsável.
- **Navegação por wikilinks**: os links `[[L0X_B0Y]]` no texto conectam conceitos entre dias. Use `Ctrl+Click` (ou `Cmd+Click` no Mac) para abrir o bloco vinculado numa nova aba.
- **Abrir notebooks no Colab**: cada bloco prático tem um botão Colab no cabeçalho. Nos blocos teóricos, o botão aparece na seção "Demonstração Prática" — para exploração independente após a aula.
- **Símbolo de cor das células dos notebooks**: 🟢 explicação de conceito (sem código), 🔵 código central (execute), 🟡 pergunta-relâmpago (preveja antes de rodar), 🟣 extensão opcional (para quem quer ir além).
- **Índice de recursos**: os recursos de autoestudo vivem em [[L01_B01]], não neste arquivo — assim permanecem próximos ao contexto pedagógico do Dia 1.

---

*Créditos: Daniel López Cano (instrutor e autor do material) — I Escola de Inverno do IFUSP, julho de 2026. Notebooks e blocos disponíveis em [`daniellopezcano/I-Escola-de-Inverno-do-IFUSP`](https://github.com/daniellopezcano/I-Escola-de-Inverno-do-IFUSP) sob licença MIT.*
