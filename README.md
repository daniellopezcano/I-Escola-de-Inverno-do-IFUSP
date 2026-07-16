# 🌌 Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos

**I Escola de Inverno do IFUSP** · 21–24 de julho de 2026 · Instituto de Física da USP
Minicurso de 4 aulas. **Instrutor:** Dr. Daniel López Cano

---
## Estrutura conceitual: slides ⇄ Markdown ⇄ notebook

O material de cada dia foi desenhado em três camadas que se espelham:

1. **Slides do Google** (link em cada bloco teórico) — o que o instrutor mostra e narra ao vivo.
2. **Bloco Markdown** em `course-materials/LXBY.md`:
   **Blocos teóricos** (`L01_B01`, `L02_B01`, `L03_B01`, `L04_B01`, `L04_B02`) usam o modo **slide-schematic**: um roteiro compacto, slide a slide, listando os conceitos/equações/figuras que cada slide dos Google Slides deve cobrir — não é prosa narrada, é o rascunho estrutural que o instrutor usa para montar os slides de verdade.
3. **Notebook JAX** em `jax-examples/notebooks/` — o código executável que implementa ao vivo o que os slides e o Markdown anunciam.

---
## O arco do curso

Cada dia tem 2 blocos de 40 minutos: um bloco **teórico** (slides + Markdown em modo *slide-schematic*) e um bloco **mão-na-massa** (notebook JAX). Detalhes completos, com a narrativa dia-a-dia e os objetivos de aprendizagem, estão no hub — **[course-materials/00_INDEX.md](course-materials/00_INDEX.md)**.

| Bloco  (Markdown) | Título                                                                       | Slides                                                                                                            | Notebook                                                                            |
| ----------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [[L1B1]]          | Aprendizado de máquina e Física: o mapa do território                        | [Slides](https://docs.google.com/presentation/d/1urJoVZ1Oeko21DEa6jq737MJcpetG1whUMFMDD05oq0/edit?usp=drive_link) | —                                                                                   |
| [[L1B2]]          | A caixa de ferramentas: Python, Colab e JAX na prática                       | —                                                                                                                 | [00_caixa_de_ferramentas](jax-examples/notebooks/00_caixa_de_ferramentas.ipynb)     |
| [[L2B1]]          | Mudança de domínio: quando o treino não é a prova                            | [Slides](https://docs.google.com/presentation/d/1pIMOeHfmTVYm2h_TUT8vcqtHDXz3jW1oxVN8rdWgm9s/edit?usp=drive_link) | —                                                                                   |
| [[L2B2]]          | Mão na massa I: quebrar e consertar um classificador                         | —                                                                                                                 | [01_domain_shift_toy](jax-examples/notebooks/01_domain_shift_toy.ipynb)             |
| [[L3B1]]          | Aprendizagem contrastiva: geometria da similaridade                          | [Slides](https://docs.google.com/presentation/d/17ssxMhezRtTREFM1FZc32VMsYP1cQ5eFazUUM1QdQQs/edit?usp=drive_link) | —                                                                                   |
| [[L3B2]]          | Mão na massa II: esculpindo um espaço de embeddings                          | —                                                                                                                 | [02_contrastive_embeddings](jax-examples/notebooks/02_contrastive_embeddings.ipynb) |
| [[L4B1]]          | Estudo de caso I: prevendo a formação de halos com segmentação de instâncias | [Slides](https://docs.google.com/presentation/d/1ZVmImbVYYQAWHdR6NNlSLlCw8jtiLWMYDwlg4315dhk/edit?usp=drive_link) | —                                                                                   |
| [[L4B2]]          | Estudo de caso II: do mock ao céu — adaptação de domínio no J-PAS            | [Slides](https://docs.google.com/presentation/d/1E4n9hgIszUmmZiGFGFF2BJMCBhqDiU1iSYfgl3rX6HE/edit?usp=drive_link) | —                                                                                   |

---
## Como usar este material
Este repositório é o mapa e o kit de ferramentas do curso. Cada um dos oito arquivos de bloco (`L0X_B0Y.md`) é um guia: explica o conceito, apresenta a matemática e descreve o que será executado ao vivo na aula. Os três notebooks interativos ficam em `jax-examples/notebooks/` e se abrem diretamente no Google Colab pelo botão de cada bloco prático. Uma boa forma de usar o material após o curso: leia o bloco teórico para rever o conceito, abra o notebook correspondente para experimentar o código, e siga os wikilinks quando quiser aprofundar ou conectar com outros dias.

---
## O arco dos quatro dias

### Dia 1 — O mapa e as ferramentas (21 de julho)
O primeiro dia responde a uma pergunta antes de qualquer outra: *o que este campo realmente é, e por que ele importa para um físico em 2026?* Começamos com dois números reveladores — os 20 TB de imagens que o Observatório Rubin produzirá por noite e o Nobel de Física de outubro de 2024 concedido a Hopfield e Hinton — e estabelecemos a tese central do curso: redes neurais aprendem coordenadas. Assim como a mecânica analítica troca variáveis emaranhadas por modos normais ou centro de massa, um encoder neural encontra o sistema de coordenadas onde objetos similares ficam perto e objetos distintos ficam longe. No segundo bloco do dia, você vê essa ideia ganhar corpo em código JAX: construímos uma rede completamente conexa do zero, ajustamos uma senoide amortecida ruidosa via descida do gradiente explícita — sem nenhuma biblioteca de otimizador — e plantamos a primeira semente do Dia 2 ao observar o que acontece quando a rede é grande demais para os dados que tem.

### Dia 2 — Quando os dados mudam (22 de julho)
O segundo dia começa onde o primeiro terminou: a rede memorizou o ruído, mas o problema mais profundo não é memória — é quando os dados de teste vêm de uma distribuição completamente diferente da de treino. Isso tem um nome, *mudança de domínio*, e ocorre cronicamente na física e astrofísica toda vez que treinamos em simulações e aplicamos em observações reais. O bloco teórico constrói a taxonomia (covariate shift, prior shift, concept shift) e apresenta o modo de falha mais perigoso do ML científico: modelos que erram com 95% de confiança. Na sessão prática, você executa o ciclo completo num universo de brinquedo 2D — causa a falha, diagnostica o shift sem usar nenhum rótulo do domínio alvo (AUC ≈ 0,785 de um classificador de domínio binário), e compara três estratégias de adaptação culminando na curva que mostra quando o pré-treino vale ouro e quando não vale.

### Dia 3 — Esculpindo representações (23 de julho)
O terceiro dia faz da geometria do espaço latente o protagonista. A pergunta de abertura é um quebra-cabeça: entre um dígito "7" e o mesmo dígito deslocado 3 pixels, qual par está mais próximo em distância euclidiana bruta? A resposta contraintuitiva motiva tudo o que vem depois: aprendizagem contrastiva formulada como um potencial de interação físico — molas ($\mathcal{L}_\text{pull}$) atraem pontos de mesma classe para o centroide, cargas ($\mathcal{L}_\text{push}$) repelem centroides de classes diferentes, e um regularizador ancora o sistema. No bloco prático, você vê esse potencial em ação primeiro como dinâmica molecular pura (200 partículas se reorganizando sem nenhuma rede neural), depois como encoder treinando em dígitos MNIST, e por fim colhendo o espaço latente com k-means (ARI = 0,743) para fazer segmentação de instâncias sem usar rótulos na inferência — exatamente o pipeline que o Dia 4 aplicará em escala cosmológica.

### Dia 4 — Da teoria à fronteira (24 de julho)
O quarto dia aplica tudo que foi construído nos três dias anteriores a dois estudos de caso reais. No primeiro, você reconhece — não aprende do zero — a perda contrastiva do Dia 3 nas equações de um artigo publicado na *Astronomy & Astrophysics*: as "bolinhas coloridas" do sandbox são substituídas por partículas do universo primordial, e a pergunta é quais delas acabam no mesmo halo de matéria escura. No segundo estudo de caso, você vê o experimento de três regimes do Dia 2 reproduzido em escala real com quasares no J-PAS: um classificador treinado em espectros simulados degrada quando encontra o céu real, e a adaptação de encoder congela a cabeça e recalibra a percepção — exatamente como a analogia dos sotaques previu. O dia termina com o mapa do curso reaceso bloco a bloco, cada um acrescentando uma cláusula à sentença de síntese: *«Representações são coordenadas; coordenadas se esculpem; esculturas quebram quando o mundo muda; e adaptá-las é barato — se você souber o que congelar.»*

---
## Notebooks interativos
Os três notebooks ficam em `jax-examples/notebooks/` e executam em menos de 3 minutos no Google Colab com `PRETRAINED = True` (padrão). Cada um carrega checkpoints pré-computados em `jax-examples/assets/` e exibe fallbacks estáticos em PNG caso a renderização ao vivo falhe, para que a demo flua mesmo sem GPU ou com conexão instável.

---
### NB0 — A Caixa de Ferramentas
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/00_caixa_de_ferramentas.ipynb)

**Arquivo:** `jax-examples/notebooks/00_caixa_de_ferramentas.ipynb` | **Bloco:** [[L1B2]]

Constrói do zero uma rede neural completamente conexa em JAX — parâmetros como lista de `(W, b)`, ativação `tanh`, perda MSE — e ajusta uma senoide amortecida ruidosa via descida do gradiente explícita sem nenhuma biblioteca de otimizador. A figura-troféu mostra o ajuste progressivo nas épocas 0, 200, 500 e 1000; a célula de sobreajuste demonstra o que acontece quando a rede `[1→128→128→128→1]` treina por 5 000 épocas, plantando a semente conceitual do Dia 2. O mapa de vocabulário da última célula conecta o código ao jargão padrão.

---
### NB1 — Domain Shift: Quebrar e Consertar
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/01_domain_shift_toy.ipynb)

**Arquivo:** `jax-examples/notebooks/01_domain_shift_toy.ipynb` | **Bloco:** [[L2B2]]

Quatro atos num universo gaussiano 2D: gera dois domínios com covariate shift visual (4 classes desbalanceadas, duas nuvens deslocadas ~2,5σ), treina encoder + cabeça na fonte e observa a falha catastrófica no alvo (acurácia cai de ~100% para 67,8%), diagnostica o shift sem nenhum rótulo de classe do alvo usando um classificador binário (AUC ≈ 0,785), e compara zero-shot / somente alvo / SSDA numa varredura de K rótulos. O experimento exato — mesma lógica, mesmos três regimes — que o estudo de caso do J-PAS no Dia 4 reproduz com quasares reais e 56 bandas fotométricas.

---
### NB2 — Embeddings Contrastivos
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/02_contrastive_embeddings.ipynb)

**Arquivo:** `jax-examples/notebooks/02_contrastive_embeddings.ipynb` | **Bloco:** [[L3B2]]

Três atos de complexidade crescente: sandbox de partículas 2D relaxando sob potencial pull/push — com a demonstração do colapso trivial quando a repulsão é removida — sem nenhuma rede neural; encoder MLP `[784→256→64→2]` treinando em MNIST com a perda idêntica (dígitos 4/9 e 3/8 ficam adjacentes por "degenerescência física da escrita"); e colheita do espaço latente com k-means (ARI = 0,743) e t-SNE em três perplexidades. O pipeline completo de *embed-then-cluster* que o estudo de caso de halos do Dia 4 realiza em escala cosmológica com partículas do universo primordial.

---
## Como usar o vault do Obsidian
- **Estrutura dual dos blocos**: cada arquivo `.md` tem uma seção de estudante em português e um callout `[!instructor]` em inglês com o plano de tempo e notas de preparação. No Obsidian, o callout é colapsável.
- **Navegação por wikilinks**: os links `[[L0X_B0Y]]` no texto conectam conceitos entre dias. Use `Ctrl+Click` (ou `Cmd+Click` no Mac) para abrir o bloco vinculado numa nova aba.
- **Abrir notebooks no Colab**: cada bloco prático tem um botão Colab no cabeçalho. Nos blocos teóricos do Dia 4, o botão aparece na seção "Demonstração Prática" e leva ao notebook da sessão anterior correspondente — para exploração independente após a aula.
- **Convenção de cores das células dos notebooks**: 🟢 explicação de conceito (sem código obrigatório), 🔵 código central (execute), 🟡 pergunta-relâmpago (preveja antes de rodar a próxima célula), 🟣 extensão opcional (para quem quer ir além — o instrutor anuncia "não precisam fazer agora").
- **Índice de recursos**: os recursos de autoestudo vivem em [[L1B1]], não neste arquivo — assim permanecem próximos ao contexto pedagógico do Dia 1.