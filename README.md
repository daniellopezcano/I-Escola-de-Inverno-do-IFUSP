# Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos

**I Escola de Inverno do IFUSP** · 21–24 de julho de 2026 · Instituto de Física da USP  
Minicurso de 4 aulas · Instrutor: Dr. Daniel López Cano

[website link](https://portal.if.usp.br/pesquisa/pt-br/node/2745)

---

## Arco do curso

| Dia | Data | Objetivo |
|-----|------|----------|
| 1 | 21 jul | Mapa do território: o que ML faz (e não faz) em astrofísica; o espaço latente como fio condutor; os dois tópicos do curso como respostas a limitações reais. |
| 2 | 22 jul | Mudança de domínio: diagnose, falha silenciosa e três regimes de adaptação num universo gaussiano 2D. |
| 3 | 23 jul | Aprendizagem contrastiva: esculpir um espaço latente com potenciais pull/push e colher clusters sem rótulos. |
| 4 | 24 jul | Estudos de caso reais: reconhecer os mesmos métodos em segmentação de halos e na adaptação mock→céu no J-PAS. |

---

## Blocos

| Bloco    | Tipo     | Slides / Colab                                                                                                                                                                                                                       |
| -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [[L1B1]] | teoria   | [Slides](https://docs.google.com/presentation/d/1urJoVZ1Oeko21DEa6jq737MJcpetG1whUMFMDD05oq0/edit?usp=drive_link)                                                                                                                    |
| [[L1B2]] | notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/00_caixa_de_ferramentas.ipynb)   |
| [[L2B1]] | teoria   | [Slides](https://docs.google.com/presentation/d/1pIMOeHfmTVYm2h_TUT8vcqtHDXz3jW1oxVN8rdWgm9s/edit?usp=drive_link)                                                                                                                    |
| [[L2B2]] | notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/01_domain_shift_toy.ipynb)       |
| [[L3B1]] | teoria   | [Slides](??)                                                                                                                                                                                                                         |
| [[L3B2]] | notebook | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/02_contrastive_embeddings.ipynb) |
| [[L4B1]] | teoria   | [Slides](??)                                                                                                                                                                                                                         |
| [[L4B2]] | teoria   | [Slides](??)                                                                                                                                                                                                                         |

---

## Como usar estes materiais

Cada dia tem dois blocos de 40 min. O material se organiza em três camadas:

1. **Markdown teórico** (`course-materials/LxBy.md`, blocos L1B1, L2B1, L3B1, L4B1, L4B2) — narrativa compacta que explica os conceitos do bloco. É o **artefato primário**: o instrutor constrói os slides Google *a partir* deste texto. Para revisão, leia o markdown e siga as referências inline; use o link de slides para a versão visual.
2. **Markdown de notebook** (`course-materials/LxBy.md`, blocos L1B2, L2B2, L3B2): link do Colab e uma linha de contexto.
3. **Notebook JAX** (`jax-examples/notebooks/`) — código executável no Colab. Os notebooks geram todos os dados e cálculos no próprio runtime. Acesse pelo botão na tabela acima.

No Obsidian, use os wikilinks `[[L1B1]]`, `[[L2B1]]` etc. para navegar entre blocos (`Ctrl+Click` abre numa nova aba). Os recursos de auto-estudo vivem em [[L1B1]].

---

## O arco dos quatro dias

### Dia 1 — O mapa e as ferramentas (21 de julho)
O primeiro dia responde a uma pergunta antes de qualquer outra: *o que este campo realmente é, e por que ele importa para um físico em 2026?* Começamos com dois números reveladores — os 20 TB de imagens que o Observatório Rubin produzirá por noite e o Nobel de Física de outubro de 2024 concedido a Hopfield e Hinton — e estabelecemos a tese central do curso: redes neurais aprendem coordenadas. Assim como a mecânica analítica troca variáveis emaranhadas por modos normais ou centro de massa, um encoder neural encontra o sistema de coordenadas onde objetos similares ficam perto e objetos distintos ficam longe. No segundo bloco do dia, você vê essa ideia ganhar corpo em código JAX: construímos uma rede completamente conexa do zero, ajustamos uma senoide amortecida ruidosa via descida do gradiente explícita — sem nenhuma biblioteca de otimizador — e plantamos a primeira semente do Dia 2 ao observar o que acontece quando a rede é grande demais para os dados que tem.

### Dia 2 — Quando os dados mudam (22 de julho)
O segundo dia começa onde o primeiro terminou: a rede memorizou o ruído, mas o problema mais profundo não é memória — é quando os dados de teste vêm de uma distribuição completamente diferente da de treino. Isso tem um nome, *mudança de domínio*, e ocorre cronicamente na física e astrofísica toda vez que treinamos em simulações e aplicamos em observações reais. O bloco teórico constrói a taxonomia (covariate shift, prior shift, concept shift) e apresenta o modo de falha mais perigoso do ML científico: modelos que erram com 95% de confiança. Na sessão prática, você executa o ciclo completo num universo de brinquedo 2D — causa a falha, diagnostica o shift sem usar nenhum rótulo do domínio alvo (AUC ≈ 0,749 de um classificador de domínio binário), e compara três estratégias de adaptação culminando na curva que mostra quando o pré-treino vale ouro e quando não vale.

### Dia 3 — Esculpindo representações (23 de julho)
O terceiro dia faz da geometria do espaço latente o protagonista. A pergunta de abertura é um quebra-cabeça: entre um dígito "7" e o mesmo dígito deslocado 3 pixels, qual par está mais próximo em distância euclidiana bruta? A resposta contraintuitiva motiva tudo o que vem depois: aprendizagem contrastiva formulada como um potencial de interação físico — molas ($\mathcal{L}_\text{pull}$) atraem pontos de mesma classe para o centroide, cargas ($\mathcal{L}_\text{push}$) repelem centroides de classes diferentes, e um regularizador ancora o sistema. No bloco prático, você vê esse potencial em ação primeiro como dinâmica molecular pura (200 partículas se reorganizando sem nenhuma rede neural), depois como encoder treinando em dígitos MNIST, e por fim colhendo o espaço latente com k-means (ARI = 0,743) para fazer segmentação de instâncias sem usar rótulos na inferência — exatamente o pipeline que o Dia 4 aplicará em escala cosmológica.

### Dia 4 — Da teoria à fronteira (24 de julho)
O quarto dia aplica tudo que foi construído nos três dias anteriores a dois estudos de caso reais. No primeiro, você reconhece — não aprende do zero — a perda contrastiva do Dia 3 nas equações de um artigo publicado na *Astronomy & Astrophysics*: as "bolinhas coloridas" do sandbox são substituídas por partículas do universo primordial, e a pergunta é quais delas acabam no mesmo halo de matéria escura. No segundo estudo de caso, você vê o experimento de três regimes do Dia 2 reproduzido em escala real com quasares no J-PAS: um classificador treinado em espectros simulados degrada quando encontra o céu real, e a adaptação de encoder congela a cabeça e recalibra a percepção — exatamente como a analogia dos sotaques previu. O dia termina com o mapa do curso reaceso bloco a bloco, cada um acrescentando uma cláusula à sentença de síntese: *«Representações são coordenadas; coordenadas se esculpem; esculturas quebram quando o mundo muda; e adaptá-las é barato — se você souber o que congelar.»*

---

## Notebooks interativos

Os três notebooks ficam em `jax-examples/notebooks/` e executam no Google Colab. Cada um gera dados e cálculos no próprio runtime, sem artefatos comprometidos no repositório.

---

### NB0 — A Caixa de Ferramentas
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/00_caixa_de_ferramentas.ipynb)

`jax-examples/notebooks/00_caixa_de_ferramentas.ipynb` · Bloco [[L1B2]]

Constrói do zero uma rede neural completamente conexa em JAX — parâmetros como lista de `(W, b)`, ativação `tanh`, perda MSE — e ajusta uma senoide amortecida ruidosa via descida do gradiente explícita sem nenhuma biblioteca de otimizador. A célula de sobreajuste demonstra o que acontece quando a rede `[1→128→128→128→1]` treina por 5 000 épocas, plantando a semente conceitual do Dia 2. O mapa de vocabulário da última célula conecta o código ao jargão padrão.

---

### NB1 — Domain Shift: Quebrar e Consertar
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/01_domain_shift_toy.ipynb)

`jax-examples/notebooks/01_domain_shift_toy.ipynb` · Bloco [[L2B2]]

Quatro atos num universo gaussiano 2D: gera dois domínios com covariate shift visual (4 classes desbalanceadas), treina encoder + cabeça na fonte e observa a falha catastrófica no alvo, diagnostica o shift sem nenhum rótulo de classe do alvo usando um classificador binário (AUC ≈ 0,749), e compara zero-shot / somente alvo / SSDA numa varredura de K rótulos. O experimento exato — mesma lógica, mesmos três regimes — que o estudo de caso do J-PAS no Dia 4 reproduz com quasares reais e 55 bandas fotométricas.

---

### NB2 — Embeddings Contrastivos
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniellopezcano/I-Escola-de-Inverno-do-IFUSP/blob/main/jax-examples/notebooks/02_contrastive_embeddings.ipynb)

`jax-examples/notebooks/02_contrastive_embeddings.ipynb` · Bloco [[L3B2]]

Três atos de complexidade crescente: sandbox de partículas 2D relaxando sob potencial pull/push — com demonstração do colapso trivial quando a repulsão é removida; encoder MLP `[784→256→64→2]` treinando em MNIST com a mesma perda (dígitos 4/9 e 3/8 ficam adjacentes por "degenerescência física da escrita"); e colheita do espaço latente com k-means (ARI = 0,743) e t-SNE em três perplexidades.

---

## Referências e recursos

- [references/2602.13902v1.pdf](references/2602.13902v1.pdf) — López-Cano et al., *arXiv:2602.13902* — J-PAS domain adaptation (âncora de L4B2)
- [references/2311.12110v3.pdf](references/2311.12110v3.pdf) — López-Cano et al., *A&A 685 A37* — Instance segmentation / halo formation (âncora de L4B1)
- [GoogleCollab_and_notebooks_setup.md](GoogleCollab_and_notebooks_setup.md) — instruções de ambiente e manutenção dos notebooks
