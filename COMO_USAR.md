# Como usar este repositório

Guia rápido para navegar o material da **I Escola de Inverno do IFUSP**. Nada aqui exige instalação para acompanhar o curso — só um navegador.

## O que tem aqui

```
course-materials/          8 arquivos .md (L1B1 ... L4B2) — o material teórico e os links de cada bloco, em pt-BR
jax-examples/notebooks/    3 notebooks Colab (JAX) — a parte prática
jax-examples/requirements.txt  dependências, caso prefira rodar localmente
assets/references/         os dois artigos científicos por trás da Aula 4
assets/figures_and_materials/  figuras de apoio usadas na preparação das aulas
README.md                  visão geral do curso e tabela de navegação dos blocos
```

Cada bloco (`LxBy`) tem um arquivo `course-materials/LxBy.md` correspondente: `L1B1` = Dia 1, Bloco 1; `L1B2` = Dia 1, Bloco 2; e assim por diante. Blocos `B1` são teóricos (slides); blocos `B2` são práticos (notebook).

## Lendo o material teórico

Você tem duas opções, da mais simples à mais completa.

**Direto no GitHub (mais simples).** Abra qualquer arquivo em `course-materials/` na aba "Code" do GitHub — ele renderiza como Markdown normal. Único detalhe: os links internos entre blocos aparecem como `[[L2B1]]` (sintaxe do Obsidian) em vez de links clicáveis — o GitHub não os interpreta. Você ainda sabe para qual bloco olhar, só precisa navegar até ele manualmente.

**Como vault do Obsidian (experiência completa).** Se você tem o [Obsidian](https://obsidian.md/) instalado: `Abrir pasta como vault` → selecione a pasta `course-materials/`. Os `[[wikilinks]]` viram links clicáveis de verdade, o painel de backlinks mostra quais blocos referenciam qual, e a visão de grafo mostra o curso inteiro conectado. É a forma mais agradável de navegar entre um bloco e o próximo.

## Rodando os notebooks (Google Colab)

Cada bloco prático (`L1B2`, `L2B2`, `L3B2`) tem um botão **Open in Colab** — no `course-materials/LxBy.md` do bloco e no README. Clique nele: o notebook abre direto no seu navegador, com GPU disponível gratuitamente, sem nenhuma instalação local.

**No primeiro `Shift+Enter`** (ou `Ambiente de execução → Executar tudo`), a primeira célula instala as bibliotecas que faltam no Colab (JAX, Equinox, Optax, etc.) — isso leva alguns segundos e só acontece uma vez por sessão. Todos os dados usados no notebook são gerados ou baixados nessa mesma execução; nada precisa ser baixado à parte.

**Para guardar o seu progresso:** o Colab não salva sozinho no seu Drive. Vá em `Arquivo → Salvar uma cópia no Drive` assim que abrir o notebook — daí em diante você edita a sua própria cópia, e o original do repositório fica intacto.

## Rodando localmente (opcional)

Se preferir rodar fora do Colab:

```bash
python -m venv .venv && source .venv/bin/activate   # ou seu gerenciador de ambientes preferido
pip install -r jax-examples/requirements.txt
jupyter lab jax-examples/notebooks/
```

Os notebooks não dependem de GPU — rodam em CPU, só mais devagar nas células de treino.

## Os dois artigos de referência (Aula 4)

Os dois estudos de caso do Dia 4 (`L4B2`) vêm de trabalhos publicados do instrutor; os PDFs estão em `assets/references/` para leitura offline, e os mesmos artigos estão linkados no corpo de `L4B2.md`:

- **Segmentação de instâncias de halos** — López-Cano et al. (2024), *A&A* — [arXiv:2311.12110](https://arxiv.org/abs/2311.12110)
- **J-PAS: adaptação de domínio sim-to-obs** — López-Cano et al. (2026) — [arXiv:2602.13902](https://arxiv.org/abs/2602.13902)
