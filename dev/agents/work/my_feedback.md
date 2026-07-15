# comments

Note that now the structure of this repo (I changed a few things such as making `.dev` visible , now `dev`)
```
(base) ➜  I-Escola-de-Inverno-do-IFUSP git:(main) ✗ tree -a -L 6   
.
├── assets
│   ├── figures
│   └── logos
├── .claude
│   ├── agents
│   │   ├── block-writer.md
│   │   ├── course-architect.md
│   │   ├── course-reviewer.md
│   │   └── notebook-builder.md
│   └── settings.json
├── CLAUDE.md
├── Course Development Multi-Agent Pipeline — Complete Setup Guide.md
├── course-materials
│   ├── 00_INDEX.md
│   ├── 01_Lectures
│   ├── 02_Hands_On
│   ├── 03_References
│   ├── L01_B01.md
│   ├── L01_B02.md
│   ├── L02_B01.md
│   ├── L02_B02.md
│   ├── L03_B01.md
│   ├── L03_B02.md
│   ├── L04_B01.md
│   ├── L04_B02.md
│   ├── Master Plan — «Das representações de redes neurais às aplicações em Física, Astrofísica e dados de levantamentos astronômicos».md
│   └── Templates
│       └── Block_Template.md
├── .dev
│   └── drafts
├── dev
│   └── agents
│       └── work
│           ├── build_logs
│           │   ├── 00_caixa_de_ferramentas.log
│           │   ├── 01_domain_shift_toy.log
│           │   └── 02_contrastive_embeddings.log
│           ├── coherence_report.md
│           ├── course_manifest.md
│           └── my_feedback.md
├── .gitignore
├── jax-examples
│   ├── 00_caixa_de_ferramentas.ipynb
│   ├── 01_domain_shift_toy.ipynb
│   ├── 02_contrastive_embeddings.ipynb
│   ├── assets
│   │   ├── mnist_4k.npz
│   │   ├── _mnist_raw
│   │   │   ├── t10k-images-idx3-ubyte.gz
│   │   │   ├── t10k-labels-idx1-ubyte.gz
│   │   │   ├── train-images-idx3-ubyte.gz
│   │   │   └── train-labels-idx1-ubyte.gz
│   │   ├── nb0_epoch0_params.pkl
│   │   ├── nb0_epoch200_params.pkl
│   │   ├── nb0_epoch500_params.pkl
│   │   ├── nb0_fcnn_params.pkl
│   │   ├── nb0_fig_overfit.png
│   │   ├── nb0_fig_senoide_intro.png
│   │   ├── nb0_fig_trophy.png
│   │   ├── nb0_overfit_params.pkl
│   │   ├── nb1_encoder_source.pkl
│   │   ├── nb1_encoder_ssda.pkl
│   │   ├── nb1_encoder_targetonly.pkl
│   │   ├── nb1_fig_comparison.png
│   │   ├── nb1_fig_decision_source.png
│   │   ├── nb1_fig_decision_target.png
│   │   ├── nb1_fig_gmm.png
│   │   ├── nb1_fig_k_sweep.png
│   │   ├── nb1_fig_latent_shift.png
│   │   ├── nb1_fig_latent_ssda.png
│   │   ├── nb1_head_source.pkl
│   │   ├── nb1_head_ssda.pkl
│   │   ├── nb1_head_targetonly.pkl
│   │   ├── nb1_ksweep.npz
│   │   ├── nb2_encoder_16d_late.pkl
│   │   ├── nb2_encoder_early.pkl
│   │   ├── nb2_encoder_epoch0.pkl
│   │   ├── nb2_encoder_late.pkl
│   │   ├── nb2_fig_evolution.png
│   │   ├── nb2_fig_sandbox_final.png
│   │   ├── nb2_fig_tsne.png
│   │   ├── nb2_fig_umap.png
│   │   ├── nb2_sandbox_collapsed.npz
│   │   ├── nb2_sandbox_final.npz
│   │   ├── nb2_sandbox_initial.npz
│   │   └── toy_2d_4class.npz
│   ├── notebooks
│   ├── README.md
│   ├── requirements.txt
│   ├── src
│   ├── src_00_caixa_de_ferramentas.py
│   ├── src_01_domain_shift_toy.py
│   ├── src_02_contrastive_embeddings.py
│   └── utils
│       ├── make_assets_00_caixa_de_ferramentas.py
│       ├── make_assets_01_domain_shift_toy.py
│       └── make_assets_02_contrastive_embeddings.py
├── LICENSE
├── .obsidian
│   ├── appearance.json
│   ├── app.json
│   ├── core-plugins.json
│   ├── graph.json
│   └── workspace.json
├── README.md
└── references
    ├── 2602.13902v1.pdf
    └── aa4896523.pdf
```

Improve the README, the objective of that file is to first present the course and it's materials but to also serve as the core structural backbone that links all the different files so we have this conceptual structure (centralized even though some of the contents within the specific markdown files might relate to each other with obsidian links for examples references between hands on sessions to some content notes etc)

Take into account that for each lecture I will also prepare some google slides that will help me to explain, in here I'll incorporate key text and equations, relevant/visual figures to support my explanations and narrative etc. The markdown files within the course-materials should be structured in such a way that they reflect the contents of those slides, serving as a parallel (1-to-1 correspondence with the slides) more narrated version explaining a bit more in-depth (still briefly but clear and compact) the key concepts (include equations where necessary). So please elaborate the associated markdown files for each lecture block in such a way that they serve me to produce the related google slides (focus on the concepts that they should cover, I'll personally then structure the contents, images etc to be included in the slides). Here are the links to the slides (I still need to prepare the contents but these links should be included in the different .md files):
L1B1: https://docs.google.com/presentation/d/1urJoVZ1Oeko21DEa6jq737MJcpetG1whUMFMDD05oq0/edit?usp=drive_link
L1B2: https://docs.google.com/presentation/d/1WDPyB7RwiyfdQaY3YQUktrd7_G8ZmtJbtO7tJT13qO4/edit?usp=drive_link
L2B1: https://docs.google.com/presentation/d/1pIMOeHfmTVYm2h_TUT8vcqtHDXz3jW1oxVN8rdWgm9s/edit?usp=drive_link
L2B2: https://docs.google.com/presentation/d/1ketbGyOy96r_Mm7WF6oP8PDxeZBBErbfyWCudwvNuu4/edit?usp=drive_link
L3B1: https://docs.google.com/presentation/d/17ssxMhezRtTREFM1FZc32VMsYP1cQ5eFazUUM1QdQQs/edit?usp=drive_link
L3B2: https://docs.google.com/presentation/d/1UI1RycsVcagsoXPOS5581GF-Mu0Ooi13uGcr41kZ0sk/edit?usp=drive_link
L4B1: https://docs.google.com/presentation/d/1ZVmImbVYYQAWHdR6NNlSLlCw8jtiLWMYDwlg4315dhk/edit?usp=drive_link
L4B2: https://docs.google.com/presentation/d/1E4n9hgIszUmmZiGFGFF2BJMCBhqDiU1iSYfgl3rX6HE/edit?usp=drive_link


I would like you to now work on this second pass trying not to elaborate in great depth (as it seems you tried to do in the first pass) the concepts that will be treated in each lecture, but to provide a more schematic view of the different parts, topics, and concepts that the slides should contain so I can start preparing those. We can carefully go back to each of the blocks for explaining in more depth the concepts introduced at the slide level so the students can use the .md files as material for reference.

Focus more on the structure and topics of the "theory" blocks (regarding the markdown files) since the hands-on ones (L1B2, L2B2, and L3B2) should be more self-explanatory from the notebooks we are generating (besides, the concepts that are illustrated in here would reflect the theory part already introduced) and the markdown files for those should contain a much less extensive descriptions. Also, L4B1 and L4B2 are going to be easy for me to adapt later, so for now, focus on the markdowns for: L1B1, L2B1, and L3B1, while focusing on polishing the notebooks for L1B2, L2B2, and L3B2 (the accompanying .md for these would be supplementary)

Regarding the notebooks you have produced, please, store them directly at "jax-examples/notebooks" and include a new GoogleCollab_and_notebooks_setup.md file explaining how to then "upload" or convert those notebooks to the google collab platform so I can share them easier afterwards with the students (even though the debugging and runing you first do locally).



 
