# Enhancement brief — L2B2 notebook (01_domain_shift_toy)

## Purpose of this notebook
A SIMPLE TOY MODEL that teaches, step by step: (1) how to train a classifier in JAX with
Equinox/Optax, (2) how to EVALUATE it properly with a rich set of metrics and
visualizations, (3) what DOMAIN SHIFT looks like and how it degrades those metrics, and
(4) how SUPERVISED TRANSFER (fine-tuning on target labels) repairs it.

This is a near-total reshape of the existing notebook. Keep its 2D-toy-data spirit
(everything visualizable), but restructure into the seven blocks below.

## Scope decisions (IMPORTANT)
- DE-EMPHASIZE latent space / embeddings. They are secondary here and belong to L3.
  Do not build the notebook around an encoder/head latent story. If a latent view
  appears at all, it is a brief aside, not a section.
- The star of this notebook is TRAINING + EVALUATION TOOLING, then domain shift, then
  supervised transfer.
- Match the pedagogical philosophy and style of the already-polished L1B2 notebook
  (00_caixa_de_ferramentas): Equinox + Optax, helpers defined INLINE where first needed
  (no big utility block up front), short cells, one idea per cell, plots interleaved,
  pt-BR narrative, compact explanations.

## Reference material (adapt and SIMPLIFY; do not copy wholesale)
The instructor's research notebooks show the target style of the evaluation tooling:
- https://github.com/daniellopezcano/JPAS_Domain_Adaptation/blob/main/notebooks/06_training_tools.ipynb
- https://github.com/daniellopezcano/JPAS_Domain_Adaptation/blob/main/notebooks/08_evaluation_tools.ipynb
- https://github.com/daniellopezcano/JPAS_Domain_Adaptation/blob/main/JPAS_DA/evaluation/evaluation_tools.py
Fetch them if network access is available; otherwise implement from this brief.
Everything must live SELF-CONTAINED in the notebook (no external repo imports),
simplified to teaching scale — same spirit as L1B2.

## Structure — seven blocks

### Bloco 1 — Gerador de dados 2D (fonte)
- A toy 2D point-distribution generator producing several labeled classes.
- Non-trivial but not baroque: class regions should have curved/non-linear boundaries so
  a trained classifier learns interesting contours. Classes should PARTIALLY OVERLAP /
  blend, so evaluation later is interesting (errors are real, not zero).
- Classes must be IMBALANCED (state the proportions).
- Produce training and validation sets from this source distribution.
- Plot: scatter of the source data, colored by class, showing the overlap and imbalance.

### Bloco 2 — Treinar um classificador (Equinox + Optax)
- Small MLP classifier defined with Equinox; Optax optimizer; mini-batch training loop.
- Cross-entropy loss; mention/handle class imbalance explicitly (e.g. class weights) with
  a one-line justification.
- Mirror L1B2's style so students recognize the machinery they already learned.
- Plot: training/validation loss curves.

### Bloco 3 — Avaliação: visualizações e estatísticas de resumo
The evaluation toolbox — the heart of the notebook.
(a) PROBABILITY MAPS: evaluate the model on a dense grid over the 2D feature space and
    show background probability maps (use a seismic/diverging colormap) with the sampled
    points overlaid, so students SEE what the model believes everywhere.
(b) DECISION REGIONS: apply a probability cut/threshold and show which regions get
    assigned to which class (hard decision boundaries), overlaid with the data.
(c) SUMMARY STATISTICS: confusion matrix, plus per-class TPR (recall), PPV (precision)
    and F1 — with the imbalance making the difference between accuracy and per-class
    metrics obvious. Explain briefly in pt-BR what each captures.
(d) ROC curves and AUC, per class (one-vs-rest), with a SHORT clear explanation of what
    a ROC curve is, what AUC means, and what it captures that accuracy doesn't.
CORRECTNESS REQUIREMENT: compute ROC/AUC properly — guard against degenerate single-class
slices (a previous version produced NaN AUC from a single-class y_true). Verify numerically.

### Bloco 4 — Introduzindo o domain shift
- Take the source generator and SHIFT its properties slightly (e.g. translate/rotate/
  rescale class clouds, change spread) to create a TARGET distribution.
- Sample target train/validation sets.
- Closing figure of this block: source vs. target distributions plotted together (or
  side by side) so the shift is visually unmistakable.

### Bloco 5 — Quanto o modelo se degrada no alvo?
- Apply the SOURCE-trained model (unchanged) to the TARGET data.
- Re-run the full Bloco-3 evaluation suite on the target: probability maps, decision
  regions, confusion matrix, TPR/PPV/F1, ROC/AUC.
- Quantify the degradation by comparing target performance against the source-validation
  performance (a compact side-by-side table and/or paired plots).
- pt-BR narrative: the model is confidently wrong in the shifted regions.

### Bloco 6 — Transfer learning supervisionado (fine-tuning no alvo)
- Load/reuse the previously trained source model and RETRAIN (fine-tune) it using a
  labeled training sample from the TARGET distribution.
- This exemplifies the SUPERVISED TRANSFER case presented in L2B1 — a simple but real
  remedy for domain shift.
- Keep the mechanics explicit and simple (which parameters are updated, learning rate,
  how many target labels are used).
- Plot: fine-tuning loss curves.

### Bloco 7 — Comparação final (a mensagem do bloco)
Three-way comparison, presented clearly enough that the lesson is unmistakable:
  (A) source-trained model, evaluated on SOURCE validation  (the original baseline)
  (B) source-trained model, evaluated on TARGET             (the degradation)
  (C) fine-tuned model,     evaluated on TARGET             (the repair)
- Compare with the same metrics: confusion matrices, per-class TPR/PPV/F1, ROC/AUC,
  and the probability-map / decision-region visuals.
- One compact summary figure and/or table holding all three.
- Short pt-BR closing narrative tying back to L2B1's concepts.

## Technical requirements
- Equinox + Optax (as in L1B2). Self-contained: no imports from external repos.
- Plots inline; DO NOT save figures to disk; nothing written to a committed path.
- Notebook must run top-to-bottom clean in the WinterSchool kernel and on a fresh Colab
  with an empty assets/ (self-generate everything; no committed artifacts).
- Keep total runtime modest (target: a few minutes CPU); toy-scale data and small nets.
- pt-BR markdown, Colab badge at top, "Para casa" exercises at the end.
