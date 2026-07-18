# Polish brief — L2B2 notebook (01_domain_shift_toy), round 2

## Context
The current notebook has the right seven-block skeleton but is TOO SIMPLE: the class
clouds are trivially separable, the point counts are low, the learned decision contours
are uninteresting, and the target shift is too plain. This round raises sophistication
while KEEPING the seven-block structure, the pt-BR narrative, the Equinox/Optax style,
the inline-helper philosophy, and the L1B2 tone. Do not restructure the blocks.

Output: jax-examples/notebooks/01_domain_shift_toy_v2.ipynb (leave the canonical
01_domain_shift_toy.ipynb untouched). Resync the mirror from the canonical notebook
FIRST, per the source-of-truth policy in CLAUDE.md.

## REQUIRED CHANGES

### R1 — A genuinely non-trivial source distribution (Bloco 1)
Three classes whose supports are NON-CONVEX and INTERLEAVED, so the classifier must
learn curved, non-obvious contours. Concretely (adapt as needed, but keep this spirit):
- one class formed by TWO DISJOINT modes (a multi-modal class forces disconnected
  decision regions — pedagogically the single most valuable feature)
- one class shaped as a curved crescent / arc that partially wraps another
- one broader, more diffuse class acting as a partial background that OVERLAPS the
  others in specific regions (so errors concentrate in identifiable places)
Requirements:
- Classes must PARTIALLY OVERLAP — irreducible Bayes error must be clearly non-zero,
  and it must be VISIBLE in the scatter which regions are ambiguous.
- Keep the classes IMBALANCED (state proportions explicitly, e.g. ~60/30/10).
- RAISE the sample size substantially (target ~6000–12000 training points, plus
  validation), while keeping total notebook runtime modest on CPU (a few minutes).
- The generator must be a clean, parameterized function so Bloco 4 can perturb it.
Also: check the model has enough capacity for this harder problem — tune width/depth/
epochs so the fit is good but NOT perfect, and show train/val curves demonstrating it.

### R2 — Confusion matrix upgrade (Bloco 3, reused in 5 and 7)
Replace the current confusion-matrix plot with one modeled on the instructor's
`plot_confusion_matrix` (reference implementation below). Required visual features:
- row-normalized colouring with an 'RdBu' (or similar diverging) colormap, vmin=0 vmax=1,
  with a colorbar
- each cell annotated with the COUNT and the row PERCENTAGE
- the DIAGONAL cells additionally annotated with per-class TPR (recall), PPV (precision)
  and F1, computed from the confusion matrix itself
- automatic text colour switching (white on dark cells, black on light) via a threshold
- class names on both axes, rotated x labels, title parameter
Adapt it to this notebook: implement it INLINE where first needed, in pt-BR, using the
notebook's own conventions; drop the `logging` dependency (use simple guards); keep
numpy/matplotlib. Handle missing/absent classes gracefully (zero rows must not crash or
produce NaN). Reference implementation to adapt:

```python
def plot_confusion_matrix(yy_true, yy_pred_P, class_names, normalize=True,
                          figsize=(10, 7), cmap='RdBu', title=None, threshold_color=0.5):
    yy_pred = np.argmax(yy_pred_P, axis=1)
    unique_classes = np.unique(yy_true); num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(yy_true, yy_pred):
        if t in unique_classes:
            cm[int(t), int(p)] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, where=row_sums != 0)
    norm = Normalize(vmin=0, vmax=1)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap, norm=norm)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(num_classes)); ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    ax.set_title(title if title else 'Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
    precision, recall, f1 = [], [], []
    for i in range(num_classes):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        p_i = tp / (tp + fp) if tp + fp > 0 else 0.0
        r_i = tp / (tp + fn) if tp + fn > 0 else 0.0
        f_i = 2 * p_i * r_i / (p_i + r_i) if p_i + r_i > 0 else 0.0
        precision.append(p_i); recall.append(r_i); f1.append(f_i)
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j]
            percent = cm_percent[i, j] * 100 if row_sums[i] != 0 else 0
            text_color = "white" if cm_percent[i, j] > threshold_color else "black"
            if i == j:
                text = (f"{count}\nTPR:{recall[i]*100:.1f}%"
                        f"\nPPV:{precision[i]*100:.1f}%\nF1:{f1[i]:.2f}")
                ax.text(j, i, text, ha="center", va="center", color=text_color,
                        fontsize=10, fontweight='bold')
            else:
                ax.text(j, i, f"{count}\n{percent:.1f}%", ha="center", va="center",
                        color=text_color, fontsize=11)
    plt.tight_layout(); plt.show()
    return cm
```

### R3 — A non-trivial target shift (Bloco 4)
The target must NOT be a single global translation. Make the perturbation
CLASS-DEPENDENT and heterogeneous, so degradation differs per class (a much richer
lesson). For example, combine:
- one class rotated and/or displaced along a curved direction
- one class whose SPREAD/covariance changes (more diffuse in the target)
- one class that barely moves at all (so students see that shift damage is uneven)
- plus a PRIOR SHIFT: change the class proportions between source and target
State explicitly, in pt-BR, which named shift types from L2B1 are present (covariate
shift and prior shift), and keep the shift subtle enough to be realistic — the target
must still look like "the same problem, slightly different world", not a new dataset.
The block's closing figure must make the source→target change unmistakable (overlay
and/or side-by-side, with the shift per class visible).

## ADDITIONAL IMPROVEMENTS (include; they are cheap and high-value)

### A1 — Confidence / calibration under shift (extends Bloco 5)
Histogram of the model's max-probability (confidence) on source-validation vs. target.
Show that the model remains CONFIDENT while becoming WRONG. Add a simple reliability
diagram (accuracy vs. confidence bins) for source and target side by side. This is the
direct experimental payoff of L2B1's "errado e confiante" warning. Keep it compact.

### A2 — Shift detection WITHOUT target labels (small addition to Bloco 4 or 5)
Train a tiny domain classifier (source vs. target, ignoring class labels). A high AUC
proves the shift is detectable from unlabeled data alone. 5–10 lines; it operationalizes
L2B1's detection checklist and is a habit students can reuse.

### A3 — Error localization map (extends Bloco 3 and 5)
A 2D plot marking WHERE the misclassified points live in feature space. On the source it
shows errors concentrating in the genuine overlap regions (irreducible); on the target it
shows errors migrating into the shifted regions (fixable). Contrasting these two panels
distinguishes "the physics is hard" from "my training set was wrong".

### A4 — How many target labels do you need? (extends Bloco 6)
A small sweep of the number of labeled target examples K used for fine-tuning
(e.g. K ∈ {25, 50, 100, 250, 500, 1000}), plotting a metric (macro-F1) vs. K, with the
zero-shot (source→target) level as a horizontal reference. Teaches the label-budget
question directly. Keep K values few and the nets small so runtime stays modest.

### A5 — Full fine-tune vs. partial (small addition to Bloco 6)
Compare fine-tuning ALL parameters against fine-tuning only the last layer(s), at the
same K. Frame it as a transfer-learning design choice (what to adapt vs. what to keep),
without invoking latent-space language. One extra curve or one extra row in the table.

### A6 — Metric hygiene notes (throughout Bloco 3)
Where the imbalance makes it matter, briefly contrast ACCURACY with MACRO-averaged
metrics, and note in one or two lines why accuracy is misleading for the rare class.
Keep the ROC/AUC as one-vs-rest per class, with the correct-handling guard (no NaN from
degenerate single-class slices) verified numerically.

## OPTIONAL (include only if runtime and length allow)
- A brief note on the operating point: moving the decision threshold trades PPV against
  TPR for the rare class, tied to a "how pure must a candidate list be?" framing.
- A one-line seed/variance caveat (results shift slightly with the random seed).

## CONSTRAINTS (unchanged)
- Keep the seven-block structure and pt-BR narrative; match L1B2's style.
- Equinox + Optax; self-contained; no external repo imports.
- Plots inline; SAVE NO FIGURES; nothing written to a committed path.
- Must run top-to-bottom clean in the WinterSchool kernel and on a fresh Colab.
- Do NOT build the notebook around latent spaces (that is L3's material).
- Keep total runtime modest on CPU; prefer smaller nets/fewer epochs over long training.
- Length discipline: this is a polish, not a bloat. Prefer better figures and sharper
  narrative over more text.
