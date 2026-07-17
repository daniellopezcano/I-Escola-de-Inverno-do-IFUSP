# Append-only brief — L1B2 notebook (00_caixa_de_ferramentas)

## Scope discipline (STRICT)
- APPEND new sections only. Do NOT modify, reorder, or delete ANY existing cell.
- Reuse the objects already defined earlier in the notebook: the trained "small" and
  "big" models, the damped-sinusoid generator `senoide_amortecida`, the training-x range,
  noise level, and any existing plotting helpers. Do not redefine them; if a helper is
  needed, define it inline in the new cells.
- pt-BR markdown, same style/tone as the rest of the notebook. Short cells, one idea each.
- Produce nice visualization plots inline; DO NOT save any figure to disk.
- Keep artifact hygiene: nothing written to a committed path.

## New sections to append (in this order)

### Seção A — Desempenho em dados de teste (small vs big)
- Sample a NEW test set from the SAME distribution used for training (same
  `senoide_amortecida`, same lam, same x-range, fresh noise, new random seed).
- Evaluate BOTH the small and the big model on it; report test loss (MSE) for each.
- Plot: the test data points + both models' predictions over the x-range; annotate the
  two test losses. The intended lesson: the big model overfit (good train, worse test);
  the small model generalizes better. Add one or two lines of pt-BR narrative making
  this point.

### Seção B — Extrapolação fora do domínio de treino (small model)
- Use the SMALL model. Evaluate it on x values EXTENDED beyond the training range (e.g.
  extend to the left and right of [x_min, x_max]).
- Plot: true `senoide_amortecida` curve over the extended range vs. the small model's
  prediction; shade or mark the training region so the in-range vs. out-of-range contrast
  is obvious. Lesson (pt-BR narrative, brief): neural nets do not reliably extrapolate —
  ties directly to L1B1's "no reliable extrapolation" limitation.

### Seção C — Mudança de parâmetro: lam ligeiramente diferente (small model)
- Use the SMALL model (trained on the original lam). Generate data from the SAME
  `senoide_amortecida` but with a slightly DIFFERENT `lam` (state the value; keep the
  same x-range).
- Plot: the new (shifted-lam) data/true curve vs. the small model's prediction; report
  the loss. Brief pt-BR narrative: the model degrades when the data-generating process
  shifts — a gentle, concrete foreshadowing of domain shift (L2), even though here it's a
  parameter change rather than a full domain gap. Do not over-explain; one short paragraph.

## Verification
- The whole notebook must still run top-to-bottom clean in the WinterSchool kernel,
  with the three new sections producing their plots, and NO existing cell altered.
