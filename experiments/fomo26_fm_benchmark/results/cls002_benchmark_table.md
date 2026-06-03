# structurebench-v1.0 — FOMO26 CLS002 Infarct leaderboard

Pooled-CV classification metrics (n=21). Majority (predict-all-positive) baseline accuracy = 0.619. DWI = single diffusion channel; FULL =
all available channels. Accuracy is the headline; bold beats baseline.

| arm | DWI_accuracy | DWI_balanced_accuracy | DWI_f1 | FULL_accuracy | FULL_balanced_accuracy | FULL_f1 |
|-----|-----|-----|-----|-----|-----|-----|
| smri_siam | **0.667** | 0.562 | 0.788 | 0.619 | 0.500 | 0.765 |
| smri_simclr3d | **0.667** | 0.611 | 0.759 | 0.619 | 0.572 | 0.714 |
| smri_mmunetvae | 0.619 | 0.500 | 0.765 | 0.571 | 0.510 | 0.690 |
| smri_anatcl | 0.476 | 0.457 | 0.560 | 0.524 | 0.471 | 0.643 |
| smri_fomo60k | 0.476 | 0.385 | 0.645 | 0.429 | 0.370 | 0.571 |
| smri_triad | 0.524 | 0.423 | 0.688 | 0.524 | 0.567 | 0.500 |
| smri_brainiac | n/a | n/a | n/a | n/a | n/a | n/a |

## Interpretation

- **No arm meaningfully separates from the majority prior.** The best
  FULL-input accuracies (siam, simclr3d 0.619) only *tie* the 0.619
  baseline; only siam and simclr3d clear it on DWI-only (0.667). High
  recall at baseline-level accuracy means the better arms mostly predict
  the positive class — balanced accuracy (~0.50-0.61) confirms weak true
  discrimination on this small (n=21) infarct cohort.
- **DWI >= FULL for the top arms** (siam 0.667 vs 0.619; simclr3d 0.667 vs
  0.619), consistent with infarct being a diffusion-salient finding and
  with FOMO300K's diffusion-plurality pretraining corpus.
- **brainiac is absent** (CLS002 sweep produced no result; same gap as its
  REGR002 row). Reported n/a rather than excluded — a re-run would be
  needed to place it.
