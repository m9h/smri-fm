## structurebench RG-diagnostics vs downstream benchmark

Weights-only Martin-RG diagnostics (left) vs FOMO26 downstream (right).
MAE years lower=better; CLS002 acc, majority baseline 0.619; n=21 pooled CV.

| arm | alpha_median | frac_alpha_lt2 | phi_1_median | M_tr_median | n_traps_isolated_total | regr002_mae | cls002_acc_full | cls002_acc_dwi |
|---|---|---|---|---|---|---|---|---|
| siam | 2.519 | 0.333 | n/a | n/a | 0.000 | 5.662 | 0.619 | 0.667 |
| fomo60k | 3.298 | 0.066 | 0.013 | 144.220 | 0.000 | 5.657 | 0.429 | 0.476 |
| anatcl | 17.275 | 0.000 | n/a | n/a | 0.000 | 5.925 | 0.524 | 0.476 |
| simclr3d | 2.002 | 0.450 | n/a | n/a | 0.000 | 5.003 | 0.619 | 0.667 |
| triad | 7.525 | 0.000 | 0.012 | 153.391 | 0.000 | 11.650 | 0.524 | 0.524 |
| brainiac | 4.668 | 0.124 | 0.005 | 384.346 | 0.000 | 9.422 | n/a | n/a |
| mmunetvae | 14.815 | 0.000 | n/a | n/a | 0.000 | 10.390 | 0.571 | 0.619 |

### Spearman: diagnostic vs downstream metric (n arms with both)

| diagnostic | spearman_regr002_quality(-MAE) | spearman_cls002_acc_full | spearman_cls002_acc_dwi |
|---|---|---|---|
| alpha_median | -0.714 | -0.530 | -0.677 |
| frac_alpha_lt2 | 0.741 | 0.563 | 0.625 |
| phi_1_median | 0.500 | n/a | n/a |
| M_tr_median | -0.500 | n/a | n/a |
| M_tr_frac_median | 0.500 | n/a | n/a |
| n_traps_isolated_total | n/a | n/a | n/a |
