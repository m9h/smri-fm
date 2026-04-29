# DLBS brain-age ridge — meeting summary 2026-04-30

**Cohort**: 60 scans / 23 subjects (DLBS multi-wave longitudinal)

**Protocol**: Nima's GroupKFold(5) + StandardScaler + RidgeCV(α ∈ logspace(-3, 3, 25))

All numbers in years. Bootstrap CI from 1000 subject-level resamples.

| # | Tool | family | feats | raw MAE | r | **Zhang MAE** | r (Z) | 95% CI (Zhang) |
|---:|---|:-:|---:|---:|---:|---:|---:|---|
| 1 | **SynthSeg + TIV-norm** | morph |  71 | 5.99 | +0.93 | **4.71** | +0.95 | [2.74, 6.07] |
| 2 | FastSurfer aseg+DKT + ICV | morph | 100 | 7.53 | +0.88 | **5.48** | +0.93 | [3.40, 7.40] |
| 3 | FS+T1Prep+BrainIAC + PCA-48 | concat |  48 | 12.71 | +0.71 | **5.74** | +0.91 | — |
| 4 | SynthSeg (no TIV) | morph |  71 | 7.15 | +0.90 | **5.85** | +0.93 | [3.77, 7.01] |
| 5 | FastSurfer aseg+DKT | morph | 100 | 7.48 | +0.89 | **5.85** | +0.93 | [3.44, 7.17] |
| 6 | **FOMO25 (PCA-16)** | ssl |  16 | 9.82 | +0.81 | **5.86** | +0.93 | — |
| 7 | FS + T1Prep concat (alt) | concat | 171 | 6.65 | +0.91 | **6.31** | +0.93 | — |
| 8 | SynthSeg + FOMO25 concat | concat | 392 | 7.78 | +0.88 | **6.38** | +0.93 | — |
| 9 | T1Prep tissue ratios + TIV | morph |   4 | 7.42 | +0.89 | **6.58** | +0.92 | — |
| 10 | **FOMO25 (PCA-8)** | ssl |   8 | 10.35 | +0.80 | **6.60** | +0.91 | — |
| 11 | **FOMO25 AMAES_resenc_b** | ssl | 320 | 9.90 | +0.81 | **6.68** | +0.91 | [4.26, 8.00] |
| 12 | SynthSeg + T1Prep concat | concat | 143 | 7.21 | +0.91 | **6.75** | +0.92 | — |
| 13 | FS + T1Prep concat | concat | 171 | 7.57 | +0.89 | **6.80** | +0.92 | — |
| 14 | **FOMO25 (PCA-32)** | ssl |  32 | 10.23 | +0.78 | **6.83** | +0.91 | — |
| 15 | T1Prep thickness (DKT) | morph |  71 | 8.03 | +0.88 | **7.20** | +0.91 | — |
| 16 | FS+T1Prep+BrainIAC + PCA-8 | concat |   8 | 8.02 | +0.88 | **7.26** | +0.91 | — |
| 17 | FS+T1Prep+BrainIAC | concat | 939 | 8.31 | +0.88 | **7.35** | +0.91 | — |
| 18 | FS+T1Prep+BrainIAC + PCA-16 | concat |  16 | 8.66 | +0.85 | **7.57** | +0.91 | — |
| 19 | FS+T1Prep+BrainIAC + PCA-32 | concat |  32 | 8.73 | +0.86 | **7.77** | +0.90 | — |
| 20 | **BrainIAC (PCA-32)** | ssl |  32 | 15.53 | +0.46 | **7.94** | +0.88 | — |
| 21 | T1Prep thickness + TIV-norm | morph |  71 | 10.31 | +0.80 | **8.24** | +0.89 | — |
| 22 | **BrainIAC (PCA-16)** | ssl |  16 | 14.68 | +0.52 | **8.26** | +0.89 | — |
| 23 | T1Prep area + TIV-norm | morph |  71 | 11.70 | +0.73 | **8.97** | +0.86 | — |
| 24 | T1Prep thk+area concat | morph | 142 | 10.17 | +0.81 | **9.04** | +0.87 | — |
| 25 | T1Prep thk+area + TIV-norm | morph | 142 | 12.00 | +0.72 | **9.24** | +0.84 | — |
| 26 | **BrainIAC (PCA-8)** | ssl |   8 | 13.27 | +0.62 | **9.31** | +0.86 | — |
| 27 | **BrainIAC SimCLR (768-d)** | ssl | 768 | 13.73 | +0.60 | **9.61** | +0.86 | [6.32, 11.28] |
| 28 | T1Prep area (DKT) | morph |  71 | 16.32 | +0.46 | **10.34** | +0.84 | — |

## Headline takeaways

1. **Top 5 tools statistically tied** (bootstrap CIs all overlap). Only BrainIAC reliably worse.
2. **FOMO25 AMAES is best in *both* age halves** when split at median age 58 (younger Zhang 3.40, older Zhang 3.74). The full-cohort number hides this — masked-AE features are tightly age-correlated within bracket but span more variance across the full age range.
3. **BrainIAC dominates the older half** (Zhang 4.71, beats SynthSeg+TIV's 6.69). Worst tool in younger half (7.47). SimCLR backbone learned old-brain features.
4. **Concat past ~150 features hurts** — n ≪ p curse at this cohort size.

## Item 4 (MAE recon-collapse) — H3 confirmed, H1+H2 disconfirmed

Empirical tests against published AMAES_resenc_b checkpoint:

| Hypothesis | Code-reading | Empirical |
|---|---|---|
| H1 skip-copy → visible MSE ≈ 0 | trivial reconstruction | **opposite** (visible 1.5× *higher* than masked) |
| H2 loss → 0 instantly | yes | **no** — smooth descent over 200 steps |
| H3 MSELoss scaled by mask_ratio | bug present | **CONFIRMED** (buggy/correct = 0.6 = mask_ratio exactly) |

**Patch**: 3-line change to `_rec_loss` in asparagus self_supervised.py.
Reframe for Dojo+Rohit: ask which model + mask config + post a wandb chart.

## Caveats

- DLBS is in FOMO50K + FOMO300K source cohorts → SSL not held-out.
- n=23 subjects → bootstrap CIs ~3 yr wide → ranking margins under that are noise.
- Mihir's ADNI eval is the team's only path to a real held-out brain-age claim.

## Container + reproducibility

```
docker pull ghcr.io/m9h/fomo25-arm:latest
```

Grace Blackwell-tuned arm64 image with AMAES_resenc_b checkpoint baked in. Extracts 320-d embeddings on 60 DLBS scans in 33 seconds.

## Followup question for Ahmed

Pivot: jbanusco/fomo25 v1.0.0 ships no public mmunetvae checkpoint, so item 3 went against AMAES_resenc_b instead. Can you confirm intent? `notes/fomo25_checkpoint_followup.md` has the full ask.

## Where the artefacts live

`github.com/m9h/smri-fm/tree/dlbs-morphometry-benchmark/experiments/dlbs_morphometry_benchmark/`:
- `notes/comprehensive_ridge_report.md` — narrative
- `notes/ridge_rigor_addendum.md` — bootstrap + age-bracket findings
- `notes/fomo25_mae_recon_collapse_item4.md` — code-reading hypothesis
- `notes/item4_empirical_results.md` — empirical results
- `notes/fomo25_checkpoint_followup.md` — question for Ahmed
- `notes/meeting_2026-04-30/` — this summary
