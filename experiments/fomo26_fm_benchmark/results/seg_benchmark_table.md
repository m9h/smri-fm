# structurebench-v1.0 — FOMO26 segmentation leaderboard

Foreground-mean test Dice (background class 0 excluded). Two arms with
real pretrained decoders; 200 epochs, 128^3 patches, split_80_10_10.

| Task | smri_siam | smri_mmunetvae |
|------|------|------|
| SEG009_FOMO26_Meningioma | 0.0000 | 0.0000 |
| SEG010_FOMO26_TrigeminalNeuralgia | 0.1838 | 0.2764 |

## Per-class Dice

### SEG009_FOMO26_Meningioma
| arm | class 0 | class 1 |
|-----|-----|-----|
| smri_siam | 0.9996 | 0.0000 |
| smri_mmunetvae | 0.9998 | 0.0000 |

### SEG010_FOMO26_TrigeminalNeuralgia
| arm | class 0 | class 1 | class 2 |
|-----|-----|-----|-----|
| smri_siam | 1.0000 | 0.3676 | 0.0000 |
| smri_mmunetvae | 1.0000 | 0.3293 | 0.2236 |

## Interpretation

- **SEG009 Meningioma: both arms score 0.0000 foreground Dice** — they
  predict all-background. The tumor class is tiny and sparse relative to
  the thin-slice (~29-slice) FLAIR volume, so the Dice-only objective is
  minimized by emptying the prediction; neither pretrained decoder rescues
  it. This is a task/loss-config failure (needs a region-balanced or
  compound loss + foreground oversampling), not an FM-quality signal — the
  arms are indistinguishable here.
- **SEG010 TrigeminalNeuralgia: mmunetvae > SIAM** (fg-mean 0.276 vs 0.184).
  mmunetvae recovers both foreground structures (class 1 0.329, class 2
  0.224) while SIAM finds only class 1 (0.368) and misses class 2 entirely
  (0.000). mmunetvae being the strongest seg arm is consistent with it being
  the most FOMO-domain-matched (pretrained on raw FOMO60K MRI) — notable
  given it is the *worst* arm on REGR002 brain-age, i.e. seg and regression
  rank the arms differently.
