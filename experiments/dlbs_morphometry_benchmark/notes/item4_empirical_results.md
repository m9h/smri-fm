# Item 4 — empirical results

Ran the four tests proposed in `fomo25_mae_recon_collapse_item4.md`
against the published `FOMO-MRI/AMAES_resenc_b` checkpoint and the
asparagus SSL training code, on 5–10 BrainIAC-preprocessed DLBS T1s.
Results are mixed: **one hypothesis confirmed cleanly, two
disconfirmed**. The disconfirmation is itself useful — it tells
Dojo+Rohit which paths to *not* go down and where the real signal
might be.

Raw outputs at `notes/item4_empirical/`:
- `item4_results.json` — all measurements
- `item4_recon_figure.png` — reconstruction visualization (T4)
- `item4_training_curve.png` — 200-step SSL training trajectory (T3)

## H3 (MSELoss scaling bug) — **CONFIRMED ✅**

Pure-pytorch unit test:

```
buggy_loss                 = 0.97874719
correct_masked_only_loss   = 1.63170373
mask_ratio                 = 0.59983146
buggy / correct            = 0.59983143    ← matches mask_ratio exactly
predicted_buggy = correct × mask_ratio = 0.97874724
delta(buggy, predicted)    = 4.7e-08      ← floating-point noise
```

The reported loss is **multiplied by the mask ratio** (~0.6×). For a
true masked-region MSE of e.g. 0.5, the reported number is 0.3 — a
~40% under-statement of how poorly the model reconstructs the
masked regions. **Cross-run / cross-config loss comparisons are
distorted by this bug** unless mask_ratio is held constant.

**Recommended fix**: change `_rec_loss` to compute
`((pred - y)[mask] ** 2).mean()` directly. Three-line patch.

## H1 (skip-copy makes visible MSE near zero) — **DISCONFIRMED ❌**

Loaded the published AMAES_resenc_b checkpoint, ran inference on 5
T1s with 60% token-mask, measured MSE on visible vs masked regions:

| Scan | MSE visible | MSE masked | Ratio |
|---|---:|---:|---:|
| sub-1003_ses-wave1 | 2.91 | 1.82 | **1.60** |
| sub-1003_ses-wave2 | 2.93 | 1.89 | **1.55** |
| sub-1003_ses-wave3 | 2.99 | 1.91 | **1.57** |
| sub-1007_ses-wave1 | 3.07 | 1.96 | **1.57** |
| sub-1007_ses-wave2 | 3.03 | 1.99 | **1.52** |

**Visible MSE is *higher* than masked MSE, not lower.** The opposite
of what the skip-copy hypothesis predicted. The published model is
in fact reconstructing masked regions *better* than it's
reconstructing visible regions. Skip connections in this U-Net are
*not* acting as identity-pass-through during inference — they pass
*features*, which the decoder integrates with the bottleneck.

This means **the model is genuinely doing MAE inference**, not
gaming the loss via skip-copying. Whatever Dojo+Rohit are seeing,
the skip-copy explanation isn't it.

## H2 (loss → 0 instantly during training) — **DISCONFIRMED ❌**

Trained a fresh random-init `resenc_unet_b` for 200 steps on a 4-T1
batch with the buggy loss. Trajectory:

| step | buggy loss | MSE visible | MSE masked | MSE total |
|---:|---:|---:|---:|---:|
| 0 | 0.4867 | 0.9006 | 0.8111 | 0.8469 |
| 9 | 0.1831 | 0.7990 | 0.3051 | 0.5027 |
| 19 | 0.1524 | 0.5797 | 0.2541 | 0.3843 |
| 49 | 0.0958 | 0.3745 | 0.1597 | 0.2456 |
| 99 | 0.0740 | 0.2358 | 0.1233 | 0.1683 |
| 199 | 0.0548 | 0.1477 | 0.0914 | 0.1139 |

Two findings:

1. **Loss does not collapse to zero "instantly"**. It descends
   smoothly from 0.49 → 0.05 over 200 steps — a normal SSL training
   curve, not a pathological instant-zero collapse.
2. **Masked-region MSE descends faster than visible-region MSE.**
   By step 9 the model is already 2.7× better on masked regions
   (MSE 0.31) than visible (MSE 0.80). At step 199 masked MSE is
   0.5× of visible. So the model is learning to *fill in* masked
   regions and largely failing to *preserve* the visible ones —
   counterintuitive but consistent with H1's disconfirmation.

The "loss → 0 instantly" pattern Dojo+Rohit observed must come from
something else — possibly:

- A different model (mmunetvae has a different architecture; we
  tested AMAES_resenc_b because that's what's actually published).
- A different masking scheme (e.g., `mask_ratio` very low, or
  spatial vs token masking matters).
- A specific data-normalization that makes targets near-constant.
- Logging-driven artifact: H3's scaling bug shrinks reported numbers
  by 40%, which combined with a normal-but-low SSL loss could *look*
  like an "instant zero" from a wandb chart.

## What this means for the team

| Hypothesis | Code reading said | Empirical test says |
|---|---|---|
| H1 skip-copy | trivial visible reconstruction | **opposite** — masked is better |
| H2 instant loss collapse | yes | **no** — smooth descent |
| H3 MSE scaling bug | bug present | **confirmed** to factor mask_ratio |

So **the only verified contribution to the "loss looks weird" pattern is H3** — the scaling bug. That alone could explain a ~40% optical reduction in reported numbers across the SSL training. The architectural arguments about skip connections were wrong against this checkpoint.

## Recommended next moves

1. **Patch H3** (3-line change) — costs nothing, makes loss numbers
   comparable across configs. Worth doing regardless.
2. **Re-frame H1/H2 question for Dojo+Rohit**: ask which model
   they're training (mmunetvae or AMAES_resenc_b?), what mask ratio
   and token size, and post a representative wandb screenshot. The
   "loss → 0" symptom may be specific to their config.
3. **Inspect the training data normalization**: if the pretraining
   pipeline normalizes targets in a way that makes them
   near-constant within tokens, the loss could indeed approach zero
   trivially. The `Torch_Normalize` step in the pretrain transforms
   is suspect.
4. **(Optional) Re-run T1+T3 against mmunetvae** if a checkpoint
   surfaces. The skip-copy + collapse predictions might still hold
   for a different architecture.

## Caveats on this analysis

- The published AMAES_resenc_b checkpoint has been trained to
  convergence on FOMO300K (1000s of GPU-hours). Its inference-time
  MSE numbers are after extensive learning — they don't tell us
  what the *training-time* loss trajectory looked like. T3 (200 fresh
  steps) is what we use to probe training dynamics, and it doesn't
  show the pathological collapse.
- The token-mask we used (size=4, ratio=0.6) matches the gardening_tools
  default. If pretraining used different parameters, behavior could
  differ.
- T1's 1.5–1.6 visible/masked MSE ratio is *interesting on its own
  terms*: it suggests the published model has learned a
  representation that is more accurate at predicting masked content
  from context than at preserving fine detail in visible regions.
  That's a feature of MAE-trained models in general, but the
  magnitude of the asymmetry here is worth flagging.

## Pointers

- Test script: `experiments/dlbs_morphometry_benchmark/scripts/item4_empirical_tests.py`
- Raw output: `experiments/dlbs_morphometry_benchmark/notes/item4_empirical/`
- Original code-reading hypothesis: `experiments/dlbs_morphometry_benchmark/notes/fomo25_mae_recon_collapse_item4.md`
- Container the tests ran in: `ghcr.io/m9h/fomo25-arm:latest`
