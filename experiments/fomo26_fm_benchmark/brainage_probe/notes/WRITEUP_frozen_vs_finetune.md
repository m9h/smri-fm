# Foundation models for sMRI: frozen probes mis-rank what finetuning rewards

*A two-part study on FOMO26-adjacent tasks. Companion to the team's finetune
baselines; all code on `fomo26-siam-bridge`, results in
`experiments/fomo26_fm_benchmark/brainage_probe/`.*

## TL;DR

1. **On finetuned, modality-matched, dense prediction (ISLES22 ischemic-stroke
   lesion segmentation), foundation-model pretraining clearly helps** — most FM
   backbones beat an identical-architecture from-scratch control by **+0.07–0.10
   Dice**, on a leakage-verified held-out set.
2. **But frozen linear-probe rankings do not predict that finetuned performance**
   (Spearman ρ ≈ 0.31 across 6 backbones, n.s.). The frozen-probe *winner* isn't the
   seg winner; the frozen-probe *loser* (SIAM) is a seg winner; a strong-frozen arm
   (SimCLR3D) is mediocre finetuned.
3. **Pretraining is not free insurance:** one backbone (AnatCL) finetunes to *below*
   from-scratch. A foundation model is not automatically better than random init.

**Practical message: don't select or reject an sMRI foundation model on a frozen
linear probe — it's an unreliable proxy for finetuned utility.**

## Why these two tasks

The team's brain-age (REGR002) baselines are T1-only — a regime where classical
morphometry is already strong and the FMs' multimodal/diffusion pretraining is barely
exercised. So we ran two complementary probes:

- **Brain-age frozen linear probe + a morphometry floor** (cheap, label-light) — to
  read frozen representation quality and compare against a classical baseline.
- **ISLES22 stroke-lesion segmentation, finetuned** — the *modality-matched* task
  (DWI/ADC, the plurality modality in FOMO300K pretraining), dense prediction (what
  these models are built for), with a from-scratch control.

## Part 1 — Brain-age frozen probe (DLBS REGR002, T1, ridge on frozen embeddings)

12-arm roster, GroupKFold-5 over 494 scans (the load-bearing column; the 50-subject
official TEST is too small to rank). Each FM extracted at its native input size.

- **Only 2 of the FMs beat the FastSurfer aseg-volume floor (CV 6.41 MAE) frozen:**
  fomo60k-regular (**6.05**) and simclr3d (**6.23**). The flagship FOMO models *lost*:
  AMAES 6.89, the FOMO25-winner mmunetvae 8.26; SIAM was worst (10.01, ~⅓ of its
  embedding channels are dead post-ReLU).
- Frozen-feature quality spans **4.4 years** of MAE across the roster.

*Caveats:* this is a linear probe, not the team's finetune protocol; frozen-vs-finetune
isn't a pixel-identical pipeline; and **DLBS is in FOMO pretraining** (verified in the
manifest), so brain-age carries a memorization confound. Treat Part 1 as exploratory
context, not a headline.

## Part 2 — ISLES22 stroke-lesion segmentation (finetuned, the controlled result)

ISLES22 (Zenodo 7153326, CC-BY): 250 cases, DWI/ADC/FLAIR + expert lesion masks; one of
the FOMO260K paper's own held-out validation tasks. We added it as a new asparagus seg
task (`SEG011_ISLES22_IschStroke`, DWI+ADC — co-registered with the masks; FLAIR is
off-grid). **Leakage-verified clean:** grepped both the FOMO260K (public) and FOMO300K
(gated) `mri_info.tsv` manifests — ISLES22 is absent from all 910/915 source datasets.
5-fold, 200-epoch finetune; from-scratch nnU-Net control; run on Modal (A100).

**5-fold lesion Dice (mean ± std):**

| Arm | Backbone / decoder | Dice | vs scratch |
|---|---|---|---|
| triad | Swin-MAE / SwinUNETR | **0.763 ± 0.012** | +0.10 |
| fomo60k | Swin-MAE / SwinUNETR | 0.754 ± 0.047 | +0.09 |
| siam | nnU-Net / nnU-Net | 0.740 ± 0.015 | +0.07 |
| mmunetvae | UNet-VAE / native | 0.734 ± 0.033 | +0.07 |
| simclr3d | ResNet-18 / ResNet-UNet | 0.678 ± 0.025 | +0.01 |
| **scratch** | nnU-Net (random init) | **0.667 ± 0.021** | — |
| anatcl | ResNet-18 / ResNet-UNet | 0.617 ± 0.010 | **−0.05** |

(brainiac deferred: its ViT pretrained at 96³ + UNETR's fixed input size conflict with
the 128³ sliding-window inference.)

**Findings:**
- **Pretraining helps the strong backbones** (+0.07–0.10), but **not universally** —
  AnatCL finetunes to below scratch.
- **Frozen↔finetune ranking is decoupled** (Spearman ρ ≈ 0.31, n=6, n.s.):
  - **SIAM** — *worst* frozen (dead features) — is **3rd on seg and beats scratch +0.07
    on the *same* nnU-Net decoder** (a decoder-confound-free comparison).
  - **SimCLR3D** — *2nd-best* frozen — only ties scratch.
- Transformer-MAE backbones (triad, fomo60k) are the most robust across both tasks.

## The synthesis

Frozen linear-probe rank and finetuned-seg rank don't track each other. Finetuning
*rescues* some backbones the frozen probe writes off (SIAM) and fails to rescue others
(AnatCL); a backbone that probes well frozen (SimCLR3D) can underwhelm finetuned. So a
frozen probe is a poor model-selection signal for any task you intend to finetune.

## Limitations (read before citing)

- **Cross-arm seg ranking mixes encoder + decoder.** Decoders differ by arm
  (SwinUNETR / our ResNet-UNet / nnU-Net), so the ResNet arms' weakness may be partly
  the simpler decoder. *The clean follow-on is a uniform shared decoder.* What survives
  the confound: SIAM is worst-frozen yet +0.07 over scratch **on the identical nnU-Net
  decoder** — no decoder difference explains that.
- Our from-scratch baseline (0.667) is weaker than the FOMO paper's few-shot scratch
  (0.729) — different topology/budget/2-vs-3 modalities — so the +0.07 gains are "vs
  *this* scratch," internally controlled, not the paper's number.
- n=25 test/fold; 5-fold gives the error bars. ρ=0.31 at n=6 isn't significant on its
  own, but the per-arm dissociations (SIAM, SimCLR3D) are individually clear.

## Reusable for the team

- **`SEG011_ISLES22_IschStroke`** — a new working seg task; the team's seg leaderboard
  was meningioma 0.0 / trigeminal 0.18 (both failures), this one reaches ~0.74.
- **A SIAM seg-inference bug fix** — SIAM's anisotropic `3d_fullres` plan (stride
  64×64×32) crashed sliding-window inference; fixed in `seg_inference.py` (pad tiles to
  mult-of-64). Helps any SIAM seg run.
- **Bolt-on seg decoders** for the encoder-only FM arms (SwinUNETR re-parent for Swin,
  ResNet-UNet for ResNet) — `models_smri_*.py` + `seg_decoders.py`, wired so asparagus's
  encoder/decoder LR split works.
- **Modal seg sweep** (`scripts/modal_seg_sweep.py`) — 5-fold × roster in ~2 h vs ~30 h
  on one GPU, with a `--debug` smoke for cheap red-green validation.

## Next steps

1. **Uniform decoder** across arms → a clean *encoder* ranking on seg (removes the
   decoder confound).
2. Add **brainiac** (UNETR at 96³ with a matched patch size) for completeness.
3. **3-modality** ISLES22 (resample FLAIR to DWI) for the exact paper protocol.
4. A **better-tuned scratch** (proper nnU-Net schedule) to test how much of the +0.07
   gain a stronger random-init baseline closes.
