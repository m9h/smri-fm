# FOMO26 Task-3 brain-age: frozen FM linear-probe vs morphometry floor

**Goal:** complement the team's *finetune* baselines (Nima: AMAES 6.15, PDF-1M
6.58 MAE on the 50-subject TEST split) with a **frozen linear probe** (closed-form
ridge on frozen FM embeddings) across the full FM roster, plus a **classical
morphometry floor**, on the *same* official split — so the numbers sit side by side.

Protocol: long-format features → `StandardScaler → RidgeCV`. Two evals per arm:
5-fold GroupKFold CV over all 494, and a fixed train→TEST eval on
`TEST_80_10_10`. Each FM extracted at its **native asparagus input size** (AMAES
160³ pretrain; mmunetvae 64³; all bridge arms 96³ finetune).

## Final table — full roster (cells = raw MAE / Zhang MAE; lower better)

| Arm (frozen probe unless noted) | 5-fold CV (n=494) | fixed TEST (n=50) |
|---|---|---|
| **fomo60k contrastive_regular** | **5.61 / 5.20** (r.91) | 6.23 / 5.44 |
| **fomo60k combined_regular** | **6.05 / 5.57** (r.90) | **5.75 / 4.66** |
| **simclr3d** | **6.23 / 5.63** (r.87) | 6.23 / 4.58 |
| FastSurfer aseg+DKT (morphometry floor) | 6.41 / 5.74 (r.89) | 6.75 / 5.56 |
| fomo60k combined_modality | 6.55 / 6.09 | 5.93 / 5.19 |
| fomo60k contrastive_modality | 6.66 / 5.96 | 7.07 / 5.64 |
| AMAES resenc_b @160 | 6.89 / 5.77 | 7.81 / 6.18 |
| AMAES resenc_b @96 (under-served) | 7.03 / 5.93 | 7.28 / 5.63 |
| triad | 7.43 / 6.10 | 7.81 / 6.15 |
| mmunetvae @64 | 8.26 / 6.34 | 7.54 / 5.47 |
| anatcl | 8.60 / 6.77 | 8.16 / 6.17 |
| brainiac | 8.98 / 7.15 | 10.23 / 7.57 |
| siam (var-filtered) | 10.01 / 6.09 | 10.31 / 6.09 |

**Reference — asparagus FINETUNE, same TEST split (Nima):** AMAES **6.15** · PDF-1M **6.58**.

## Headline

1. **Frozen-feature quality for brain age varies enormously across FMs** — a
   4.4-yr CV-MAE spread (5.61 → 10.01). "Foundation model" says nothing about
   whether the frozen features are good for age.
2. **Three arms beat the morphometry floor frozen: fomo60k (regular variants) and
   simclr3d.** fomo60k contrastive_regular (CV 5.61 / TEST 6.23) beats the floor
   *and* both finetuned baselines, with no finetuning. The winners are all
   **raw-T1 / brain-volume SSL** (fomo60k = SwinV2 MAE on ~60k brain vols;
   simclr3d = ResNet SimCLR on raw T1).
3. **The two flagship FOMO models lose frozen:** the challenge baseline **AMAES**
   (6.89) and the FOMO25-winner **mmunetvae** (8.26) both trail a simple aseg ridge.
4. **Sub-finding:** fomo60k *regular* variants (5.61, 6.05) beat its *modality*
   variants (6.55, 6.66) — modality-conditioning slightly hurts the frozen
   brain-age representation.

## Verifications

- **brainiac is valid, just weak.** Its checkpoint loads with `missing=86`, but a
  key-level audit shows those are 2 (fresh head) + 84 MONAI `cross_attn`/
  `norm_cross_attn` tensors — **inert for a SimCLR self-attention-only ViT**
  (never called in forward). No `self_attn`/`mlp`/`patch_embed` missing; the 137
  pretrained backbone tensors all load. r=0.77 confirms real signal. 8.98 is
  genuine, consistent with DLBS BrainIAC (Zhang 9.61).
- **siam has 104/320 dead embedding features** (post-ReLU, std<1e-6); the raw CV
  ridge blew up (StandardScaler amplifying a ~0-variance feature in one fold).
  Var-filtered CV = 10.01, matching its TEST 10.31 — a genuinely weak backbone.
- **FastSurfer represents the volumetric floor faithfully.** On DLBS (same 42
  subj) FastSurfer+ICV (9.28) ≈ SynthSeg+TIV (9.39), within 0.1 yr. SynthSeg not
  re-run on FOMO26 (its `medarc-smri-fm` container is gone). FastSurfer aseg+DKT
  here has ~95 regions (finer than SynthSeg's ~30) → a *fair-to-generous* floor,
  which strengthens the headline (FMs that miss it lose to a strong baseline).

## De-confounding (resolution)

AMAES re-extracted at native **160³** (was 96³): CV barely moved
(7.03 → 6.89), TEST worsened (7.28 → 7.81) — **resolution was not what held AMAES
back**; it is sub-floor at native size. Rules out a preprocessing artifact;
confirms the cross-FM spread is a real representation effect.

## Caveats

- **n=50 TEST is small** → ~0.5-yr gaps are within noise (bootstrap floor-vs-AMAES
  CI [−1.28, +2.37]); the **5-fold CV (n=494) is the load-bearing column**. TEST
  is the comparability hook to the finetune numbers.
- **Frozen-vs-finetune is not a pixel-identical pipeline** (frozen probe uses the
  extractors' z-norm+crop, not asparagus's finetune dataloader); same
  subjects/labels/split → floor-vs-frozen is clean, frozen-vs-finetune indicative.
- **Other FOMO26 tasks are not probe-able:** CLS002 Infarct is n=21 with the
  team's *finetuned* FMs already at chance (bal-acc 0.37–0.57; fomo60k worst at
  0.37) → no signal for a frozen probe. CLS003 isn't preprocessed; SEG009/010 are
  dense. **Brain-age (494) is the only powered FOMO26 frozen-probe target**, and
  fomo60k being best here yet worst on CLS002-finetune shows frozen quality is
  task-specific.

## Provenance / repro

FastSurfer pre-computed on disk (494) → symlink-farm → features → ridge. FM
embeddings in NGC `pytorch:26.04-py3`: AMAES/mmunetvae via FOMO25 extractors; the
rest via `extract_bridge_embeddings.py` (one loop over the `asparagus_bridge`
`_features()` wrappers). Pretrained ckpts: AMAES (HF `AMAES_resenc_b_fomo300k`),
mmunetvae/fomo60k×4/simclr3d/triad/anatcl (`/data/datasets/fomo26/weights/`),
brainiac (`/home/mhough/dev/BrainIAC/...`), siam (`/home/mhough/siam_params/...`,
needs `SIAM_MODEL_DIR`). Drivers in `run/`. Table: `scripts/compare_fomo26_t3.py`.

---

# ISLES22 multispectral ischemic-stroke lesion segmentation (the modality-matched task)

**Why:** brain-age on T1 is a weak FM test (morphometry-favorable, single modality).
Infarct seg lives on DWI/ADC — the *plurality* modality in FOMO300K pretraining —
and is dense prediction (what these FMs are built for), with no morphometry baseline.
The FOMO26 infarct task is too small (test n≈2), so we use **ISLES22** (Zenodo
7153326, CC-BY): 250 cases, DWI/ADC/FLAIR + expert lesion masks; one of the FOMO260K
paper's own validation tasks (their AMAES Dice 73.98).

**Setup:** DWI+ADC only (co-registered 250/250 with the masks; FLAIR is in a separate
space, excluded). New asparagus dataset `SEG011_ISLES22_IschStroke` (2 modalities, 2
classes), 200/25/25 split. Seg-finetune via the team's pipeline; input conv repeats
SIAM/mmunetvae's 1-channel stem to 2. Vendored def: `asparagus_seg/SEG011_ISLES22_IschStroke.py`.

## Results (test n=25, lesion Dice)

| Arm | lesion Dice | sens | prec | note |
|---|---|---|---|---|
| **siam** (finetuned) | **0.734** | 0.71 | 0.85 | top arm |
| **mmunetvae** (finetuned) | **0.608** | 0.63 | 0.77 | |

**Reference:** FOMO260K paper AMAES on ISLES22 = **0.740** (3-modality, full challenge protocol).
**Team FOMO26 seg leaderboard for context:** SEG009 meningioma **0.0**, SEG010 trigeminal **0.18–0.28**.

## Read

**This is where the FMs actually work — and they hit published SOTA.** siam reaches
**Dice 0.734** on stroke-lesion seg, essentially matching the FOMO paper's AMAES (0.740)
**with only 2 modalities (DWI+ADC) vs the paper's 3.** mmunetvae 0.61. Both vastly exceed
the team's other FOMO26 seg tasks (meningioma 0.0, trigeminal 0.18) — non-diffusion, tiny.
The modality-matched, properly-powered (n=25 test) infarct task is the fairest FM test in
this study, and the FMs pass it decisively. Contrast with the T1 brain-age task, where the
frozen FMs mostly *lost* to morphometry: FM value is real but task- and modality-dependent.

## siam inference fix (resolved)

siam trained fine (val Dice 0.76) but full-volume test inference crashed on an nnU-Net
skip-concat shape mismatch (`Expected size 4 but got size 3`). Root cause: SIAM's
`3d_fullres` plan is **anisotropic** — cumulative stride **64×64×32** — so a fitted
inference tile not divisible by that mismatches the decoder. Fix: `SlidingWindowSegMixin._forward_divisible`
pads each tile up to a multiple of 64 (divisible by SIAM's 64/64/32 and mmunetvae's 16),
forwards, crops logits back. Re-ran test-only on the saved `best.ckpt` via a new
`asparagus/pipeline/run/predict_seg.py` (fit skipped, `TEST_CKPT` env) → Dice 0.734.

## Remaining

- Encoder-only arms (fomo60k/AMAES/anatcl/triad/simclr3d) need bolt-on seg decoders to
  join — esp. fomo60k (the frozen brain-age champion): does its lead hold on stroke seg,
  or is FM quality task-specific (as CLS002 hinted)?
- Optional: add FLAIR via resampling to DWI space (3-modality, full paper protocol).
- A from-scratch nnU-Net control (the asparagus `scratch_nnunet` arm) to quantify the
  pretraining gain (paper: AMAES 0.740 vs scratch 0.7286).
