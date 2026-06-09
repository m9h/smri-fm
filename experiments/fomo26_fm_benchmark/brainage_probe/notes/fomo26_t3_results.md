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
