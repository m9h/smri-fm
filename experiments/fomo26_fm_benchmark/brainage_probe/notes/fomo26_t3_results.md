# FOMO26 Task-3 brain-age: frozen FM linear-probe vs morphometry floor

**Goal:** complement the team's *finetune* baselines (Nima: AMAES 6.15, PDF-1M
6.58 MAE on the 50-subject TEST split) with two things they lacked — a **frozen
linear probe** (closed-form ridge on frozen FM embeddings) across the FM roster,
and a **classical-morphometry floor** — on the *same* official split, so the
numbers sit side by side.

Protocol: long-format features → `StandardScaler → RidgeCV`. Two evals per arm:
5-fold GroupKFold CV over all 494, and a fixed train→TEST eval on the official
`TEST_80_10_10` (Nima's protocol). raw + Zhang-corrected MAE. Each FM is extracted
at its **native asparagus input size** (AMAES 160³ pretrain; mmunetvae 64³;
fomo60k/anatcl/triad 96³ finetune) — see the de-confounding note below.

## Final table (cells = raw MAE / Zhang MAE; lower better)

| Arm | 5-fold CV (n=494) | fixed TEST (n=50) |
|---|---|---|
| **fomo60k comb_reg @96 (frozen)** | **6.05 / 5.57** (r .90) | **5.75 / 4.66** (r .91) |
| FastSurfer aseg+DKT (morphometry floor) | 6.41 / 5.74 (r .89) | 6.75 / 5.56 (r .85) |
| AMAES resenc_b @160 (frozen) | 6.89 / 5.77 (r .87) | 7.81 / 6.18 (r .81) |
| triad @96 (frozen) | 7.43 / 6.10 (r .85) | 7.81 / 6.15 (r .82) |
| mmunetvae @64 (frozen) | 8.26 / 6.34 (r .81) | 7.54 / 5.47 (r .84) |
| anatcl @96 (frozen) | 8.60 / 6.77 (r .79) | 8.16 / 6.17 (r .76) |

**Reference — asparagus FINETUNE, same TEST split (Nima, fold 0):**
AMAES official **6.15** · PDF-1M **6.58** (raw MAE).

## Headline

1. **Frozen-feature quality for brain age varies enormously across foundation
   models** — a 2.5-yr CV-MAE spread (6.05 → 8.60). Being "a foundation model"
   says nothing about whether its frozen features are good for age.
2. **Only fomo60k clears the bar.** Its frozen probe (CV 6.05 / TEST 5.75) beats
   the morphometry floor (6.41 / 6.75) **and** both finetuned baselines (AMAES
   6.15, PDF-1M 6.58) — with no finetuning. fomo60k is a SwinV2 MAE-pretrained on
   ~60k brain volumes.
3. **Every other FM's frozen probe trails the classical aseg-volume floor**
   (AMAES 6.89, triad 7.43, mmunetvae 8.26, anatcl 8.60 CV). For these, a ridge on
   FreeSurfer aseg volumes beats the frozen FM embedding.

## De-confounding (why the ranking is trustworthy)

Initial extraction put every FM at 96³, but AMAES pretrains at 160³ — a possible
disadvantage. Re-extracted AMAES at native **160³**: CV barely moved
(7.03 → **6.89**) and TEST worsened (7.28 → 7.81). So **resolution was not what
held AMAES back** — it is sub-floor at its native size too. This rules out
"AMAES just needed bigger crops" and confirms the cross-FM spread is a real
representation-quality effect, not a preprocessing artifact. (96³ row preserved
as `amaes96` for the record.) All other arms were already at their native
finetune size (96³ / 64³).

## Caveats

- **n=50 TEST is small.** Bootstrap on the floor-vs-frozen-AMAES diff gave 95% CI
  [−1.28, +2.37] — ~0.5-yr gaps are within noise; Nima's 6.15-vs-6.58 likewise.
  **The 5-fold CV (n=494) is the load-bearing column;** TEST is the comparability
  hook to the finetune numbers.
- **Frozen-vs-finetune is not a pixel-identical pipeline** (frozen probe uses the
  extractors' z-norm + crop, not asparagus's finetune dataloader); same subjects
  / labels / split, so floor-vs-frozen (both frozen-feature ridge) is clean and
  frozen-vs-finetune is indicative.
- **fomo60k variants:** only `combined_regular` probed; 3 other pkoutsouvelis
  variants on disk, not yet run. siam/brainiac/simclr3d need pretrained ckpts
  located (only finetuned ckpts on disk).

## Provenance / repro

- FastSurfer pre-computed on disk (494) → symlink-farm to `sub-XXX_ses-wave1` →
  features → ridge. FM embeddings via NGC `pytorch:26.04-py3`: AMAES/mmunetvae
  (extract_fomo25_*.py) + bridge roster (extract_bridge_embeddings.py looping the
  `asparagus_bridge` wrappers). Drivers in `run/`. Table: `scripts/compare_fomo26_t3.py`.
- Convention: every Task-3 scan is `sub-XXX / ses-wave1`, age in
  `participants.tsv` `AgeMRI_W1`, so the DLBS ridge harness runs unmodified.
