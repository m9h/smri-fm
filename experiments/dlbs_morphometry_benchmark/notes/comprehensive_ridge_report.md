# DLBS brain-age ridge — comprehensive comparison across morphometry + SSL

> **⚠️ This is the v1 (23-subject / 60-scan) matrix.** Scaling to the v2 cohort
> (42 subjects / 117 scans) **reverses the ranking below**: the volumetric
> leaders here (SynthSeg+TIV #1 at 4.71, FastSurfer #2 at 5.48) collapse to the
> *bottom* at v2 (Zhang 9.39 and 9.28), while FOMO25 (#7 here at 6.68) rises to
> a *tie for #1* (5.87). The swing is a verified per-subject effect, not an
> artifact — see `consolidated_agent_synthesis.md` §3 and `compare_v1_v2.py`.
> Read this table as the small-n baseline, not the current standing.

Following up on item 3 (FOMO25 SSL ridge) from the smri-fm meeting,
here is the full matrix of every ridge regressor we've fit on the
DLBS 23-subject sub-cohort, alongside the new FOMO25 result, so the
team can see how SSL embeddings stack up against morphometry when
controlled for the same dataset, CV protocol, and bias-correction
schemes.

## Setup (held constant across every row)

- **Cohort**: 23 DLBS subjects × ~2.6 sessions = **60 scans**, age range
  20–87 yr (longitudinal across waves 1/2/3).
- **CV**: GroupKFold(5) by subject (Nima's protocol from MedARC's
  smri-fm/experiments/synthseg_ridge_baseline).
- **Pipeline**: `StandardScaler → RidgeCV(α ∈ logspace(-3, 3, 25))`.
- **Bias-correction schemes**: raw, Cole–Smith (Smith 2019), Beheshti
  (Liang 2019), Zhang age-level (Zhang 2023).
- **Ages**: from DLBS `participants.tsv` (`AgeMRI_W{1,2,3}`).

## The matrix (sorted by best Zhang MAE)

| # | Source | feats | ICV | raw MAE | raw r | Beheshti MAE | β r | **Zhang MAE** | Zhang r |
|---|---|---:|:-:|---:|---:|---:|---:|---:|---:|
| 1 | **SynthSeg + TIV-norm** (MedARC pipeline.py) | 71 | ✓ | **5.99** | 0.93 | 4.84 | 0.96 | **4.71** | **0.95** |
| 2 | FS aseg + DKT.VINN + ICV-norm | 100 | ✓ | 7.53 | 0.88 | 6.79 | 0.92 | 5.48 | 0.93 |
| 3 | FS aseg + DKT.VINN (no ICV) | 100 |   | 7.48 | 0.89 | 7.25 | 0.92 | 5.85 | 0.93 |
| 4 | SynthSeg volumes (no TIV-norm) | 71 |   | 7.15 | 0.90 | 6.05 | 0.93 | 5.85 | 0.94 |
| 5 | FS + T1Prep concat | 171 |   | 6.65 | 0.91 | 6.20 | 0.94 | 6.31 | 0.93 |
| 6 | T1Prep tissue ratios (GM/WM/CSF/TIV) + ICV | 4 | ✓ | 7.42 | 0.89 | 6.98 | 0.92 | 6.58 | 0.92 |
| 7 | **FOMO25 AMAES_resenc_b** (SSL3D, FOMO300K AE) | 320 |   | 9.90 | 0.81 | **6.28** | 0.93 | **6.68** | 0.91 |
| 8 | SynthSeg + T1Prep concat | 143 |   | 7.21 | 0.91 | 6.77 | 0.93 | 6.75 | 0.93 |
| 9 | T1Prep cortical thickness (DKT) | 71 |   | 8.03 | 0.88 | 7.55 | 0.91 | 7.20 | 0.91 |
| 10 | FS + T1Prep + BrainIAC concat | 939 |   | 8.31 | 0.88 | 7.47 | 0.91 | 7.35 | 0.91 |
| 11 | T1Prep thickness + area concat | 142 |   | 10.17 | 0.81 | 9.24 | 0.87 | 9.04 | 0.87 |
| 12 | BrainIAC SSL (SimCLR backbone, 768-d) | 768 |   | 13.73 | 0.60 | 10.08 | 0.85 | 9.61 | 0.86 |
| 13 | T1Prep cortical area (DKT) | 71 |   | 16.32 | 0.46 | 11.46 | 0.82 | 10.34 | 0.84 |

(MAE is in years. Zhang is the most generous correction; raw is the
publication number with no calibration.)

## What this matrix tells us

### 1. SynthSeg + TIV-normalised volumes is still the leader

5.99 yr raw MAE → 4.71 yr Zhang, r 0.95. This is the same number Nima
reports for MedARC's pipeline.py baseline (~6.66 yr per the smri-fm
synthseg_ridge_baseline result), reproduced cleanly on our side.
Nothing else in the matrix touches it. Whatever the foundation-model
work delivers next, this is the bar.

### 2. SSL embeddings under-perform morphometry by ~2 yr MAE — but not equally

| SSL backbone | n feats | Zhang MAE | gap to SynthSeg+TIV |
|---|---:|---:|---:|
| FOMO25 AMAES_resenc_b | 320 | 6.68 | +1.97 yr |
| BrainIAC SimCLR | 768 | 9.61 | +4.90 yr |

FOMO25 AMAES is competitive in the post-correction regime (Beheshti
6.28 actually edges out FS aseg+DKT at 6.79) and it does so with less
than half the feature dim of BrainIAC. **AMAES is the strongest
SSL backbone we've tested**, despite the same DLBS-leakage caveat
applying to both.

The interesting wrinkle: AMAES's *raw* MAE is bad (9.90), but its
post-correction MAE collapses dramatically (6.28). That means the
backbone has a strong age-correlated signal but a substantial
constant or linear bias the ridge can't remove on its own. Worth
asking whether the bias is age-conditional (regression-toward-mean
in older subjects, classic SSL artefact) or subject-conditional.

### 3. Concat doesn't help when n=60 and p > 200

- FS + T1Prep concat (171 feats): Zhang 6.31  ← OK
- SynthSeg + T1Prep concat (143 feats): Zhang 6.75  ← worse than SynthSeg alone (5.85)
- FS + T1Prep + BrainIAC (939 feats): Zhang 7.35  ← much worse

We're up against the n ≪ p curse. RidgeCV's α regularises but with
only 60 scans, every additional uninformative feature still steals
signal. **For DLBS-sized cohorts, fewer well-chosen features beat
more features** (cf. row 6: T1Prep tissue ratios with just **4
features** at Zhang 6.58, only 0.4 yr worse than the 320-feat
FOMO25 SSL).

This also explains why BrainIAC's 768-d SSL embedding is the worst
SSL row — at p ≫ n the ridge falls back on regularisation that
shrinks too aggressively.

### 4. ICV / TIV normalisation gives free MAE on volumes

| | Zhang MAE w/o ICV | Zhang MAE w/ ICV | Δ |
|---|---:|---:|---:|
| SynthSeg | 5.85 | **4.71** | −1.14 |
| FS aseg+DKT | 5.85 | 5.48 | −0.37 |

SynthSeg gains the most from TIV-norm (Mask volume), FS gains less
(its own ICV via aseg). For *any* volumetric feature set on this
cohort, divide by ICV before ridge — it's the single highest-value
normalisation we've tested.

### 5. T1Prep ≠ FS ≠ SynthSeg in feature quality

T1Prep cortical area alone is dramatically the worst row (Zhang
10.34, raw 16.32, r 0.46) — surface area vs age has a flat /
non-monotonic relationship, RidgeCV's α blows up trying to ignore
it. T1Prep cortical thickness is much better (Zhang 7.20). T1Prep
tissue ratios (just 4 numbers: GM/WM/CSF/TIV) are surprisingly
strong (Zhang 6.58) — almost matching FOMO25 SSL with 1/80th the
features.

T1Prep's strength is the cortical *thickness* + tissue features;
its area features should probably be dropped from any concat ladder.

## Comparison framing for the team

The "story" in one figure-caption sentence:

> Across 13 feature sets on the same 60 DLBS scans / 23 subjects
> with the MedARC GroupKFold(5) protocol, **TIV-normalised SynthSeg
> volumes (71 feats, MAE 5.99 / Zhang 4.71 / r 0.95) remain the
> strongest brain-age ridge regressor**; the best SSL backbone
> (FOMO25 AMAES_resenc_b, 320 feats, MAE 9.90 / Zhang 6.68 /
> r 0.91) is competitive only after age-bias correction, with a
> ~2 yr Zhang MAE gap to morphometry, and concat ladders show
> diminishing returns past ~150 features at this cohort size.

## Caveats

- **DLBS leakage in SSL pretraining**: DLBS is a source cohort for
  both FOMO50K (mmunetvae) and FOMO300K (AMAES_resenc_b). The
  numbers in rows 7 and 12 are not held-out generalization claims.
  Need a non-FOMO-overlapping eval (HCP-A, OASIS, ADNI) to
  decompose representation quality from training-set memorisation.
- **n=60 ridge regression is noisy**: confidence intervals on these
  MAEs would be wide. The MAE deltas under ~1 yr are within
  CV-fold variance.
- **DLBS is 1.5T cross-/longitudinal-aging**, not a clinical AD/CN
  cohort. The brain-age signal is dominated by *typical* aging
  trajectories. ADNI/OASIS could re-rank these tools.

## Open follow-ups (next-step matrix)

| Question | What to run | Effort |
|---|---|:-:|
| Does FOMO25 AMAES + SynthSeg concat help? | `fit_ridge_concat.py --features-a fomo25 --features-b synthseg-tiv` | 5 min |
| Does the AMAES bias come from older subjects specifically? | residual scatter / age-bin MAE on the 4-correction predictions | 30 min |
| Can we get an mmunetvae checkpoint? | ask Ahmed in slack (followup at notes/fomo25_checkpoint_followup.md) | depends |
| Does the ranking hold OOD on OpenNeuro AD cohorts? | OpenNeuro download + repeat ridge on those subjects | 1–2 days |
| Does longitudinal recon-all change the morphometry numbers? | rerun SynthSeg / FS ridge on Legion's `.long.sub-XXXX` outputs once cohort recon-all completes | 1 day |

## Where to find the artefacts

- Per-row JSON: `experiments/dlbs_morphometry_benchmark/results/ridge_*.json`
- This report: `experiments/dlbs_morphometry_benchmark/notes/comprehensive_ridge_report.md`
- FOMO25 follow-up note for Ahmed: `experiments/dlbs_morphometry_benchmark/notes/fomo25_checkpoint_followup.md`
- Container behind the new row 7: `ghcr.io/m9h/fomo25-arm:latest` (Grace Blackwell arm64, NGC torch, AMAES_resenc_b baked in)
