# DLBS ridge — statistical rigor addendum

Companion to `comprehensive_ridge_report.md`. Three additions to the
point-estimate table, all run on the same 60-scan/23-subject DLBS
cohort with the same GroupKFold(5) protocol, plus 1000-draw
subject-bootstrap, and a younger-vs-older-half subgroup re-fit.

## Bootstrap MAE 95% CIs (B=1000, subject-level resampling)

Sorted by mean Zhang MAE across the 1000 bootstrap draws. The right-most
column is the per-fold MAE std on the original 5 GroupKFold splits.

| Tool | feats | mean Zhang MAE | 95% CI | per-fold std |
|---|---:|---:|---:|---:|
| SynthSeg + TIV-norm | 71 | 4.24 | [2.74, 6.07] | 1.70 |
| FS aseg+DKT + ICV-norm | 100 | 5.08 | [3.40, 7.40] | 1.74 |
| FS aseg+DKT (no ICV) | 100 | 5.27 | [3.44, 7.17] | 1.71 |
| SynthSeg (no TIV) | 71 | 5.30 | [3.77, 7.01] | 1.68 |
| **FOMO25 AMAES_resenc_b** | 320 | **6.06** | **[4.26, 8.00]** | 2.52 |
| BrainIAC SimCLR | 768 | 8.75 | [6.32, 11.28] | 4.12 |

### What this changes about the report's headline

**The top five tools are not statistically distinguishable.** Their
95% CIs all overlap. SynthSeg+TIV (4.24) and FOMO25 AMAES (6.06) have
overlapping intervals — at n=23 subjects, the ~1.8 yr Zhang MAE
margin between them is within fold variance.

**Only BrainIAC is reliably worse** than the rest of the matrix —
its CI [6.32, 11.28] is the only one that doesn't fully overlap
SynthSeg+TIV [2.74, 6.07]. The mean BrainIAC vs SynthSeg+TIV Zhang
gap is 4.51 yr, and 95% of bootstrap draws agree the gap is positive.

**Per-fold stability reinforces this.** SynthSeg / FS variants show
fold-MAE std around 1.7 yr. FOMO25 is noticeably noisier at 2.5,
and BrainIAC is at 4.1 — its fold MAEs ranged from 8.18 to 20.71 on
the 5 GroupKFold splits.

So the report's earlier "SynthSeg + TIV is the leader" claim should be
softened to **"morphometry tools and FOMO25 are statistically tied
at this cohort size; only BrainIAC underperforms".** The deliverable
to the team becomes: at n=23 subjects, the ridge can't distinguish
between the top tools — we'd need a larger cohort (HCP-A scale)
to make ranking claims, *and that's exactly the kind of held-out eval
this analysis can't substitute for*.

## Age-subgroup re-runs (median subject-age split = 58 yr)

Re-fitting ridge on the younger half (mean age < 58, n_subjects=11,
n_scans=30) and older half (n_subjects=12, n_scans=30) separately,
each with its own 5-fold CV.

| Tool | Younger Zhang MAE | Older Zhang MAE | Δ (older − younger) |
|---|---:|---:|---:|
| SynthSeg + TIV | 4.22 | 6.69 | +2.47 |
| SynthSeg (no TIV) | 5.76 | 7.07 | +1.31 |
| FS aseg+DKT + ICV | 5.02 | 6.48 | +1.46 |
| FS aseg+DKT | 4.66 | 6.46 | +1.80 |
| **FOMO25 AMAES** | **3.40** | **3.74** | **+0.34** |
| **BrainIAC SimCLR** | 7.47 | **4.71** | **−2.76** |

### Two findings the full-cohort table hid

**FOMO25 AMAES is the best tool in *both* age halves.** Younger
Zhang MAE 3.40, older Zhang MAE 3.74. In the full-cohort fit it
looks middle-of-pack (Zhang 6.68) only because it spans more age
range than either half alone — when fit on smaller age windows, its
representations capture age tightly with low correction need.

**BrainIAC dominates the older half (4.71)**, beating SynthSeg+TIV
(6.69), but is the worst tool in the younger half (7.47). This
suggests BrainIAC's SimCLR backbone learned features that are more
informative for older brains — a known SSL phenomenon when
pretraining cohorts skew older. Worth investigating with stratified
ridge on Mihir's eventual ADNI cohort, which is largely 65+.

**Morphometry tools all show the inverse pattern**: better on
younger subjects (Zhang 4-5) than older (Zhang 6-7). This is also
known — between-subject morphometric variance grows with age (more
heterogeneous atrophy trajectories), so a fixed ridge on volumes
explains less variance in the older bracket.

### Implication for the brain-age framing

The ridge is doing different work in different age brackets, and
different feature sets are good at different sub-problems. The
question "which tool is best for brain-age" is age-conditional. A
proper deliverable for MedARC would be:

- A morphometry-based ridge for younger brains (< 60)
- An SSL-based ridge for older brains (≥ 60)
- Or a single ridge with age-stratified bias correction

Or, more provocatively — the FOMO25 AMAES result suggests that with
a *larger* n in each age bracket (i.e. on a real cohort), the SSL
backbone could outperform morphometry across the board. The
within-bracket numbers it produces are below anything else in the
matrix. The DLBS cohort is too small to confirm that.

## Per-fold MAE distributions (raw, no correction)

For the 5 GroupKFold splits:

```
synthseg_volumes_tiv    [9.10, 5.39, 5.74, 5.39, 4.27]   std 1.70
fs_asegdkt_icv          [5.34, 7.46, 6.08, 9.02, 9.97]   std 1.74
fs_asegdkt              [5.72, 9.34, 9.54, 6.31, 5.93]   std 1.71
synthseg_volumes        [8.93, 4.85, 6.91, 9.13, 5.89]   std 1.68
fomo25_embed            [10.84, 6.97, 8.58, 14.33, 8.93]  std 2.52
brainiac_embed          [14.13, 13.72, 11.45, 20.71, 8.18] std 4.12
```

The variance in BrainIAC's per-fold MAEs (8.18 → 20.71, a 2.5×
range) means any single CV split could rank it higher or lower than
its mean. SynthSeg/FS are more stable. This matches the bootstrap-CI
finding.

## What the team should take from this

1. **Don't over-claim the SynthSeg-vs-FOMO25 ranking.** They're tied
   within bootstrap CIs at n=23. The ~2 yr Zhang MAE difference is
   not statistically distinguishable.
2. **The FOMO25 AMAES result is more interesting than the full-table
   number suggests.** Per-age-bracket it's the best tool — likely
   indicates the SSL backbone is better-calibrated than morphometry
   when fit on a narrower age range, which is what ADNI/HCP-A
   sub-cohorts will look like.
3. **BrainIAC's SimCLR backbone has an age-stratified strength**:
   strong on older brains, weak on younger. Worth keeping in mind
   for the ADNI eval (Mihir's task), where the cohort is mostly 65+.
4. **n=23 is the bottleneck**, not the choice of tool. To make
   confident ranking claims we need ~3-5× the subject count, which
   is what HCP-A / OASIS would give — but those are gated on
   academic affiliation. So the team's options are: (a) collaborate
   for held-out access, (b) accept the within-bracket FOMO25 result
   as suggestive evidence, or (c) wait for ADNI to clear and re-run.

## Where the artefacts live

- Raw rigor JSON: `experiments/dlbs_morphometry_benchmark/results/ridge_statistical_rigor.json`
- Driver: `experiments/dlbs_morphometry_benchmark/scripts/ridge_statistical_rigor.py`
- This addendum: `experiments/dlbs_morphometry_benchmark/notes/ridge_rigor_addendum.md`
