# Memorization-audit calibration on DLBS (real data)

Calibration of `scripts/audit_memorization.py` (3D volume-vs-volume Pearson,
generalized from Akbar/Wang/Eklund 2024) on **real** DLBS T1w before we trust it
to gate synthetic data into FOMO26 finetune sets. No synthetic data was needed —
DLBS is longitudinal, so a subject's own later scans act as a real
"near-duplicate" positive control.

## Setup

- Corpus: `/data/datasets/smri-fm-cmp/brainiac-preproc/ds004856`, 129 volumes,
  uniform `(170, 206, 162)`, HD-BET skull-stripped + rigid-MNI (73.6% of voxels
  exactly zero). 46 subjects, 44 with >=2 waves.
- Split: query = each subject's `wave1`; corpus = all `wave2`/`wave3`. A query
  subject's own later waves therefore sit in the corpus as true near-duplicates.
- Full 129x129 leave-self-out pairwise run used for the same-vs-different
  separation table.

## Finding: the paper's 0.93 threshold does NOT transfer to registered sMRI

The shared zero background of skull-stripped + registered volumes inflates
inter-subject Pearson correlation. After mean-centering, the identical zero
region becomes a shared constant across every volume.

| metric | full-volume | brain-masked (union nonzero) |
|---|---|---|
| different-subject floor (max) | 0.97 | **0.90** |
| different-subject frac >0.93 | **39% (false positives)** | **0%** |
| same-subject top1 retrieval | 74% | 71% |
| same-subject best-corr (mean) | 0.94 | 0.85 |

Brain-masking (restrict Pearson to voxels nonzero in >=1 real volume, here 40%
of the grid) drops the distinct-subject ceiling to ~0.90, so 0.93 cleanly
separates "novel brain" from "exact regurgitation" (corr -> 1.0). The audit
machinery itself is sound: top1 retrieval is the same subject ~71-74% of the
time, i.e. Pearson does pull a person's other scan out of the corpus.

Same- vs different-subject distributions still overlap (expected — longitudinal
scans of one person are not exact duplicates), so a single global threshold
cannot label "same person". That is fine: a memorization audit targets corr->1.0
regurgitation, which now sits well above the 0.90 real-brain ceiling.

## Recipe for FOMO26 synthetic-data gating

1. Always run with `--brain-mask` on skull-stripped/registered sMRI.
2. Set the flag threshold from the empirical real-vs-real ceiling per dataset
   (DLBS brain-masked: distinct-subject p95 0.88, max 0.90), not the 2D-slice
   0.93 constant. Treat any synthetic above the real max as a memorization flag.

## Artifacts

- `results/audit_dlbs_calibration_brainmasked.csv` — per-query top-5 matches,
  brain-masked, wave1-vs-later-waves split.

## Repro

```
python scripts/audit_memorization.py \
  --synthetic <wave1 dir> --real <wave2/3 dir> \
  --out results/audit_dlbs_calibration_brainmasked.csv \
  --topk 5 --brain-mask
```
