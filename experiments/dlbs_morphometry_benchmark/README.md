# DLBS multi-pipeline morphometry benchmark

Experiment contributor: Morgan Hough (@m9h), branch
`dlbs-morphometry-benchmark` on fork `m9h/smri-fm`.

## Goal

Build a **four-rung classical morphometry ladder** on the Dallas
Lifespan Brain Study (DLBS, `ds004856`) as the baseline against which
any sMRI-FM must be evaluated. Each rung adds exactly one axis of
feature information that the previous rung lacks, so the FM's value is
pinned down to *"what does it add beyond rung k?"*.

## Why this is needed

1. **Connor's project requirement:** *"any FM must be compared with
   simple brain morphology baselines. Features like total gray matter
   volume are highly correlated with age"*
   (Discord #neuro-fm 2026-04-10).
2. **Schulz, Siegel & Ritter 2025 (*PLOS Biology*):** simpler models
   (Ridge, 1.4k params) have **higher disease-detection sensitivity**
   than complex models (CNN 46M, Swin 10M) even though they lose at
   age MAE. Reframes the entire FM value proposition away from
   age-MAE-maximisation.
3. **Nima's preliminary baseline** on 100 DLBS images (SynthSeg + 71
   ICV-normalised features + ridge): **MAE 6.66 yr, R² 0.779,
   r 0.883, bias −0.10 yr**. No bias correction applied — Mihir flagged
   this.

## The four rungs

| Rung | Features | Tool | Status |
|---|---|---|---|
| **1** | SynthSeg regional volumes (71 ICV-normalised features) | `smri-fm/preprocessing/pipeline.py` | Nima done for 100 subjects |
| **2** | + **N4 upstream** of SynthSeg (tests Connor's 2026-04-13 proposal) | pipeline + N4 | planned |
| **3** | SynthSeg **→ FastSurfer VINN** (higher-quality seg network, same feature space) | `fastsurfer-grace` | 1 subject ✓ |
| **4** | + **cortical thickness** (CAT-Surface) + **nonlinear-MNI Jacobian** (T1Prep) | `t1prep-grace` | 1 subject ✓ |

Each rung is evaluated under four bias-correction regimes:
raw / Cole-Smith / Beheshti / Zhang age-level. Reported metrics: age
MAE **and** patient-vs-control effect size per the Schulz 2025 recipe.

## FM arms to benchmark in parallel

| FM | Preprocessing | Status |
|---|---|---|
| BrainIAC (AIM-KannLab, *Nat. Neurosci.* 2026) | their HD-BET + N4 + rigid-MNI → 128³ | Nima reports **predictions concentrate in narrow younger band on DLBS** — domain-shift finding |
| FOMO25 `mmunetvae` baseline (Gordaliza 2026, arxiv 2601.13166, **won MICCAI 2025 SSL3D + FOMO25**) | skull-strip → RAS → 1 mm iso → z-norm → bbox-crop → 96³ patches | Docker `jbanusco/sslmmunetave:1.0.0` pulling |
| Dojo's FOMO-default retrain | same as FOMO25 | need checkpoint URL |
| AnatCL (Barbano 2024, OpenBHB, 2.61 yr) | OpenBHB-style | weights public at `EIDOSLAB/AnatCL` |
| MedARC `smri-fm` (forthcoming) | project's own pipeline | waiting on release |

## Hardware

- **DGX Spark (arm64, GB10 Blackwell)**: FastSurfer (seg_only),
  T1Prep, FOMO25 mmunetvae (if arm-compatible).
  `fastsurfer-grace` and `t1prep-grace` built locally from
  `nvcr.io/nvidia/pytorch:2{4.12,6.03}-py3`.
- **Legion workstation (x86, FreeSurfer 8.2.0 podman)**: FreeSurfer
  recon-all reference, CAT12, smriPrep, SAMSEG.
- User maintains a fedora COPR with FreeSurfer-arm64 RPMs (name
  pending) that will unblock the MedARC pipeline natively on Spark.

## Directory layout

```
experiments/dlbs_morphometry_benchmark/
  README.md             # this file
  scripts/
    extract_fastsurfer_features.py   # aseg+DKT.stats → parquet
    extract_t1prep_features.py       # T1Prep morphometric outputs → parquet
    extract_synthseg_features.py     # re-parse Nima's 71-feature vector
    compute_fm_embeddings.py         # FOMO25/BrainIAC embedding extraction
    fit_ridge_baseline.py            # ridge per rung with bias-correction ablation
  notebooks/
    01_concordance_fastsurfer_vs_t1prep.ipynb
    02_bethlehem_metrics_across_rungs.ipynb
    03_fm_vs_rung_ladder.ipynb
    04_longitudinal_stability_vidalpineiro.ipynb
  slurm/                 # DGX Spark sbatch wrappers
  results/               # parquet feature tables, metrics tables
```

## Links

- Project Notion: <https://www.notion.so/Structural-MRI-foundation-model-33c82b7aafbc80debda9cc64203a0095>
- Upstream repo: <https://github.com/MedARC-AI/smri-fm>
- DLBS on OpenNeuro: <https://openneuro.org/datasets/ds004856>
- Related work write-up (this author):
  <https://github.com/ChristianGaser/T1Prep/tree/smri-fm-dlbs-comparison/comparison>
  (branch on a T1Prep fork; holds the longer-form paper draft and
  method notes that motivated this experiment).
