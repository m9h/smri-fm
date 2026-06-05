# Consolidated MedARC sMRI-FM Benchmark Evaluation

> **Updated for the v2 cohort (117 scans, 42 subjects).** The headline has
> changed since the v1 (60-scan, 23-subject) writeup: scaling the cohort
> **reverses the tool ranking** — volumetric morphometry collapses while
> surface-based and Foundation-Model features hold. The reversal has been
> verified as a real per-subject effect, not an artifact (§3).

## 1. Executive Summary: The Reversal
At v1 (23 subjects) the volumetric segmentation tools led: **SynthSeg + TIV**
was the nominal #1 (Zhang 4.71 yr) and the FM lagged (FOMO25 6.68 yr). Scaling
to v2 (42 subjects) **inverts this completely**. Volumetric tools roughly
double their error (SynthSeg 4.71 → 9.39, FastSurfer 5.48 → 9.28) while the FM
and surface-tissue features hold or improve (FOMO25 AMAES 6.68 → 5.87, T1Prep
tissue 6.58 → 5.87). In v2 the volumetric tools are at the **bottom** of the
ladder and the FM is tied for the **top**.

The mechanism is verified (§3): the 19 subjects added in v2 carry a volume→age
relationship the ridge cannot reconcile with the original 23, so pooling them
**corrupts** the volumetric fit (error on the original subjects nearly doubles,
5.85 → 9.94). The FM embeddings share a consistent age axis across both groups,
so pooling **helps** them as more data should (8.56, down from 9.90). The FM has
learned a representation in which age is linearly decodable across a
heterogeneous cohort; raw region volumes have not.

## 2. Comparison Matrix (GroupKFold-5, Zhang-Corrected MAE)

**v2 — primary (117 scans, 42 subjects):**
| Rank | Tool | Category | MAE (yr) | r | Δ vs v1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | **FOMO25 (AMAES_resenc_b)** | SSL/FM | **5.87** | 0.92 | −0.81 |
| 1 | **T1Prep tissue ratios + TIV** | Morphometry (surface/tissue) | **5.87** | 0.92 | −0.71 |
| 3 | FOMO25 (mmunetvae) | SSL/FM | 6.37 | 0.91 | *new arm* |
| 4 | T1Prep thickness | Morphometry (surface) | 6.82 | 0.90 | −0.38 |
| 5 | FastSurfer (aseg+DKT) + ICV | Morphometry (volumetric) | 9.28 | 0.82 | **+3.80** |
| 6 | SynthSeg + TIV-norm | Morphometry (volumetric) | 9.39 | 0.83 | **+4.68** |

**v1 — for contrast (60 scans, 23 subjects):** SynthSeg+TIV **4.71** (1st),
FastSurfer **5.48** (2nd), FOMO25 **6.68** (3rd), BrainIAC **9.61** (4th).
At n=23 the top-3 95% CIs overlapped (a 5-way "tie"); the v2 ranking is
separated. *BrainIAC was not recomputed for v2 (still v1-only).*

## 3. Verified Reversal: pooling robustness, not "hard subjects"
The v1→v2 swing is large enough to demand verification. It is **not** an
age-range, bias-correction, or batch artifact:

* **Not a wider age range.** v1 and v2 span the same range (22–91 yr), same mean
  (~51–54) and SD (~18–20). The new subjects did not stretch the age axis.
* **Not a Zhang-correction artifact.** Raw MAE moves the same way (SynthSeg raw
  5.99 → 10.48); the correction only rescales it.
* **The original 23 subjects still fit perfectly *alone*.** Refit on the v2 data
  restricted to the original subjects, SynthSeg recovers Zhang/raw exactly to v1
  (raw 5.85). Their features are byte-identical across v1/v2 — no reprocessing.
* **No univariate batch shift.** Between original and new subjects, 0 of 71
  TIV-normalized volume features show |standardized-mean-difference| > 0.8
  (max 0.63). The incompatibility is multivariate/joint, not a per-region scale.

**The smoking gun — what adding the 19 new subjects does to the original 23:**

| Tool | original-23 fit alone | original-23 *inside* the pooled 42-subj model | effect |
| :--- | :---: | :---: | :--- |
| SynthSeg + TIV (volumetric) | 5.85 | **9.94** | corrupted (+70%) |
| FOMO25 AMAES (SSL/FM) | 9.90 | **8.56** | improved (−14%) |
| FOMO25 mmunetvae (SSL/FM) | — | 9.64 (new-19: 8.07) | robust |

Adding data should *help*. It helps the FM and hurts volumetry — the two subject
groups occupy regions of volumetric feature space with conflicting linear
age-maps, but share a single age direction in FM embedding space.

**Open follow-up:** *why* are the 19 new subjects volumetrically incompatible?
They entered via the Spark cohort-extension pipeline; candidate causes are
acquisition/site differences or SynthSeg segmentation quality on those scans
(univariate scale is fine, so it would be a joint/shape effect). Worth a QC pass
on the new subjects' segmentations before the finding is published.

**Confound (applies to both cohorts):** DLBS is in FOMO's pretraining corpus, so
the FM's pooling-robustness is partly confounded by memorization. OOD validation
(ADNI / HCP-A) is required before claiming generalization.

### 3a. Prior v1 interpretation (superseded)
The v1 writeup explained FOMO25's middling full-cohort number via an
*age-conditional* split (median 58 yr): FOMO25 won both age halves separately
(younger 3.40, older 3.74) while morphometry was uniformly better on younger
brains. This still holds *within v1* but is no longer the lead story — the v2
per-subject mechanism above is the stronger, verified explanation. The
age-conditional split was a single-cohort observation at n=23; treat it as
hypothesis-generating, not as the v2 finding.

## 4. Bug Report: Item 4 (AMAES Recon-Collapse)
A code-reading hypothesis identified three potential issues; **empirical tests
against the published `AMAES_resenc_b` checkpoint and a 200-step fresh-init SSL
training loop disconfirmed two of three**. Details in
`notes/item4_empirical_results.md`.

| Hypothesis | Code-reading expectation | Empirical finding |
|---|---|---|
| H1: Skip-connection identity copy | visible MSE ≈ 0 (model "cheats") | **DISCONFIRMED** — visible MSE is 1.5× *higher* than masked across 5 scans. Skip connections pass features, not pixels; decoder integrates with bottleneck. Model is doing genuine MAE inference. |
| H2: Loss collapses to 0 instantly during training | yes | **DISCONFIRMED** — 200-step fresh-init training shows smooth 0.49 → 0.05 descent; masked-region MSE descends faster than visible. Not pathological collapse. |
| H3: `MSELoss(reduction="mean")` divides by total voxels not `mask.sum()` | bug present, scales loss by `mask_ratio` | **CONFIRMED** — `buggy/correct = 0.5998` exactly matches `mask_ratio = 0.5998`. Reported loss is suppressed by ~40% relative to true masked-only MSE. |

**Real fix surface (3-line patch)**: change `_rec_loss` to
`return ((pred - y)[mask] ** 2).mean()` and default `rec_loss_masked_only=True`.
Costs nothing; makes loss numbers comparable across configs and removes a real
(if minor) accounting artefact.

**Open question for Dojo + Rohit**: whatever "loss → 0 instantly" pattern they
observed does *not* come from the AMAES_resenc_b architecture or the loss bug
alone. Candidate causes: (a) a different model — mmunetvae has a different
architecture (and is now benchmarked as a v2 arm, Zhang 6.37); (b) different
mask ratio / token size; (c) data normalization making targets near-constant per
token; (d) an H3-induced optical effect on a wandb chart of an otherwise-normal
curve.

## 5. T1Prep Normalization Resolved
The "normalization mystery" was a cross-tool join issue: TIV lives in the
`t1prep_tissue` output, not in the thickness or area tables. After correctly
joining TIV (`fit_ridge_t1prep_with_tiv.py`):
*   **Cortical Area:** improved with normalization (v1 −1.37 yr Zhang; v2 area+TIV 7.98 vs area 11.91).
*   **Cortical Thickness:** slightly worse with normalization (largely head-size independent), and is one of the **reversal survivors** — held at v2 (6.82, Δ −0.38).
*   **Tissue ratios + TIV:** the strongest morphometry arm at v2 (5.87), tied with the FM. Surface/tissue morphometry, unlike raw region volumes, is pooling-robust.

## 6. Recommended Next Steps
1.  **OOD validation is now the critical path.** The reversal makes the
    memorization confound load-bearing — the FM's apparent robustness must be
    confirmed off the pretraining distribution (ADNI / HCP-A) before publication.
2.  **QC the 19 new subjects' segmentations (§3 follow-up).** Confirm whether the
    volumetric collapse is a genuine representational effect or a fixable
    SynthSeg/FastSurfer quality issue on the extended cohort.
3.  **Dimensionality reduction scales with n — drop the blanket "PCA-16" rule.**
    The v1 recommendation to "adopt PCA-16/32 for all SSL arms" was a small-n
    artifact. PCA ablation: optimum is **PCA-16 at n=23 (Zhang 5.86)** but moves
    to **PCA-128 at n=42 (Zhang 5.54)** — and at v2 PCA-16 is among the worst
    (6.80). Pick n_components per cohort size; do not hard-code 16. Full curves:
    `ridge_fomo25_pca_ablation.json` (v1), `ridge_fomo25_embed_pca_v2.json` (v2).
4.  **Recompute BrainIAC for v2** to complete the matrix (currently v1-only).
5.  **Implement the Item 4 fix** and check whether the loss correction improves
    latent feature diversity.
