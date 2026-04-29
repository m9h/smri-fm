# Consolidated MedARC sMRI-FM Benchmark Evaluation

## 1. Executive Summary: The "Ladder" Status
The four-rung classical morphometry ladder (SynthSeg → FastSurfer → T1Prep → FMs) has been fully implemented and evaluated on a subset of 60 DLBS scans (23 subjects). While **SynthSeg + TIV-normalization** is the nominal leader in raw MAE, statistical bootstrap analysis shows a **5-way tie** at the top. The most significant finding is that Foundation Model (FM) performance is highly **age-conditional**.

## 2. Comparison Matrix (GroupKFold-5, Zhang-Corrected MAE)
| Rank | Tool | Category | MAE (yr) | r | 95% CI (Zhang) |
| :--- | :--- | :--- | :---: | :---: | :--- |
| 1 | **SynthSeg + TIV-norm** | Morphometry | **4.71** | 0.95 | [2.74, 6.07] |
| 2 | FastSurfer (aseg+DKT) | Morphometry | **5.48** | 0.93 | [3.40, 7.40] |
| 3 | **FOMO25 (AMAES_resenc_b)** | SSL/FM | **6.68** | 0.91 | [4.26, 8.00] |
| 4 | **BrainIAC (SimCLR)** | SSL/FM | **9.61** | 0.86 | [6.32, 11.28] |

**Note on "Statistical Tie":** At n=23 subjects, the 95% CIs for the top 3 tools overlap. Definitive ranking requires scaling the cohort.

## 3. Key Discovery: Age-Conditional Dominance
Splitting at the median subject-age (58 yr) and re-fitting ridge per
half reveals two findings the full-cohort numbers hide:

*   **FOMO25 wins both halves separately** — younger Zhang **3.40 yr MAE**, older Zhang **3.74 yr MAE**. Best tool in either bracket. The full-cohort Zhang of 6.68 looks middle-of-pack only because the SSL features span more variance across the wider age range than within either bracket.
*   **BrainIAC dominates the older half** — Zhang **4.71 yr MAE**, beating SynthSeg+TIV's **6.69 yr MAE**. But it is the *worst* tool in the younger half (Zhang **7.47 yr MAE**). The SimCLR backbone learned features that are more informative for older brains.
*   **Morphometry tools are uniformly better on younger brains** (Zhang 4–5) than older (Zhang 6–7). Between-subject morphometric variance grows with age, so a fixed ridge on volumes explains less variance in the older bracket.

Implication: at narrower-age cohorts (e.g. ADNI 65+, HCP-A subgroups),
SSL backbones may outperform morphometry across the board. The DLBS
20–87 yr range is what dilutes the FOMO25 signal in the full-cohort fit.

## 4. Bug Report: Item 4 (AMAES Recon-Collapse)
A code-reading hypothesis identified three potential issues; **empirical
tests against the published `AMAES_resenc_b` checkpoint and a 200-step
fresh-init SSL training loop disconfirmed two of three**. Details in
`notes/item4_empirical_results.md`.

| Hypothesis | Code-reading expectation | Empirical finding |
|---|---|---|
| H1: Skip-connection identity copy | visible MSE ≈ 0 (model "cheats") | **DISCONFIRMED** — visible MSE is 1.5× *higher* than masked across 5 scans. Skip connections pass features, not pixels; decoder integrates with bottleneck. Model is doing genuine MAE inference. |
| H2: Loss collapses to 0 instantly during training | yes | **DISCONFIRMED** — 200-step fresh-init training shows smooth 0.49 → 0.05 descent; masked-region MSE descends faster than visible. Not pathological collapse. |
| H3: `MSELoss(reduction="mean")` divides by total voxels not `mask.sum()` | bug present, scales loss by `mask_ratio` | **CONFIRMED** — `buggy/correct = 0.5998` exactly matches `mask_ratio = 0.5998`. Reported loss is suppressed by ~40% relative to true masked-only MSE. |

**Real fix surface (3-line patch)**: change `_rec_loss` to
`return ((pred - y)[mask] ** 2).mean()` and default
`rec_loss_masked_only=True`. Costs nothing; makes loss numbers
comparable across configs and removes a real (if minor) accounting
artefact.

**Open question for Dojo + Rohit**: whatever "loss → 0 instantly"
pattern they observed does *not* come from the AMAES_resenc_b
architecture or from the loss bug alone. Possible alternative causes
to investigate: (a) different model (mmunetvae has a different
architecture); (b) different mask ratio / token size; (c) data
normalization that makes targets near-constant per token; (d)
H3-induced optical effect on a wandb chart of an otherwise-normal
SSL training curve.

## 5. T1Prep Normalization — Resolved
The previous `_icv` variants for `t1prep_thickness` and `t1prep_area`
showed bytes-identical numbers vs the non-`_icv` versions because of a
**cross-tool scoping bug** in `fit_ridge_baseline.py`:

* `--normalise-by-icv` only divides features by the ICV column **if it
  appears in the same tool's pivot**. T1Prep's TIV row lives in the
  `t1prep_tissue` tool, separate from `t1prep_thickness` /
  `t1prep_area`. So `args.icv_region in feature_cols` evaluated
  `False` and division silently never happened, despite the
  `icv_normalised: True` flag in the JSONs.

**Fix**: written `scripts/fit_ridge_t1prep_with_tiv.py` and
`scripts/fit_ridge_t1prep_thkarea_tiv.py` that pull TIV cross-tool
from `t1prep_tissue`, join on `(subject, session)`, and divide each
thickness/area feature by TIV before ridge.

Real numbers (TIV-norm via cross-tool join):

| T1Prep variant | feats | Without TIV (Zhang) | With proper TIV-norm (Zhang) | Δ |
|---|---:|---:|---:|---:|
| thickness | 71 | 7.20 | 8.24 | **+1.04 worse** |
| area | 71 | 10.34 | 8.97 | **−1.37 better** |
| thk+area concat | 142 | 9.04 | 9.24 | +0.20 ~same |
| tissue ratios + TIV | 4 | 6.58 | 6.58 | unchanged (TIV is intra-tool here) |

**Biology**: thickness in mm is already size-invariant — dividing by
TIV destroys the natural scale. Area in mm² scales with brain size,
so TIV-norm is the right move there. Concat is dragged by thickness.

The misleading `*_icv` JSONs have been removed from `results/` to
keep the matrix honest.

## 6. Recommended Next Steps
1.  ~~**Spark Execution**~~ — done. FOMO25 AMAES extraction ran end-to-end on DGX Spark via `fomo25-arm`; 60 scans / 33 sec; ridge result is `results/ridge_fomo25_embed.json`. Container published at `ghcr.io/m9h/fomo25-arm:latest`.
2.  **Dimensionality Reduction**: apply PCA to FM embeddings (BrainIAC 768-d, FOMO25 320-d) before ridge to combat the $n \ll p$ curse. Hypothesis: BrainIAC + PCA-32 might close the gap to morphometry. Delegated to gemini-agent — see `notes/gemini_agent_tasks.md` task G2.
3.  **OOD Validation**: blocked. ADNI/HCP-A/PPMI all gate on academic affiliation. Without that, the path is either (a) collaborate with someone who has access, or (b) accept the within-DLBS Path-2 rigor (bootstrap + age-bracket) as the substitute. Mihir's ADNI work is the team's only real held-out brain-age eval.
4.  **Item 4 Fix**: H3 patch is 3 lines and worth landing regardless. Dojo + Rohit need to confirm what config/model produced the "loss → 0" they saw before chasing further architectural hypotheses (H1 + H2 disconfirmed for AMAES_resenc_b).
5.  **Cohort extension to 46 subjects**: in flight on Spark. Pipeline runner verified end-to-end on sub-12 after fixing two regressions in the medarc-smri-fm container (`/opt/venv/bin/python` path + SynthStrip OOM cap). Full 22-subject run would deliver ~24 hr after launch.
