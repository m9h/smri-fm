smri-fm DLBS update — for tomorrow's meeting (n=60 scans / 23 subjects, GroupKFold(5), 4 bias-correction schemes)

**Item 3 — FOMO25 baseline ridge** ✅
- FOMO25 AMAES_resenc_b (320-d): Zhang MAE 6.06 [4.26, 8.00]
- SynthSeg+TIV (71-d, top morph): Zhang MAE 4.24 [2.74, 6.07]
- BrainIAC (768-d): Zhang MAE 8.75 [6.32, 11.28]
- Pivot note: targeted AMAES_resenc_b not mmunetvae (no public ckpt for the latter; followup q for @ahmed in repo)

**Headline finding (bootstrap + age-bracket rigor)**
- Top 5 tools have *overlapping* 95% CIs at n=23 — only BrainIAC is reliably worse.
- FOMO25 wins **both** age halves separately (younger Zhang 3.40, older 3.74).
- BrainIAC flips: best in older half (4.71), worst in younger (7.47).

**Item 4 — MAE recon-collapse** (for @Dojo @Rohit)
Empirical tests inside fomo25-arm container against the published checkpoint + 200-step fresh SSL training:
- H3 (MSELoss scaled by mask_ratio): **CONFIRMED** — buggy/correct = 0.5998 vs mask_ratio = 0.5998 exact match. 3-line patch in asparagus self_supervised.py fixes it.
- H1 (skip-copy → visible MSE ≈ 0) and H2 (instant loss collapse): **disconfirmed** for AMAES_resenc_b. Visible MSE is actually *higher* than masked (1.5×), training descends smoothly. Whatever you observed is config- or model-specific — happy to re-run on mmunetvae or your config if useful.

**Container** ready for the team:
```
docker pull ghcr.io/m9h/fomo25-arm:latest
```
Grace Blackwell arm64 / NGC torch / AMAES_resenc_b ckpt baked in.

**All artefacts**: github.com/m9h/smri-fm @ dlbs-morphometry-benchmark
- `notes/comprehensive_ridge_report.md` + `notes/ridge_rigor_addendum.md` (full matrix)
- `notes/item4_empirical_results.md` (item 4 honest write-up)
- `notes/meeting_2026-04-30/headline_figure.png` (forest plot + bracket comparison)

**Caveat to flag**: DLBS is in FOMO50K + FOMO300K source cohorts; this is a within-pretraining-data eval. Mihir's ADNI is the path to held-out.
