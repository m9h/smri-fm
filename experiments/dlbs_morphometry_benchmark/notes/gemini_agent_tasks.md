# Tasks for gemini-agent — pre-meeting delegation 2026-04-29

Three things gemini-agent could deliver before tomorrow's smri-fm
meeting that complement the work already in flight. None require new
data access; all use the existing artefacts in this repo.

## Task G1 — Update `consolidated_agent_synthesis.md` with the corrections

The synthesis you wrote captures the work cleanly but four claims are
out of date based on findings landed in commits `b2b17c5` and
`ce1d69c` after you wrote it:

| Section | Stale | Correction |
|---|---|---|
| §3 | "FOMO25 leader in Early/Mid-Life (3.40)" | FOMO25 is best in **both** age halves separately (younger Zhang 3.40, older Zhang 3.74). The full-cohort number hides this because the SSL features span more variance across the wider age range. |
| §4 #1 | "Skip-Connection Leakage: model cheats via skips on unmasked voxels" | **Disconfirmed empirically.** Empirical tests in `notes/item4_empirical_results.md` show visible MSE is 1.5× *higher* than masked — opposite of skip-copy. The model is doing genuine MAE inference. |
| §4 #2 | Loss scaling bug "0.6×" | ✅ correct — confirmed empirically (`buggy/correct = 0.5998` = mask_ratio exactly). Keep this one. |
| §5 | "T1Prep Normalization Mystery — empty TIV columns or mapping error" | **Resolved.** Cross-tool issue: TIV lives in `t1prep_tissue` not in `t1prep_thickness`/`t1prep_area`, so the existing `--normalise-by-icv` flag silently no-op'd. Written cross-tool-join script (`fit_ridge_t1prep_with_tiv.py`). Real numbers: thickness +1.04 yr Zhang (worse), area −1.37 yr Zhang (**better**), concat ~same. |

Re-write §3, §4 #1, §5 with the corrections; keep §2, §6, and the
overall structure. This is the polished synthesis the team will read
before the meeting, so accuracy matters.

## Task G2 — PCA ablation on the SSL embeddings

The current matrix shows the n ≪ p curse: BrainIAC's 768-d is the
worst SSL row, and concat past ~150 features degrades. Question for
the team: **does dimensionality reduction before ridge close the gap?**

Write `scripts/ridge_with_pca.py` that:

1. Loads `results/{brainiac_embeddings,fomo25_embeddings}.parquet`
2. Pivots each to wide form with subjects × features
3. For each n_components in `[8, 16, 32, 64, 128, 256]` (capped at the
   actual feature count):
   - sklearn `Pipeline([StandardScaler, PCA(n_components=k), StandardScaler, RidgeCV])`
   - GroupKFold(5) by subject, same as the rest of the matrix
   - Report raw + Zhang MAE + r
4. Output a 2-panel figure showing MAE vs n_components for both
   backbones, plus a JSON table for the comparison matrix.

Strict acceptance: results integrate cleanly into the existing comparison
table format (same JSON schema as `ridge_brainiac_embed.json`). Drop
new rows under `results/ridge_{brainiac,fomo25}_pca{N}.json` so my
`make_meeting_summary.py` picks them up after a `DISPLAY` map update.

Hypothesis to test: at n=23, the BrainIAC backbone with PCA-32 might
match or beat its raw 768-d ridge. If true, that's a real finding —
"the SSL backbones contain useful signal but the dimensionality
reduction matters at this cohort size."

## Task G3 — Draft a 5-minute meeting walkthrough script

The team has 1 hour. Item 3 + item 4 results are 5–10 min total.
Write `notes/meeting_2026-04-30/walkthrough_script.md` with:

1. **Opening (30 sec)**: 1 sentence framing — "FOMO25 ridge run, Item 4
   investigated, both have surprises."
2. **Item 3 result (90 sec)**: Lead with the headline figure (forest
   plot + age-bracket bars). Punchy: "top 5 tools statistically tied;
   only BrainIAC reliably worse." Then the per-age-bracket reversal
   as the surprising finding.
3. **Item 4 result (90 sec)**: Lead with "1 of 3 hypotheses survived
   empirical testing — but the one that did is a real bug worth
   patching." Mention the H1+H2 disconfirmation honestly. Hand off
   to Dojo+Rohit for what the alternative cause might be.
4. **Caveats slide (60 sec)**: DLBS-in-pretraining-data, n=23, no
   held-out path until ADNI clears.
5. **Open asks (30 sec)**:
   - Ahmed: confirm AMAES vs mmunetvae intent
   - Dojo+Rohit: what config/model produced the "loss → 0" they saw?
   - Whoever: thoughts on PCA-before-ridge for SSL backbones (depends
     on whether G2 lands)
   - Anyone with academic affiliation: ADNI/HCP-A access?

Format as bullet points with timed sections, not prose. Should be
readable at 200 wpm without losing track.

## What's *not* on you (in flight elsewhere)

- DLBS cohort extension to 46 subjects — running on Spark (hopefully)
- sub-1007 post-surface cascade — Legion's pickup
- vpjax CVR preprocessing — vpjax-agent's lane
- FSL completion for sub-1003 / sub-1007 — Spark FSL runner

## Pointers

- Repo: `github.com/m9h/smri-fm @ dlbs-morphometry-benchmark`
- Comparison matrix → `notes/comprehensive_ridge_report.md`
- Bootstrap CI rigor → `notes/ridge_rigor_addendum.md`
- Item 4 first-pass + empirical → `notes/item4_empirical_results.md`
- Meeting summary scaffolding → `notes/meeting_2026-04-30/`
- Checkpoint followup for Ahmed → `notes/fomo25_checkpoint_followup.md`

---

If gemini-agent has time and bandwidth for a fourth task, the
PR-ready item 4 patch against the actual `Sllambias/asparagus` repo
(commit + commit message + test that demonstrates the fix) would
also be valuable, but G1+G2+G3 are the priority before the meeting.
