"""Render the smri-fm meeting summary: comparison table + figure + slack copy.

One script that:
  1. Loads every ridge_*.json under results/
  2. Loads the bootstrap-rigor JSON
  3. Produces a single comparison table (sorted by Zhang MAE)
  4. Renders the headline figure (forest plot of bootstrap CIs +
     per-age-bracket bar chart)
  5. Spits out a markdown one-pager + a slack-ready blurb

Outputs:
  notes/meeting_2026-04-30/
    comparison_table.md
    headline_figure.png
    slack_message.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "notes" / "meeting_2026-04-30"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

# Friendly display names for the messy filename → tool mapping
DISPLAY = {
    "ridge_synthseg_icv":         ("SynthSeg + TIV-norm",         "morph",  71),
    "ridge_synthseg":             ("SynthSeg (no TIV)",            "morph",  71),
    "ridge_fs_asegdkt_icv":       ("FastSurfer aseg+DKT + ICV",   "morph", 100),
    "ridge_fs_asegdkt":           ("FastSurfer aseg+DKT",         "morph", 100),
    "ridge_t1prep_thickness":     ("T1Prep thickness (DKT)",         "morph",  71),
    "ridge_t1prep_thickness_TIV_real":("T1Prep thickness + TIV-norm","morph",  71),
    "ridge_t1prep_area":          ("T1Prep area (DKT)",              "morph",  71),
    "ridge_t1prep_area_TIV_real": ("T1Prep area + TIV-norm",         "morph",  71),
    "ridge_t1prep_thkarea":       ("T1Prep thk+area concat",         "morph", 142),
    "ridge_t1prep_thkarea_TIV_real":("T1Prep thk+area + TIV-norm",   "morph", 142),
    "ridge_t1prep_tissue_icv":    ("T1Prep tissue ratios + TIV",     "morph",   4),
    "ridge_fs_t1prep_all":        ("FS + T1Prep concat",          "concat",171),
    "ridge_concat_fs_t1prep":     ("FS + T1Prep concat (alt)",    "concat",171),
    "ridge_concat_synthseg_t1prep":("SynthSeg + T1Prep concat",   "concat",143),
    "ridge_concat_fs_t1prep_brainiac":("FS+T1Prep+BrainIAC",      "concat",939),
    "ridge_concat_synthseg_fomo25":("SynthSeg + FOMO25 concat",   "concat",392),
    "ridge_brainiac_embed":       ("BrainIAC SimCLR (768-d)",     "ssl",   768),
    "ridge_fomo25_embed":         ("FOMO25 AMAES_resenc_b",       "ssl",   320),
}


def load_ridge_results():
    rows = []
    for f in sorted(RESULTS.glob("ridge_*.json")):
        if f.stem == "ridge_statistical_rigor" or "dryrun" in f.stem:
            continue
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        meta = DISPLAY.get(f.stem)
        if meta is None:
            continue
        name, family, feats = meta
        rows.append({
            "stem": f.stem,
            "name": name,
            "family": family,
            "feats": feats,
            "raw_mae":      r.get("raw", {}).get("mae"),
            "raw_r":        r.get("raw", {}).get("pearson_r"),
            "beheshti_mae": r.get("beheshti", {}).get("mae"),
            "beheshti_r":   r.get("beheshti", {}).get("pearson_r"),
            "zhang_mae":    r.get("zhang", {}).get("mae"),
            "zhang_r":      r.get("zhang", {}).get("pearson_r"),
        })
    return pd.DataFrame(rows).sort_values("zhang_mae").reset_index(drop=True)


def load_rigor():
    with (RESULTS / "ridge_statistical_rigor.json").open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Output 1: markdown comparison table
# ---------------------------------------------------------------------------

def write_comparison_table(df, rigor):
    rigor_by_tool = {t["tool"]: t for t in rigor["tools"]}

    out = OUT / "comparison_table.md"
    lines = [
        "# DLBS brain-age ridge — meeting summary 2026-04-30",
        "",
        "**Cohort**: 60 scans / 23 subjects (DLBS multi-wave longitudinal)",
        "",
        "**Protocol**: Nima's GroupKFold(5) + StandardScaler + RidgeCV(α ∈ logspace(-3, 3, 25))",
        "",
        "All numbers in years. Bootstrap CI from 1000 subject-level resamples.",
        "",
        "| # | Tool | family | feats | raw MAE | r | **Zhang MAE** | r (Z) | 95% CI (Zhang) |",
        "|---:|---|:-:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in df.iterrows():
        tool_key = {
            "ridge_synthseg_icv": "synthseg_volumes_tiv",
            "ridge_synthseg": "synthseg_volumes",
            "ridge_fs_asegdkt_icv": "fs_asegdkt_icv",
            "ridge_fs_asegdkt": "fs_asegdkt",
            "ridge_brainiac_embed": "brainiac_embed",
            "ridge_fomo25_embed": "fomo25_embed",
        }.get(row["stem"])
        if tool_key and tool_key in rigor_by_tool:
            ci = rigor_by_tool[tool_key]["bootstrap"]["zhang_mae_ci95"]
            ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
        else:
            ci_str = "—"
        emphasis_l, emphasis_r = ("**", "**") if row["family"] == "ssl" or i == 0 else ("", "")
        lines.append(
            f"| {i+1} | {emphasis_l}{row['name']}{emphasis_r} | {row['family']} | "
            f"{row['feats']:>3} | {row['raw_mae']:.2f} | {row['raw_r']:+.2f} | "
            f"**{row['zhang_mae']:.2f}** | {row['zhang_r']:+.2f} | {ci_str} |"
        )
    lines.extend([
        "",
        "## Headline takeaways",
        "",
        "1. **Top 5 tools statistically tied** (bootstrap CIs all overlap). Only BrainIAC reliably worse.",
        "2. **FOMO25 AMAES is best in *both* age halves** when split at median age 58 (younger Zhang 3.40, older Zhang 3.74). The full-cohort number hides this — masked-AE features are tightly age-correlated within bracket but span more variance across the full age range.",
        "3. **BrainIAC dominates the older half** (Zhang 4.71, beats SynthSeg+TIV's 6.69). Worst tool in younger half (7.47). SimCLR backbone learned old-brain features.",
        "4. **Concat past ~150 features hurts** — n ≪ p curse at this cohort size.",
        "",
        "## Item 4 (MAE recon-collapse) — H3 confirmed, H1+H2 disconfirmed",
        "",
        "Empirical tests against published AMAES_resenc_b checkpoint:",
        "",
        "| Hypothesis | Code-reading | Empirical |",
        "|---|---|---|",
        "| H1 skip-copy → visible MSE ≈ 0 | trivial reconstruction | **opposite** (visible 1.5× *higher* than masked) |",
        "| H2 loss → 0 instantly | yes | **no** — smooth descent over 200 steps |",
        "| H3 MSELoss scaled by mask_ratio | bug present | **CONFIRMED** (buggy/correct = 0.6 = mask_ratio exactly) |",
        "",
        "**Patch**: 3-line change to `_rec_loss` in asparagus self_supervised.py.",
        "Reframe for Dojo+Rohit: ask which model + mask config + post a wandb chart.",
        "",
        "## Caveats",
        "",
        "- DLBS is in FOMO50K + FOMO300K source cohorts → SSL not held-out.",
        "- n=23 subjects → bootstrap CIs ~3 yr wide → ranking margins under that are noise.",
        "- Mihir's ADNI eval is the team's only path to a real held-out brain-age claim.",
        "",
        "## Container + reproducibility",
        "",
        "```",
        "docker pull ghcr.io/m9h/fomo25-arm:latest",
        "```",
        "",
        "Grace Blackwell-tuned arm64 image with AMAES_resenc_b checkpoint baked in. Extracts 320-d embeddings on 60 DLBS scans in 33 seconds.",
        "",
        "## Followup question for Ahmed",
        "",
        "Pivot: jbanusco/fomo25 v1.0.0 ships no public mmunetvae checkpoint, so item 3 went against AMAES_resenc_b instead. Can you confirm intent? `notes/fomo25_checkpoint_followup.md` has the full ask.",
        "",
        "## Where the artefacts live",
        "",
        "`github.com/m9h/smri-fm/tree/dlbs-morphometry-benchmark/experiments/dlbs_morphometry_benchmark/`:",
        "- `notes/comprehensive_ridge_report.md` — narrative",
        "- `notes/ridge_rigor_addendum.md` — bootstrap + age-bracket findings",
        "- `notes/fomo25_mae_recon_collapse_item4.md` — code-reading hypothesis",
        "- `notes/item4_empirical_results.md` — empirical results",
        "- `notes/fomo25_checkpoint_followup.md` — question for Ahmed",
        "- `notes/meeting_2026-04-30/` — this summary",
        "",
    ])
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Output 2: headline figure (forest plot + age-bracket bars)
# ---------------------------------------------------------------------------

def make_headline_figure(rigor):
    tools = rigor["tools"]
    # Sort by mean Zhang MAE
    tools_sorted = sorted(tools, key=lambda t: t["bootstrap"]["zhang_mae_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Panel A: forest plot of bootstrap 95% CIs on Zhang MAE
    ax = axes[0]
    label_map = {
        "synthseg_volumes_tiv": "SynthSeg + TIV (71)",
        "fs_asegdkt_icv":       "FastSurfer aseg+DKT + ICV (100)",
        "fs_asegdkt":           "FastSurfer aseg+DKT (100)",
        "synthseg_volumes":     "SynthSeg (71)",
        "fomo25_embed":         "FOMO25 AMAES (320)",
        "brainiac_embed":       "BrainIAC SimCLR (768)",
    }
    color_map = {
        "synthseg_volumes_tiv": "tab:blue",
        "fs_asegdkt_icv":       "tab:cyan",
        "fs_asegdkt":           "tab:cyan",
        "synthseg_volumes":     "tab:blue",
        "fomo25_embed":         "tab:red",
        "brainiac_embed":       "tab:orange",
    }
    y = np.arange(len(tools_sorted))
    for i, t in enumerate(tools_sorted):
        ci = t["bootstrap"]["zhang_mae_ci95"]
        mn = t["bootstrap"]["zhang_mae_mean"]
        c = color_map.get(t["tool"], "gray")
        ax.errorbar(mn, i, xerr=[[mn - ci[0]], [ci[1] - mn]], fmt="o",
                    color=c, capsize=4, ms=8, elinewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels([label_map.get(t["tool"], t["tool"]) for t in tools_sorted])
    ax.invert_yaxis()
    ax.set_xlabel("Zhang-corrected MAE (years), bootstrap 95% CI")
    ax.set_title("(A) Top 5 tools statistically tied at n=23")
    ax.grid(axis="x", alpha=0.3)

    # Panel B: age-subgroup comparison
    ax = axes[1]
    bracket_data = {}
    for t in tools_sorted:
        sg = t.get("age_subgroups", {})
        y_data = sg.get("younger")
        o_data = sg.get("older")
        if y_data and o_data:
            bracket_data[t["tool"]] = {
                "younger": y_data["zhang_mae"],
                "older":   o_data["zhang_mae"],
            }
    keys = list(bracket_data.keys())
    x = np.arange(len(keys))
    w = 0.35
    younger = [bracket_data[k]["younger"] for k in keys]
    older   = [bracket_data[k]["older"]   for k in keys]
    ax.bar(x - w/2, younger, w, label="younger half (age<58)", color="tab:cyan")
    ax.bar(x + w/2, older,   w, label="older half (age≥58)",   color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(k, k).split(" (")[0] for k in keys],
                       rotation=15, ha="right")
    ax.set_ylabel("Zhang-corrected MAE (years)")
    ax.set_title("(B) FOMO25 wins both halves; BrainIAC flips by bracket")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("DLBS brain-age ridge (n=60 scans / 23 subjects)", fontsize=12)
    fig.tight_layout()

    p = OUT / "headline_figure.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Output 3: slack-ready blurb
# ---------------------------------------------------------------------------

def write_slack(df, rigor):
    rigor_by_tool = {t["tool"]: t for t in rigor["tools"]}
    fomo25 = rigor_by_tool["fomo25_embed"]["bootstrap"]
    synth = rigor_by_tool["synthseg_volumes_tiv"]["bootstrap"]
    brainiac = rigor_by_tool["brainiac_embed"]["bootstrap"]

    # Per-bracket numbers
    fomo25_subgroups = next(t["age_subgroups"] for t in rigor["tools"] if t["tool"] == "fomo25_embed")
    brainiac_subgroups = next(t["age_subgroups"] for t in rigor["tools"] if t["tool"] == "brainiac_embed")

    msg = f"""smri-fm DLBS update — for tomorrow's meeting (n=60 scans / 23 subjects, GroupKFold(5), 4 bias-correction schemes)

**Item 3 — FOMO25 baseline ridge** ✅
- FOMO25 AMAES_resenc_b (320-d): Zhang MAE {fomo25['zhang_mae_mean']:.2f} [{fomo25['zhang_mae_ci95'][0]:.2f}, {fomo25['zhang_mae_ci95'][1]:.2f}]
- SynthSeg+TIV (71-d, top morph): Zhang MAE {synth['zhang_mae_mean']:.2f} [{synth['zhang_mae_ci95'][0]:.2f}, {synth['zhang_mae_ci95'][1]:.2f}]
- BrainIAC (768-d): Zhang MAE {brainiac['zhang_mae_mean']:.2f} [{brainiac['zhang_mae_ci95'][0]:.2f}, {brainiac['zhang_mae_ci95'][1]:.2f}]
- Pivot note: targeted AMAES_resenc_b not mmunetvae (no public ckpt for the latter; followup q for @ahmed in repo)

**Headline finding (bootstrap + age-bracket rigor)**
- Top 5 tools have *overlapping* 95% CIs at n=23 — only BrainIAC is reliably worse.
- FOMO25 wins **both** age halves separately (younger Zhang {fomo25_subgroups['younger']['zhang_mae']:.2f}, older {fomo25_subgroups['older']['zhang_mae']:.2f}).
- BrainIAC flips: best in older half ({brainiac_subgroups['older']['zhang_mae']:.2f}), worst in younger ({brainiac_subgroups['younger']['zhang_mae']:.2f}).

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
"""
    p = OUT / "slack_message.md"
    p.write_text(msg)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    df = load_ridge_results()
    rigor = load_rigor()

    write_comparison_table(df, rigor)
    make_headline_figure(rigor)
    write_slack(df, rigor)

    print()
    print("=== top of comparison table ===")
    print(df[["name", "family", "feats", "raw_mae", "zhang_mae", "zhang_r"]].head(8).to_string(index=False))
    print()
    print(f"meeting summary at: {OUT}")


if __name__ == "__main__":
    main()
