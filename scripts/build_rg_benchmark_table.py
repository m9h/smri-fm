"""Combine per-arm RG/SETOL weight diagnostics with the downstream benchmark.

Joins the weights-only Martin-RG diagnostics (alpha, phi_k, M_tr, isolated trap
counts from rg_spectral_extras.py) against the two downstream leaderboards:
  - REGR002 brain-age MAE (years, lower=better)  -- from the asparagus prediction
    JSONs on disk
  - CLS002 infarct pooled-CV accuracy (n=21, majority baseline 0.619) -- from the
    Modal sweep leaderboard json

Emits a tidy per-arm table (CSV + markdown) and Spearman correlations between each
weights-only diagnostic and each downstream metric. The whole point of Martin's
2026 RG theory (see memory reference-martin-rg-learning) is that alpha alone
under-determines model quality, so we report phi_1 / M_tr / trap-count alongside
it and ask which weights-only signal best tracks the leaderboards.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["siam", "fomo60k", "anatcl", "simclr3d", "triad", "brainiac", "mmunetvae"]

REGR_PRED = ("/data/datasets/fomo26/models/regr002_{arm}/predictions/"
             "REGR002_FOMO26_BrainAge__TEST_80_10_10__best.json")


def regr002_mae(arm: str) -> float | None:
    p = REGR_PRED.format(arm=arm)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    y, yh = [], []
    for v in d.values():
        if isinstance(v, dict) and "label" in v and "prediction" in v:
            y.append(float(v["label"])); yh.append(float(v["prediction"]))
    if not y:
        return None
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yh))))


def cls002_acc(board: dict, arm: str, cohort: str) -> float | None:
    a = board.get("arms", {}).get(f"smri_{arm}", {})
    return a.get(cohort, {}).get("accuracy")


def spearman(x, y) -> float | None:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default="experiments/fomo26_fm_benchmark/results")
    ap.add_argument("--cls-leaderboard", default="/tmp/cls002_leaderboard.json")
    ap.add_argument("--out-prefix",
                    default="experiments/fomo26_fm_benchmark/results/rg_benchmark_table")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    board = json.load(open(args.cls_leaderboard)) if os.path.exists(args.cls_leaderboard) else {"arms": {}}

    rows = []
    for arm in ARMS:
        sp = rdir / f"rg_extras_{arm}_summary.json"
        rg = json.load(open(sp)) if sp.exists() else {}
        rows.append({
            "arm": arm,
            "n_layers": rg.get("n_layers"),
            "alpha_median": rg.get("alpha_median"),
            "frac_alpha_lt2": rg.get("frac_alpha_lt2"),
            "phi_1_median": rg.get("phi_1_median"),
            "M_tr_median": rg.get("M_tr_median"),
            "M_tr_frac_median": rg.get("M_tr_frac_median"),
            "n_traps_isolated_total": rg.get("n_traps_isolated_total"),
            "rand_num_spikes_total": rg.get("rand_num_spikes_total"),
            "regr002_mae": regr002_mae(arm),
            "cls002_acc_full": cls002_acc(board, arm, "FULL"),
            "cls002_acc_dwi": cls002_acc(board, arm, "DWI"),
        })
    df = pd.DataFrame(rows)

    # downstream metrics: MAE is lower=better, so flip its sign for "quality"
    diagnostics = ["alpha_median", "frac_alpha_lt2", "phi_1_median",
                   "M_tr_median", "M_tr_frac_median", "n_traps_isolated_total"]
    metrics = {
        "regr002_quality(-MAE)": -df["regr002_mae"],
        "cls002_acc_full": df["cls002_acc_full"],
        "cls002_acc_dwi": df["cls002_acc_dwi"],
    }
    corr_rows = []
    for diag in diagnostics:
        rec = {"diagnostic": diag}
        for mname, mser in metrics.items():
            rec[f"spearman_{mname}"] = spearman(df[diag], mser)
        corr_rows.append(rec)
    corr = pd.DataFrame(corr_rows)

    csv_path = Path(f"{args.out_prefix}.csv")
    corr_path = Path(f"{args.out_prefix}_spearman.csv")
    md_path = Path(f"{args.out_prefix}.md")
    df.to_csv(csv_path, index=False)
    corr.to_csv(corr_path, index=False)

    def fmt(v, nd=3):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)

    lines = ["## structurebench RG-diagnostics vs downstream benchmark", "",
             "Weights-only Martin-RG diagnostics (left) vs FOMO26 downstream (right).",
             "MAE years lower=better; CLS002 acc, majority baseline 0.619; n=21 pooled CV.", ""]
    cols = ["arm", "alpha_median", "frac_alpha_lt2", "phi_1_median", "M_tr_median",
            "n_traps_isolated_total", "regr002_mae", "cls002_acc_full", "cls002_acc_dwi"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines += ["", "### Spearman: diagnostic vs downstream metric (n arms with both)", ""]
    ccols = list(corr.columns)
    lines.append("| " + " | ".join(ccols) + " |")
    lines.append("|" + "|".join(["---"] * len(ccols)) + "|")
    for _, r in corr.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in ccols) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print(df.to_string(index=False))
    print()
    print(corr.to_string(index=False))
    print(f"\nwrote {csv_path}\nwrote {corr_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
