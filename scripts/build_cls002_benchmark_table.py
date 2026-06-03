"""Aggregate the CLS002 Infarct classification arm into a structurebench table.

Reads the pooled-CV leaderboard emitted by the Modal CLS002 sweep
(cls002_leaderboard.json: {baseline_majority, arms: {smri_<arm>: {DWI|FULL:
{accuracy, balanced_accuracy, f1, ...}}}}) and renders a markdown leaderboard
mirroring the seg/RG tables. Two input cohorts per arm: DWI (the single
diffusion channel, the modality the FOMO300K corpus is plurality-dominated by)
and FULL (all available channels). Headline metric is accuracy vs the majority
baseline (predict-all-positive). Arms missing from the leaderboard (e.g.
brainiac, whose sweep produced no result) render as n/a rather than being
dropped, so the roster stays explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Full structurebench roster, in the canonical reporting order. Arms absent from
# the leaderboard json show as n/a (brainiac's CLS002 run yielded no result).
ARMS = [
    "smri_siam", "smri_simclr3d", "smri_mmunetvae",
    "smri_anatcl", "smri_fomo60k", "smri_triad", "smri_brainiac",
]
COHORTS = ["DWI", "FULL"]
METRICS = ["accuracy", "balanced_accuracy", "f1"]


def cell(board: dict, arm: str, cohort: str, metric: str) -> str:
    v = board.get("arms", {}).get(arm, {}).get(cohort, {}).get(metric)
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--leaderboard",
        default="experiments/fomo26_fm_benchmark/results/cls002_leaderboard.json",
    )
    ap.add_argument(
        "--out",
        default="experiments/fomo26_fm_benchmark/results/cls002_benchmark_table.md",
    )
    args = ap.parse_args()

    board = json.loads(Path(args.leaderboard).read_text())
    baseline = board.get("baseline_majority")
    n = next(
        (c.get("n") for a in board.get("arms", {}).values()
         for c in a.values() if isinstance(c, dict) and "n" in c),
        None,
    )

    header_cols = [f"{coh}_{m}" for coh in COHORTS for m in METRICS]
    lines = [
        "# structurebench-v1.0 — FOMO26 CLS002 Infarct leaderboard", "",
        f"Pooled-CV classification metrics (n={n}). Majority (predict-all-positive)"
        f" baseline accuracy = {baseline}. DWI = single diffusion channel; FULL =",
        "all available channels. Accuracy is the headline; bold beats baseline.", "",
        "| arm | " + " | ".join(header_cols) + " |",
        "|-----|" + "|".join(["-----"] * len(header_cols)) + "|",
    ]
    for arm in ARMS:
        cells = []
        for coh in COHORTS:
            for m in METRICS:
                c = cell(board, arm, coh, m)
                # bold any accuracy that strictly beats the majority baseline
                if (m == "accuracy" and c != "n/a" and baseline is not None
                        and float(c) > baseline):
                    c = f"**{c}**"
                cells.append(c)
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += [
        "", "## Interpretation", "",
        "- **No arm meaningfully separates from the majority prior.** The best",
        "  FULL-input accuracies (siam, simclr3d 0.619) only *tie* the 0.619",
        "  baseline; only siam and simclr3d clear it on DWI-only (0.667). High",
        "  recall at baseline-level accuracy means the better arms mostly predict",
        "  the positive class — balanced accuracy (~0.50-0.61) confirms weak true",
        "  discrimination on this small (n={n}) infarct cohort.".format(n=n),
        "- **DWI >= FULL for the top arms** (siam 0.667 vs 0.619; simclr3d 0.667 vs",
        "  0.619), consistent with infarct being a diffusion-salient finding and",
        "  with FOMO300K's diffusion-plurality pretraining corpus.",
        "- **brainiac is absent** (CLS002 sweep produced no result; same gap as its",
        "  REGR002 row). Reported n/a rather than excluded — a re-run would be",
        "  needed to place it.",
        "",
    ]

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print(f"wrote {out}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
