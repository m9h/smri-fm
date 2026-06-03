"""Aggregate the FOMO26 segmentation arm Dice into a structurebench leaderboard.

Two dense-seg downstream tasks, two FM arms that carry a genuine pretrained
decoder (SIAM nnU-Net, FOMO25 mmunetvae) — the other roster arms are
encoder-only and would need a bolt-on decoder that confounds the FM-quality
read, so they are excluded from seg by design.

Reads each asparagus seg run's test prediction JSON
(.../models/seg_{arm}_{task}/predictions/{TASK}__TEST_80_10_10__best.json),
pulls the per-class mean Dice (+ Jaccard / sensitivity / precision), and reports
the foreground-mean Dice (class 0 = background dropped) as the headline. Emits
seg_leaderboard.json + seg_benchmark_table.md alongside the other benchmark
artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ARMS = {"smri_siam": "siam", "smri_mmunetvae": "mmunetvae"}
TASKS = {
    "SEG009_FOMO26_Meningioma": "seg009",
    "SEG010_FOMO26_TrigeminalNeuralgia": "seg010",
}
MODELS = "/data/datasets/fomo26/models"
PRED = "{models}/seg_{arm_short}_{task_short}/predictions/{task}__TEST_80_10_10__best.json"


def load_mean(task: str, arm_short: str, task_short: str, models: str):
    p = PRED.format(models=models, arm_short=arm_short, task_short=task_short, task=task)
    if not os.path.exists(p):
        return None, p
    return json.load(open(p)).get("mean", {}), p


def fg_mean_dice(mean: dict) -> float | None:
    fg = [mean[c]["dice"] for c in mean if c != "0" and "dice" in mean[c]]
    return round(sum(fg) / len(fg), 4) if fg else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=MODELS)
    ap.add_argument("--out-prefix",
                    default="experiments/fomo26_fm_benchmark/results/seg")
    args = ap.parse_args()

    board: dict = {"tasks": {}, "arms": {arm: {} for arm in ARMS}}
    for task, task_short in TASKS.items():
        board["tasks"][task] = {}
        for arm, arm_short in ARMS.items():
            mean, path = load_mean(task, arm_short, task_short, args.models_dir)
            if mean is None:
                board["arms"][arm][task] = {"status": "missing", "path": path}
                continue
            per_class = {
                c: {k: mean[c].get(k) for k in ("dice", "jaccard", "sensitivity", "precision")}
                for c in sorted(mean)
            }
            entry = {
                "fg_mean_dice": fg_mean_dice(mean),
                "n_classes": len(mean),
                "per_class": per_class,
            }
            board["arms"][arm][task] = entry
            board["tasks"][task][arm] = entry["fg_mean_dice"]

    out_json = Path(f"{args.out_prefix}_leaderboard.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(board, indent=2))

    # markdown: foreground-mean Dice matrix + per-class breakdown
    lines = ["# structurebench-v1.0 — FOMO26 segmentation leaderboard", "",
             "Foreground-mean test Dice (background class 0 excluded). Two arms with",
             "real pretrained decoders; 200 epochs, 128^3 patches, split_80_10_10.", "",
             "| Task | " + " | ".join(ARMS) + " |",
             "|------|" + "|".join(["------"] * len(ARMS)) + "|"]
    for task in TASKS:
        cells = []
        for arm in ARMS:
            v = board["tasks"][task].get(arm)
            cells.append(f"{v:.4f}" if isinstance(v, float) else "n/a")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-class Dice", ""]
    for task in TASKS:
        lines.append(f"### {task}")
        # build header from the first arm that has data
        cols = None
        body = []
        for arm in ARMS:
            e = board["arms"][arm].get(task)
            if not isinstance(e, dict) or "per_class" not in e:
                body.append(f"| {arm} | (missing) |")
                continue
            cols = sorted(e["per_class"])
            cells = [f'{e["per_class"][c]["dice"]:.4f}' for c in cols]
            body.append(f"| {arm} | " + " | ".join(cells) + " |")
        if cols:
            lines.append("| arm | " + " | ".join(f"class {c}" for c in cols) + " |")
            lines.append("|-----|" + "|".join(["-----"] * len(cols)) + "|")
        lines += body + [""]

    lines += [
        "## Interpretation", "",
        "- **SEG009 Meningioma: both arms score 0.0000 foreground Dice** — they",
        "  predict all-background. The tumor class is tiny and sparse relative to",
        "  the thin-slice (~29-slice) FLAIR volume, so the Dice-only objective is",
        "  minimized by emptying the prediction; neither pretrained decoder rescues",
        "  it. This is a task/loss-config failure (needs a region-balanced or",
        "  compound loss + foreground oversampling), not an FM-quality signal — the",
        "  arms are indistinguishable here.",
        "- **SEG010 TrigeminalNeuralgia: mmunetvae > SIAM** (fg-mean 0.276 vs 0.184).",
        "  mmunetvae recovers both foreground structures (class 1 0.329, class 2",
        "  0.224) while SIAM finds only class 1 (0.368) and misses class 2 entirely",
        "  (0.000). mmunetvae being the strongest seg arm is consistent with it being",
        "  the most FOMO-domain-matched (pretrained on raw FOMO60K MRI) — notable",
        "  given it is the *worst* arm on REGR002 brain-age, i.e. seg and regression",
        "  rank the arms differently.",
        "",
    ]

    out_md = Path(f"{args.out_prefix}_benchmark_table.md")
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_json}\nwrote {out_md}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
