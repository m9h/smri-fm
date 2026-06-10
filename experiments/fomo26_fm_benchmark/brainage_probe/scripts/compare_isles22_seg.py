"""Aggregate the ISLES22 stroke-lesion seg 5-fold sweep into an FM-vs-scratch
comparison: per-arm lesion Dice across folds (mean±std) + the pretraining gain.

Handles the fold-0 path quirks: siam fold-0 is the predict-recovery dir (the
original train run crashed at test), mmunetvae/scratch fold-0 are normal; folds
1-4 live in seg_<tag>_fold<N>. Reports whatever folds are present so it's useful
mid-sweep too.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

M = Path("/data/datasets/fomo26/models")
TASK = "SEG011_ISLES22_IschStroke"
PREDNAME = f"{TASK}__TEST_80_10_10__best.json"

# arm -> {fold: rundir}; fold-0 special cases noted inline
ARMS = {
    "siam (SIAM-pretrained)": {
        0: "seg_siam_seg011_isles22_ischstroke_predict",          # recovered
        **{f: f"seg_siam_seg011_isles22_ischstroke_fold{f}" for f in (1, 2, 3, 4)},
    },
    "mmunetvae (FOMO25-pretrained)": {
        0: "seg_mmunetvae_seg011_isles22_ischstroke",
        **{f: f"seg_mmunetvae_seg011_isles22_ischstroke_fold{f}" for f in (1, 2, 3, 4)},
    },
    "scratch_nnunet (random init)": {
        f: f"seg_scratch_nnunet_seg011_isles22_ischstroke_fold{f}" for f in range(5)
    },
}


def lesion_dice(rundir: str) -> float | None:
    p = M / rundir / "predictions" / PREDNAME
    if not p.exists():
        return None
    return json.loads(p.read_text())["mean"]["1"]["dice"]


def main() -> None:
    arm_folds = {}
    print(f"{'arm':<32}{'folds':<10}{'per-fold lesion Dice':<40}{'mean±std':<14}")
    print("-" * 96)
    for arm, folds in ARMS.items():
        vals = {f: lesion_dice(rd) for f, rd in folds.items()}
        got = {f: v for f, v in vals.items() if v is not None}
        arm_folds[arm] = got
        per = " ".join(f"{f}:{v:.3f}" for f, v in sorted(got.items()))
        ms = f"{st.mean(got.values()):.4f}±{(st.pstdev(got.values()) if len(got) > 1 else 0):.4f}" if got else "—"
        print(f"{arm:<32}{len(got):<10}{per:<40}{ms:<14}")
    print("-" * 96)

    sc = arm_folds.get("scratch_nnunet (random init)", {})
    if sc:
        sc_mean = st.mean(sc.values())
        print(f"\nPretraining gain over scratch (Δ mean lesion Dice):")
        for arm, got in arm_folds.items():
            if "scratch" in arm or not got:
                continue
            # gain on the folds both arms have, for a paired estimate
            shared = sorted(set(got) & set(sc))
            if shared:
                gains = [got[f] - sc[f] for f in shared]
                print(f"  {arm:<32} {st.mean(gains):+.4f}  (paired over folds {shared})")
            else:
                print(f"  {arm:<32} {st.mean(got.values()) - sc_mean:+.4f}  (unpaired)")
    print("\nReference — FOMO260K paper (20-shot, 3-modality, AMAES-ResEnc): AMAES 0.740 vs scratch 0.7286 (+0.011).")
    print("Note: our protocol = 200-case full finetune, 2-modality (DWI+ADC), SIAM-nnU-Net topology — not directly comparable to the paper's few-shot numbers.")


if __name__ == "__main__":
    main()
