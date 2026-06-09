"""Assemble the FOMO26 Task-3 brain-age comparison: frozen linear-probe + the
morphometry floor, both 5-fold-CV and fixed-TEST, next to the asparagus FINETUNE
baselines Nima ran (so frozen-probe vs finetune sits side by side).

Reads results/ridge_<arm>_cv.json and results/ridge_<arm>_fixedtest.json for every
arm registered below; missing files render as "—". Reference finetune numbers are
test-split MAE from Nima (Discord, fold 0).
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).parent.parent / "results"

# arm label -> result-file stem
# Each FM is extracted at its native asparagus input size (AMAES 160^3 pretrain;
# mmunetvae 64^3; fomo60k/anatcl/triad 96^3 finetune). amaes96 is the earlier
# under-served 96^3 extraction, kept to show the preprocessing-match effect.
ARMS = [
    ("FastSurfer aseg+DKT (morphometry floor)", "fastsurfer"),
    ("AMAES resenc_b @160 native (frozen)", "amaes"),
    ("  AMAES @96 under-served (frozen)", "amaes96"),
    ("mmunetvae @64 native (frozen)", "mmunetvae"),
    ("fomo60k comb_reg @96 native (frozen)", "fomo60k_combined_regular"),
    ("anatcl @96 native (frozen)", "anatcl"),
    ("triad @96 native (frozen)", "triad"),
]

# Reference: asparagus FINETUNE (not frozen) on the same TEST split, fold 0 (Nima).
FINETUNE_REF = {
    "AMAES official (finetune)": 6.15,
    "PDF-1M (finetune)": 6.58,
}


def load(stem: str, kind: str):
    p = R / f"ridge_{stem}_{kind}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def cell(d):
    if not d:
        return "—"
    return f"{d['raw']['mae']:.2f}/{d['zhang']['mae']:.2f} (r{d['raw']['pearson_r']:+.2f})"


def main() -> None:
    print("FOMO26 Task-3 brain-age — frozen ridge probe + morphometry floor")
    print("cells = raw MAE / Zhang MAE (Pearson r);  lower MAE better\n")
    print(f"{'Arm':<42}{'5-fold CV (n=494)':<26}{'fixed TEST (n=50)':<26}")
    print("-" * 94)
    for label, stem in ARMS:
        print(f"{label:<42}{cell(load(stem,'cv')):<26}{cell(load(stem,'fixedtest')):<26}")
    print("-" * 94)
    print("Reference — asparagus FINETUNE on the same TEST split (Nima, fold 0):")
    for k, v in FINETUNE_REF.items():
        print(f"  {k:<40}TEST raw MAE {v}")
    print("\nNote: frozen linear probe is expected to trail finetune; the morphometry")
    print("floor shows whether a frozen FM beats classical aseg volumes at all.")


if __name__ == "__main__":
    main()
