"""Prepare FOMO26 Task-3 (REGR002 BrainAge) inputs for the DLBS ridge harness.

Re-aims the brain-age ridge benchmark (built for DLBS) onto FOMO26 Task 3 by
emitting everything the existing harness/extractors expect, using the convention
that every Task-3 scan is treated as `sub-XXX / ses-wave1` with its age in the
`AgeMRI_W1` column. That lets `fit_ridge_baseline.py` and the FM/morphometry
extractors run UNMODIFIED.

Outputs (under <out>):
  participants.tsv      participant_id, AgeMRI_W1   (tab-separated; harness ages)
  test_subjects.json    the 50 TEST_80_10_10 subject IDs (Nima's held-out split)
  folds.json            the 5 official CV folds as {train:[sub...], val:[sub...]}
  input_csv.csv         pat_id column (sub-XXX_ses-wave1) for the FM extractors
  t3_flat/              symlink farm sub-XXX_ses-wave1.nii.gz -> raw t1w.nii.gz

Usage:
  python fomo26_prepare.py --out <dir> [--limit N]   # --limit for a smoke subset
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

LABELS = Path("/data/datasets/fomo26/raw_labels/REGR002_FOMO26_BrainAge/labels.csv")
RAW_ROOT = Path("/data/datasets/fomo26/source/Task_3/Task_3/preprocessed")
SPLIT_DIR = Path("/data/datasets/fomo26/processed/REGR002_FOMO26_BrainAge")

SUB_RE = re.compile(r"(sub-[A-Za-z0-9]+)")


def subj_of(path: str) -> str:
    m = SUB_RE.search(path)
    if not m:
        raise ValueError(f"no sub- token in {path!r}")
    return m.group(1)


def load_labels() -> dict[str, float]:
    """labels.csv is headerless `path,age`; key by subject id."""
    ages: dict[str, float] = {}
    with LABELS.open() as f:
        for path, age in csv.reader(f):
            ages[subj_of(path)] = float(age)
    return ages


def split_subject_lists() -> tuple[list[str], list[dict[str, list[str]]]]:
    folds_raw = json.loads((SPLIT_DIR / "split_80_10_10.json").read_text())
    test_raw = json.loads((SPLIT_DIR / "TEST_80_10_10.json").read_text())
    test = sorted({subj_of(p) for p in test_raw})
    folds = [
        {"train": sorted({subj_of(p) for p in fold["train"]}),
         "val": sorted({subj_of(p) for p in fold["val"]})}
        for fold in folds_raw
    ]
    return test, folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="only first N subjects (smoke test); 0 = all")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ages = load_labels()
    subjects = sorted(ages)
    if args.limit:
        subjects = subjects[: args.limit]
    print(f"labels: {len(ages)} subjects; using {len(subjects)}")

    # participants.tsv — age in AgeMRI_W1 so session 'ses-wave1' joins cleanly
    pt = args.out / "participants.tsv"
    with pt.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["participant_id", "AgeMRI_W1"])
        for s in subjects:
            w.writerow([s, ages[s]])
    print(f"wrote {pt}")

    # splits
    test, folds = split_subject_lists()
    (args.out / "test_subjects.json").write_text(json.dumps(test, indent=2))
    (args.out / "folds.json").write_text(json.dumps(folds, indent=2))
    print(f"wrote test_subjects.json ({len(test)}), folds.json ({len(folds)} folds)")

    # symlink farm + input_csv  (pat_id = sub-XXX_ses-wave1)
    flat = args.out / "t3_flat"
    flat.mkdir(exist_ok=True)
    pat_ids, missing = [], []
    for s in subjects:
        raw = RAW_ROOT / s / "ses-01" / "t1w.nii.gz"
        if not raw.exists():
            missing.append(s)
            continue
        pid = f"{s}_ses-wave1"
        link = flat / f"{pid}.nii.gz"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(raw)
        pat_ids.append(pid)
    csv_path = args.out / "input_csv.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pat_id"])
        for pid in pat_ids:
            w.writerow([pid])
    print(f"wrote {csv_path} ({len(pat_ids)} pat_ids); symlinks in {flat}")
    if missing:
        print(f"WARNING: {len(missing)} subjects missing raw t1w: {missing[:5]}...")

    # sanity: every used subject has an age and a symlink
    n_join = len(set(pat_ids) & {f"{s}_ses-wave1" for s in subjects})
    print(f"JOIN CHECK: {n_join} subjects with both age and image "
          f"(expect {len(subjects) - len(missing)})")


if __name__ == "__main__":
    main()
