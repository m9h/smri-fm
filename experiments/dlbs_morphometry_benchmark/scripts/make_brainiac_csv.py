"""Build the pat_id,label CSV for BrainIAC extraction from the set of
preprocessed files present under `--root_dir` and ages from DLBS's
participants.tsv. Filename stem = pat_id (e.g. sub-1003_ses-wave1).

Label is the MRI age (AgeMRI_W{1,2,3}) as a float. `pat_id` and file
naming match what BrainAgeDataset / our ExtractionDataset expect.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SES_TO_AGECOL = {
    "ses-wave1": "AgeMRI_W1",
    "ses-wave2": "AgeMRI_W2",
    "ses-wave3": "AgeMRI_W3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help="dir of BrainIAC-preproc'd .nii.gz files; stems become pat_ids",
    )
    ap.add_argument(
        "--participants",
        type=Path,
        default=Path("/data/raw/openneuro/ds004856/participants.tsv"),
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = pd.read_csv(args.participants, sep="\t")
    parts = parts.set_index("participant_id")

    rows = []
    for nii in sorted(args.root_dir.glob("*.nii.gz")):
        stem = nii.name[: -len(".nii.gz")]
        # stem = sub-XXXX_ses-waveN
        if "_ses-" not in stem:
            print(f"skip malformed stem: {stem}")
            continue
        subject, session = stem.split("_ses-", 1)
        session = "ses-" + session
        age_col = SES_TO_AGECOL.get(session)
        if age_col is None:
            print(f"unknown session: {session} ({stem})")
            continue
        if subject not in parts.index:
            print(f"subject not in participants.tsv: {subject}")
            continue
        raw = parts.loc[subject, age_col]
        try:
            age = float(raw)
        except (TypeError, ValueError):
            print(f"no age for {stem}: {raw!r}")
            continue
        rows.append({"pat_id": stem, "label": age, "dataset": "DLBS"})

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
