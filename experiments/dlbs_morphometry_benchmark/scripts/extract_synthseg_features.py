"""Collect MedARC pipeline.py's SynthSeg volumes.tsv files into a long-format
parquet matching extract_fastsurfer_features.py's schema.

Schema: subject, session, acq, run, tool, region, volume_mm3

Reads *_desc-synthseg_volumes.tsv beneath <medarc-smri-fm>/ds004856/<sub>/synthseg/
Each TSV is two columns: `region, volume_mm3`. Files are named like
  sub-1003_ses-wave1_acq-MPRAGE_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w_desc-synthseg_volumes.tsv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

FILENAME_RE = re.compile(
    r"(?P<subject>sub-[^_]+)_"
    r"(?P<session>ses-[^_]+)_"
    r"(?P<acq>acq-[^_]+)_"
    r"(?P<run>run-\d+)_"
    r".*_desc-synthseg_volumes\.tsv$"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--synthseg-root",
        type=Path,
        default=Path("/data/datasets/smri-fm-cmp/medarc-smri-fm/ds004856"),
    )
    ap.add_argument(
        "--tool-name",
        default="synthseg_volumes",
        help='Tag written to the `tool` column of the parquet. '
             'Matches fit_ridge_baseline.py --tool filter.',
    )
    ap.add_argument(
        "--modality",
        default="T1w",
        help='Restrict to a single modality, e.g. T1w (skip T2w/FLAIR).',
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    tsvs = sorted(args.synthseg_root.rglob("*_desc-synthseg_volumes.tsv"))
    print(f"found {len(tsvs)} synthseg volumes TSVs under {args.synthseg_root}")

    kept = 0
    skipped = 0
    for tsv in tsvs:
        if args.modality and f"_{args.modality}_" not in tsv.name:
            skipped += 1
            continue
        m = FILENAME_RE.match(tsv.name)
        if m is None:
            print(f"skip (unmatched filename): {tsv.name}")
            skipped += 1
            continue
        subject = m.group("subject")
        session = m.group("session")
        acq = m.group("acq")
        run = m.group("run")

        df = pd.read_csv(tsv, sep="\t")
        # Expected columns: region, volume_mm3
        if "region" not in df.columns or "volume_mm3" not in df.columns:
            print(f"skip (schema): {tsv.name} has columns {df.columns.tolist()}")
            skipped += 1
            continue

        for _, r in df.iterrows():
            rows.append({
                "subject": subject,
                "session": session,
                "acq": acq,
                "run": run,
                "tool": args.tool_name,
                "region": str(r["region"]),
                "volume_mm3": float(r["volume_mm3"]),
            })
        kept += 1

    if not rows:
        raise SystemExit("no SynthSeg volumes rows collected")

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(
        f"wrote {args.out} "
        f"({len(out):,} rows, {kept} scans kept, {skipped} files skipped)"
    )


if __name__ == "__main__":
    main()
