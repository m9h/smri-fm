"""Parse FastSurfer stats across all subjects/sessions → single parquet table.

Reads `aseg+DKT.VINN.stats` and `aseg.VINN.stats` from every
`$FS_ROOT/<sub>_<ses>/stats/` under the FastSurfer derivatives tree and emits
one long-format parquet with columns:

    subject, session, acq, run, tool, region, volume_mm3, n_voxels

Used by `fit_ridge_baseline.py` and the concordance notebooks.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


STAT_FILES = ("aseg+DKT.VINN.stats", "aseg.VINN.stats", "cerebellum.CerebNet.stats")

# Pull sub/ses/acq/run out of FastSurfer SID directory names like
# `sub-1003_ses-wave1`. The T1 run details aren't in the SID — they land in the
# log file. We keep them blank and fill from the sidecar later.
SID_RE = re.compile(r"^(?P<sub>sub-[A-Za-z0-9]+)(?:_(?P<ses>ses-[A-Za-z0-9]+))?$")


def parse_stats(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # columns: Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "region": parts[4],
                    "volume_mm3": float(parts[3]),
                    "n_voxels": int(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


def walk_fastsurfer(root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for sid_dir in sorted(root.iterdir()):
        if not sid_dir.is_dir():
            continue
        m = SID_RE.match(sid_dir.name)
        if not m:
            continue
        sub, ses = m.group("sub"), m.group("ses") or ""
        for stat_name in STAT_FILES:
            path = sid_dir / "stats" / stat_name
            if not path.exists():
                continue
            tool = stat_name.replace(".stats", "")
            for row in parse_stats(path):
                row.update({"subject": sub, "session": ses, "tool": tool})
                records.append(row)
    return pd.DataFrame.from_records(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/data/datasets/smri-fm-cmp/fastsurfer/ds004856"),
        help="FastSurfer derivatives root (one level up from sub_ses dirs).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "fastsurfer_features.parquet",
    )
    args = ap.parse_args()

    df = walk_fastsurfer(args.root)
    print(f"parsed {len(df)} rows across {df['subject'].nunique()} subjects")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
