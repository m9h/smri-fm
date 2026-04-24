"""Augment feature parquets with a `total intracranial` region per scan.

Teaches fit_ridge_baseline.py's --normalise-by-icv path to work on:

- FastSurfer: parses `# Measure Mask, MaskVol, Mask Volume, <float>, mm^3`
  from the aseg+DKT.VINN.stats and aseg.VINN.stats files. That value is
  the SynthStripped-volume total mask — close enough to TIV for ridge
  normalisation purposes and what FastSurfer's seg_only pipeline produces
  without the full surface-based eTIV estimator.
- T1Prep: the t1prep_tissue tool already has a `TIV` region; we pivot
  it into the t1prep_thickness, t1prep_area, t1prep_volume parquets
  under the canonical name `total intracranial` so the same ridge flag
  works across arms.

Rewrites the input parquets in place with the new rows appended.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

FS_ROOT = Path("/data/datasets/smri-fm-cmp/fastsurfer/ds004856")
MASK_RE = re.compile(r"Measure Mask,\s*MaskVol,.*,\s*([0-9.]+),\s*mm\^3")
SID_RE = re.compile(r"^(sub-[A-Za-z0-9]+)_(ses-[A-Za-z0-9]+)$")


def fastsurfer_tiv_rows() -> list[dict]:
    """Walk FS derivatives and pull Mask volume per (subject, session, tool)."""
    rows: list[dict] = []
    for sid_dir in sorted(FS_ROOT.iterdir()):
        if not sid_dir.is_dir():
            continue
        m = SID_RE.match(sid_dir.name)
        if not m:
            continue
        subject, session = m.group(1), m.group(2)
        for stat_name in ("aseg+DKT.VINN.stats", "aseg.VINN.stats"):
            stat_path = sid_dir / "stats" / stat_name
            if not stat_path.exists():
                continue
            text = stat_path.read_text()
            mm = MASK_RE.search(text)
            if not mm:
                continue
            tool = stat_name.replace(".stats", "")
            rows.append({
                "subject": subject,
                "session": session,
                "tool": tool,
                "region": "total intracranial",
                "volume_mm3": float(mm.group(1)),
                "n_voxels": float(mm.group(1)),
            })
    return rows


def augment_fastsurfer(parquet: Path) -> None:
    df = pd.read_parquet(parquet)
    # Drop any pre-existing "total intracranial" rows to keep the script idempotent
    before = len(df)
    df = df[df["region"] != "total intracranial"]
    tiv = pd.DataFrame(fastsurfer_tiv_rows())
    print(f"  fastsurfer: dropping {before - len(df)} stale TIV rows, adding {len(tiv)} fresh")
    out = pd.concat([df, tiv], ignore_index=True, sort=False)
    out.to_parquet(parquet, index=False)
    print(f"  wrote {parquet} ({len(out)} rows)")


def augment_t1prep(parquet: Path) -> None:
    df = pd.read_parquet(parquet)
    # Source: t1prep_tissue's TIV region per (subject, session)
    tiv = df[(df["tool"] == "t1prep_tissue") & (df["region"] == "TIV")][
        ["subject", "session", "volume_mm3"]
    ].copy()
    print(f"  t1prep: found TIV for {len(tiv)} scans in t1prep_tissue")
    # Drop any pre-existing stale canonical TIV rows
    df = df[df["region"] != "total intracranial"]
    # For each target tool, emit a canonical `total intracranial` row per scan.
    # thickness/area tools store thickness/area in different cols; we only
    # ever read `volume_mm3` for the ICV-norm divisor, so adding the TIV
    # there is consistent regardless of the tool's primary value column.
    new_rows = []
    for tool in ("t1prep_thickness", "t1prep_area", "t1prep_volume", "t1prep_tissue"):
        if tool not in df["tool"].unique():
            continue
        for _, row in tiv.iterrows():
            new_rows.append({
                "subject": row["subject"],
                "session": row["session"],
                "acq": "",
                "run": "",
                "tool": tool,
                "region": "total intracranial",
                "volume_mm3": float(row["volume_mm3"]),
                "n_voxels": float("nan"),
                "thickness_mm": float("nan"),
                "area_mm2": float("nan"),
            })
    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    print(f"  added {len(new_rows)} total intracranial rows across tools")
    out.to_parquet(parquet, index=False)
    print(f"  wrote {parquet} ({len(out)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fastsurfer",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results" / "fastsurfer_features.parquet",
    )
    ap.add_argument(
        "--t1prep",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results" / "t1prep_features.parquet",
    )
    args = ap.parse_args()

    if args.fastsurfer.exists():
        print(f">>> augmenting FastSurfer parquet at {args.fastsurfer}")
        augment_fastsurfer(args.fastsurfer)
    if args.t1prep.exists():
        print(f">>> augmenting T1Prep parquet at {args.t1prep}")
        augment_t1prep(args.t1prep)


if __name__ == "__main__":
    main()
