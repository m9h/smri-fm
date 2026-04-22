"""Parse T1Prep per-session outputs → single parquet.

T1Prep ships:
  mri/p0<stem>.nii.gz          # partial-volume tissue labelmap (CAT12)
  mri/{mwp1,mwp2}<stem>.nii.gz # modulated warped GM/WM (VBM)
  surf/{lh,rh}.thickness.<stem>   # FreeSurfer curv per-vertex thickness
  surf/{lh,rh}.area.<stem>        # per-vertex surface area
  surf/{lh,rh}.aparc_DK40.freesurfer.<stem>.annot   # DK40 parcellation

We emit long-format rows:

    subject, session, tool, region, volume_mm3, n_voxels, thickness_mm, area_mm2

with `tool` in {t1prep_tissue, t1prep_thickness, t1prep_area}.

Intended as the rung-4 feature source for the DLBS morphometry benchmark;
pairs with extract_fastsurfer_features.py.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as nfs
import numpy as np
import pandas as pd


SID_RE = re.compile(r"^(?P<sub>sub-[A-Za-z0-9]+)(?:_(?P<ses>ses-[A-Za-z0-9]+))?$")

# CAT12 p0 label convention: CSF ~1, GM ~2, WM ~3 (partial-volume, continuous).
# Binarise with >0.5 thresholds and count voxels.
P0_CLASSES = {"CSF": (0.5, 1.5), "GM": (1.5, 2.5), "WM": (2.5, 3.5)}


def voxel_volume(img: nib.Nifti1Image) -> float:
    zooms = img.header.get_zooms()[:3]
    return float(np.prod(zooms))


def parse_tissue(p0_path: Path) -> list[dict]:
    img = nib.load(p0_path)
    data = np.asanyarray(img.dataobj)
    vv = voxel_volume(img)
    rows = []
    for name, (lo, hi) in P0_CLASSES.items():
        mask = (data > lo) & (data <= hi)
        n = int(mask.sum())
        rows.append({"region": name, "volume_mm3": n * vv, "n_voxels": n})
    total = int((data > 0.5).sum())
    rows.append({"region": "TIV", "volume_mm3": total * vv, "n_voxels": total})
    return rows


def parse_surface_per_region(
    annot_path: Path,
    scalar_path: Path,
    hemi: str,
    reducer: str,
) -> list[dict]:
    labels, _ctab, names = nfs.read_annot(annot_path)
    scalar = nfs.read_morph_data(scalar_path)
    rows = []
    for idx, bname in enumerate(names):
        if isinstance(bname, bytes):
            raw = bname.decode("utf-8", errors="replace")
        else:
            raw = str(bname)
        # CAT12-written annot tables pack multiple names into a single name
        # field and don't always null-terminate cleanly. DK40 region names are
        # one token of lowercase alphanumerics + hyphen; split on the first
        # non-name char to recover the intended region name.
        import re as _re
        m2 = _re.match(r"[a-zA-Z0-9_\-]+", raw.lstrip("\x00"))
        region = m2.group(0) if m2 else raw.split("\x00", 1)[0]
        mask = labels == idx
        if mask.sum() == 0:
            continue
        vals = scalar[mask]
        if reducer == "mean":
            rows.append(
                {"region": f"ctx-{hemi}-{region}", "thickness_mm": float(vals.mean())}
            )
        else:
            rows.append(
                {"region": f"ctx-{hemi}-{region}", "area_mm2": float(vals.sum())}
            )
    return rows


def find_stem(mri_dir: Path) -> str | None:
    for p in mri_dir.glob("p0*.nii.gz"):
        return p.name[len("p0") : -len(".nii.gz")]
    return None


def walk_t1prep(root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for sid_dir in sorted(root.iterdir()):
        if not sid_dir.is_dir():
            continue
        m = SID_RE.match(sid_dir.name)
        if not m:
            continue
        sub, ses = m.group("sub"), m.group("ses") or ""
        mri = sid_dir / "mri"
        surf = sid_dir / "surf"
        if not mri.is_dir():
            continue

        stem = find_stem(mri)
        if stem is None:
            continue

        # Tissue volumes from p0
        for row in parse_tissue(mri / f"p0{stem}.nii.gz"):
            row.update({"subject": sub, "session": ses, "tool": "t1prep_tissue"})
            records.append(row)

        if not surf.is_dir():
            continue

        # DK40 thickness + area per hemisphere
        for hemi in ("lh", "rh"):
            annot = surf / f"{hemi}.aparc_DK40.freesurfer.{stem}.annot"
            thick = surf / f"{hemi}.thickness.{stem}"
            area = surf / f"{hemi}.area.{stem}"
            if annot.exists() and thick.exists():
                for row in parse_surface_per_region(annot, thick, hemi, "mean"):
                    row.update(
                        {"subject": sub, "session": ses, "tool": "t1prep_thickness"}
                    )
                    records.append(row)
            if annot.exists() and area.exists():
                for row in parse_surface_per_region(annot, area, hemi, "sum"):
                    row.update(
                        {"subject": sub, "session": ses, "tool": "t1prep_area"}
                    )
                    records.append(row)

    return pd.DataFrame.from_records(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/data/datasets/smri-fm-cmp/t1prep/ds004856"),
        help="T1Prep derivatives root (one level up from sub_ses dirs).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "t1prep_features.parquet",
    )
    args = ap.parse_args()

    df = walk_t1prep(args.root)
    print(f"parsed {len(df)} rows across {df['subject'].nunique()} subjects")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
