"""3D volume-vs-volume memorization audit for synthetic MRI.

Generalization of the Akbar/Wang/Eklund (2024, *Mach. Learn.: Sci. Technol.*)
2D-slice Pearson-correlation audit to whole 3D volumes with arbitrary channel
counts. Drop-in for any pair of (synthetic_dir, real_dir) of NIfTI volumes.

For every synthetic volume, computes the Pearson correlation against every
volume in the real (training) corpus and reports the maximum, the matching
real volume's filename, and optionally the top-K matches. The headline number
from the Eklund paper — diffusion max-corr ≈ 0.95 vs StyleGAN ≈ 0.92 on
BraTS — should be reproducible by pointing this script at the AIDA brgandi
release and a BRATS21 training set.

Why bother:
  Standard generative-quality metrics (FID, IS) are blind to memorization. A
  perfectly-reproducing generator can score great on FID. Pairwise correlation
  is the simplest thing that does catch it. Use this before adding any
  synthetic data to FOMO26 finetune sets.

Algorithm:
  1. Flatten each (C, D, H, W) volume to a vector of length V = C*D*H*W.
  2. For the real corpus, mean-center and L2-normalize each row once.
  3. For each synthetic, mean-center + L2-normalize, then dot-product against
     the precomputed real-norm matrix in one matmul -> all Pearson coeffs.
  4. Report top-K matches per synthetic.

Usage:
    python scripts/audit_memorization.py \
        --synthetic /path/to/synthetic_nifti_dir \
        --real /path/to/real_training_nifti_dir \
        --modalities flair t1 t2 dwi  \
        --out experiments/fomo26_fm_benchmark/results/audit_<name>.csv \
        [--device cuda] [--topk 3] [--chunk 16] [--limit 1000]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


def _list_volumes(root: Path, modalities: list[str] | None) -> list[Path | tuple[Path, ...]]:
    """Return a list of volume entries.

    If `modalities` is None, every NIfTI under `root` is one entry (assumed
    single-channel or already 4-D).

    If `modalities` is a list, we look for subject-level subdirectories under
    `root` containing one file per modality whose name contains the modality
    name. Each entry is a tuple of file paths in the order modalities were
    given. This matches the BraTS / FOMO26-style on-disk layout.
    """
    if modalities is None:
        files = sorted(p for p in root.rglob("*") if p.suffix in (".nii", ".gz") and ".nii" in p.name)
        return [f for f in files]

    entries: list[tuple[Path, ...]] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tup: list[Path] = []
        ok = True
        for m in modalities:
            cands = [
                p for p in subject_dir.iterdir()
                if p.is_file() and ".nii" in p.name and m.lower() in p.name.lower()
            ]
            if len(cands) != 1:
                ok = False
                break
            tup.append(cands[0])
        if ok:
            entries.append(tuple(tup))
    return entries


def _load_flat(entry: Path | tuple[Path, ...], target_dim: tuple[int, ...] | None) -> np.ndarray:
    """Load one volume entry and return a flat float32 vector."""
    import nibabel as nib

    if isinstance(entry, tuple):
        arrs = [nib.load(str(p)).get_fdata(dtype=np.float32) for p in entry]
        vol = np.stack(arrs, axis=0)  # (C, D, H, W)
    else:
        arr = nib.load(str(entry)).get_fdata(dtype=np.float32)
        if arr.ndim == 3:
            vol = arr[np.newaxis]  # (1, D, H, W)
        else:
            vol = np.moveaxis(arr, -1, 0)  # assume channel-last -> channel-first

    if target_dim is not None and vol.shape != target_dim:
        raise ValueError(
            f"shape mismatch: got {vol.shape} expected {target_dim} for {entry!r}. "
            "All volumes must share dims; resample upstream if needed."
        )

    return vol.reshape(-1).astype(np.float32)


def _norm_rows(x: torch.Tensor) -> torch.Tensor:
    """Mean-center and L2-normalize rows. For Pearson via matmul."""
    x = x - x.mean(dim=1, keepdim=True)
    norms = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x / norms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=Path, required=True, help="dir of synthetic NIfTI volumes")
    ap.add_argument("--real", type=Path, required=True, help="dir of real (training) NIfTI volumes")
    ap.add_argument("--modalities", nargs="*", default=None,
                    help="if set, treat subdirs as subjects and match one file per modality name")
    ap.add_argument("--out", type=Path, required=True, help="CSV path for per-synthetic max-corr report")
    ap.add_argument("--topk", type=int, default=3, help="how many top matches per synthetic to record")
    ap.add_argument("--chunk", type=int, default=16, help="synthetic-batch size for the chunked matmul")
    ap.add_argument("--limit-synth", type=int, default=None, help="optional cap on number of synthetic samples")
    ap.add_argument("--limit-real", type=int, default=None, help="optional cap on real-corpus size")
    ap.add_argument("--device", type=str, default="cpu", help="torch device for the matmul (cpu or cuda)")
    ap.add_argument("--brain-mask", action="store_true",
                    help="restrict Pearson to the union of nonzero voxels in the real corpus. "
                         "REQUIRED for skull-stripped/registered sMRI: the shared zero background "
                         "otherwise inflates inter-subject correlation (DLBS calibration: full-volume "
                         "floor 0.97 with 39%% false-positives >0.93, vs brain-masked floor 0.90 / 0%%).")
    args = ap.parse_args()

    if not args.synthetic.exists():
        print(f"ERROR: synthetic dir not found: {args.synthetic}", file=sys.stderr)
        return 2
    if not args.real.exists():
        print(f"ERROR: real dir not found: {args.real}", file=sys.stderr)
        return 2

    print(f">>> listing volumes  modalities={args.modalities}")
    synth_entries = _list_volumes(args.synthetic, args.modalities)
    real_entries = _list_volumes(args.real, args.modalities)
    if args.limit_synth:
        synth_entries = synth_entries[: args.limit_synth]
    if args.limit_real:
        real_entries = real_entries[: args.limit_real]
    if not synth_entries or not real_entries:
        print(f"ERROR: empty corpus  synth={len(synth_entries)}  real={len(real_entries)}", file=sys.stderr)
        return 2
    print(f"    synthetic: {len(synth_entries)}  real: {len(real_entries)}")

    print(">>> loading real corpus into memory")
    first = _load_flat(real_entries[0], None)
    V = first.size
    real_mat = np.empty((len(real_entries), V), dtype=np.float32)
    real_mat[0] = first
    target_shape = None  # _load_flat would have already validated against first if we passed one
    for i, e in enumerate(real_entries[1:], start=1):
        real_mat[i] = _load_flat(e, None)
        if real_mat[i].size != V:
            print(f"ERROR: real volume {i} has flat size {real_mat[i].size} != {V}", file=sys.stderr)
            return 2
        if (i + 1) % 50 == 0:
            print(f"    loaded {i+1}/{len(real_entries)}")
    print(f"    real_mat shape: {real_mat.shape}  approx GB: {real_mat.nbytes/1e9:.2f}")

    brain_cols = None
    if args.brain_mask:
        brain_cols = (real_mat != 0).any(axis=0)
        n_brain = int(brain_cols.sum())
        print(f">>> brain-masking: {n_brain:,}/{V:,} voxels nonzero in >=1 real volume ({n_brain/V:.1%})")
        real_mat = real_mat[:, brain_cols]

    device = torch.device(args.device)
    print(f">>> mean-centering + L2-normalizing real rows on {device}")
    real_t = torch.from_numpy(real_mat).to(device)
    del real_mat
    real_norm = _norm_rows(real_t)
    del real_t

    def _entry_name(e: Path | tuple[Path, ...]) -> str:
        if isinstance(e, tuple):
            return e[0].parent.name
        n = e.name
        for suf in (".nii.gz", ".nii"):
            if n.endswith(suf):
                return n[: -len(suf)]
        return e.stem

    real_names = [_entry_name(e) for e in real_entries]

    print(">>> auditing synthetic volumes")
    rows: list[dict] = []
    K = max(1, args.topk)
    n_synth = len(synth_entries)
    for batch_start in range(0, n_synth, args.chunk):
        batch_end = min(batch_start + args.chunk, n_synth)
        batch_entries = synth_entries[batch_start:batch_end]
        batch = np.stack([_load_flat(e, None) for e in batch_entries], axis=0)
        if batch.shape[1] != V:
            print(f"ERROR: synthetic batch flat size {batch.shape[1]} != real {V}", file=sys.stderr)
            return 2
        if brain_cols is not None:
            batch = batch[:, brain_cols]

        batch_t = torch.from_numpy(batch).to(device)
        batch_norm = _norm_rows(batch_t)

        # corr[i, j] = dot(synth_norm[i], real_norm[j]) — exactly Pearson because both rows are
        # mean-centered and L2-normalized.
        corr = batch_norm @ real_norm.T  # (B, N_real)
        topk_vals, topk_idx = torch.topk(corr, k=min(K, corr.shape[1]), dim=1)
        topk_vals = topk_vals.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()

        for i, entry in enumerate(batch_entries):
            synth_name = _entry_name(entry)
            row = {"synthetic": synth_name, "max_corr": float(topk_vals[i, 0])}
            for k in range(min(K, topk_vals.shape[1])):
                row[f"top{k+1}_corr"] = float(topk_vals[i, k])
                row[f"top{k+1}_real"] = real_names[int(topk_idx[i, k])]
            rows.append(row)

        print(f"    audited {batch_end}/{n_synth}  batch max_corr: {topk_vals[:, 0].max():.4f}")

    print(">>> writing report")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    max_corrs = np.array([r["max_corr"] for r in rows])
    print()
    print(f"audited {len(rows)} synthetic volumes against {len(real_entries)} real")
    print(f"max_corr: mean={max_corrs.mean():.4f}  median={np.median(max_corrs):.4f}  "
          f"p95={np.percentile(max_corrs, 95):.4f}  max={max_corrs.max():.4f}")
    print(f"frac max_corr > 0.93 (paper's 2D memorization-concern threshold): "
          f"{(max_corrs > 0.93).mean():.0%}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
