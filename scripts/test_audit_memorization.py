"""Self-contained unit test for scripts/audit_memorization.py.

Generates a synthetic NIfTI corpus on disk where the answer is known by
construction, runs the audit, and asserts the headline numbers fall where
they should:

  - Pure-noise synthetics correlate ~0 with all reals -> max_corr ~ 0.0 (low)
  - Memorized synthetics are noisy copies of real volumes -> max_corr ~ 0.99
    AND their reported top-1 match should be the source real volume.

No external data required. Runs in seconds on CPU.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPT = Path(__file__).resolve().parent / "audit_memorization.py"


def _write_nifti(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.float32), affine=np.eye(4))
    nib.save(img, str(path))


def main() -> int:
    rng = np.random.default_rng(0)
    shape = (16, 16, 16)
    n_real = 8
    n_memorized = 3   # noisy copies of reals 0,1,2
    n_fresh = 4

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        real_dir = tmp / "real"
        synth_dir = tmp / "synth"
        out_csv = tmp / "report.csv"

        # Real corpus
        real_vols = []
        for i in range(n_real):
            v = rng.standard_normal(shape).astype(np.float32)
            real_vols.append(v)
            _write_nifti(real_dir / f"real_{i:03d}.nii.gz", v)

        # Memorized synthetics: copy of real i + low-amplitude noise.
        for i in range(n_memorized):
            v = real_vols[i] + 0.05 * rng.standard_normal(shape).astype(np.float32)
            _write_nifti(synth_dir / f"synth_mem{i:03d}.nii.gz", v)

        # Fresh synthetics: independent noise.
        for i in range(n_fresh):
            v = rng.standard_normal(shape).astype(np.float32)
            _write_nifti(synth_dir / f"synth_fresh{i:03d}.nii.gz", v)

        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--synthetic", str(synth_dir),
                "--real", str(real_dir),
                "--out", str(out_csv),
                "--topk", "3",
                "--chunk", "4",
            ],
            capture_output=True, text=True,
        )
        print("--- stdout ---")
        print(proc.stdout)
        if proc.returncode != 0:
            print("--- stderr ---", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return proc.returncode

        with out_csv.open() as f:
            rows = list(csv.DictReader(f))

        memorized_rows = [r for r in rows if r["synthetic"].startswith("synth_mem")]
        fresh_rows = [r for r in rows if r["synthetic"].startswith("synth_fresh")]

        print()
        print("=== assertions ===")
        # Memorized synthetics should hit very high max_corr (well above the 0.93 threshold).
        for r in memorized_rows:
            mc = float(r["max_corr"])
            print(f"  memorized {r['synthetic']}: max_corr={mc:.4f}  top1_real={r['top1_real']}")
            assert mc > 0.95, f"expected memorized synthetic to score >0.95, got {mc}"
            idx_synth = int(r["synthetic"].replace("synth_mem", "").replace(".nii", ""))
            expected_top1 = f"real_{idx_synth:03d}"
            assert r["top1_real"] == expected_top1, (
                f"expected memorized synth_mem{idx_synth:03d} top1 to be {expected_top1}, "
                f"got {r['top1_real']}"
            )

        # Fresh synthetics should hit moderate-to-low max_corr (noise-vs-noise on this small shape
        # still gets some chance correlation; just check they're well below the memorized ones).
        for r in fresh_rows:
            mc = float(r["max_corr"])
            print(f"  fresh     {r['synthetic']}: max_corr={mc:.4f}")
            assert mc < 0.5, f"expected fresh synthetic to score <0.5, got {mc}"

        print()
        print("PASS: audit correctly distinguished memorized from fresh synthetics.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
