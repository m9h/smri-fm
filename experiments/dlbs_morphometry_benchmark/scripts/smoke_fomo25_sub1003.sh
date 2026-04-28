#!/usr/bin/env bash
# Smoke-test the FOMO25 arm extraction on a single scan (sub-1003 wave-1).
# Confirms: container loads model, checkpoint deserialises, forward
# pass returns a non-zero embedding, parquet schema is correct.
#
# Run AFTER comparison/containers/build_fomo25_arm.sh succeeds.
set -euo pipefail

REPO="${REPO:-/home/mhough/dev/smri-fm}"
COHORT_CSV="${REPO}/experiments/dlbs_morphometry_benchmark/results/brainiac_dlbs_csv.csv"
SMOKE_CSV="/tmp/fomo25_smoke_cohort.csv"
SMOKE_OUT="/tmp/fomo25_smoke_out"

# One-row CSV: just sub-1003_ses-wave1 (header preserved)
head -1 "${COHORT_CSV}" > "${SMOKE_CSV}"
grep '^sub-1003_ses-wave1,' "${COHORT_CSV}" >> "${SMOKE_CSV}"

mkdir -p "${SMOKE_OUT}"
rm -f "${SMOKE_OUT}"/*

CSV="${SMOKE_CSV}" \
OUT="${SMOKE_OUT}" \
bash "${REPO}/experiments/dlbs_morphometry_benchmark/scripts/run_fomo25_extraction_spark.sh"

echo
echo "=== smoke verification ==="
python3 - <<'PYEOF'
import numpy as np, pandas as pd
from pathlib import Path
out = Path("/tmp/fomo25_smoke_out")
npz = np.load(out / "fomo25_embeddings.npz")
print(f"npz keys: {list(npz.keys())}")
for k in npz.keys():
    e = npz[k]
    print(f"  {k}: shape={e.shape} dtype={e.dtype} "
          f"min={e.min():.3f} max={e.max():.3f} nonzero={np.count_nonzero(e)}/{e.size}")
df = pd.read_parquet(out / "fomo25_embeddings.parquet")
print(f"parquet: rows={len(df)} cols={list(df.columns)}")
print(df.head())
print(f"  unique subjects: {df['subject'].nunique()}")
print(f"  unique tools: {df['tool'].unique()}")
PYEOF
