#!/usr/bin/env bash
# Spark-side ridge runner: consumes the FOMO25 embeddings parquet
# produced by Legion and runs the same Nima-style GroupKFold(5) +
# StandardScaler + RidgeCV protocol used for the BrainIAC ridge.
#
# Outputs four bias-correction MAE/RMSE/R²/r/bias dicts (raw, cole,
# beheshti, zhang) so the result lines up alongside ridge_brainiac_embed.json
# in the same comparison table.
#
# Run after Legion delivers /data/datasets/smri-fm-cmp/fomo-embeds/ds004856/fomo25_embeddings.parquet
set -euo pipefail

REPO="${REPO:-/home/mhough/dev/smri-fm}"
EMB_PARQUET="${EMB_PARQUET:-/data/datasets/smri-fm-cmp/fomo-embeds/ds004856/fomo25_embeddings.parquet}"
OUT_DIR="${OUT_DIR:-${REPO}/experiments/dlbs_morphometry_benchmark/results}"
PARTICIPANTS="${PARTICIPANTS:-/data/raw/openneuro/ds004856/participants.tsv}"

if [ ! -f "${EMB_PARQUET}" ]; then
    echo "ERROR: ${EMB_PARQUET} does not exist yet. Wait for Legion FOMO25 extraction to land." >&2
    exit 1
fi

cd "${REPO}/experiments/dlbs_morphometry_benchmark"
"${REPO}/.venv/bin/python" scripts/fit_ridge_baseline.py \
    --features "${EMB_PARQUET}" \
    --tool fomo25_embed \
    --value-col value \
    --participants "${PARTICIPANTS}" \
    --cv-mode groupkfold5 \
    --out "${OUT_DIR}/ridge_fomo25_embed.json" \
    "${@}"

echo
echo "=== summary ==="
"${REPO}/.venv/bin/python" -c "
import json
r = json.loads(open('${OUT_DIR}/ridge_fomo25_embed.json').read())
print(f\"FOMO25 ridge | n_scans={r['n_scans']} subj={r['n_subjects']} feats={r['n_features']}\")
for s in ('raw', 'cole', 'beheshti', 'zhang'):
    m = r[s]
    print(f\"  {s:10s} MAE={m['mae']:6.3f}  RMSE={m['rmse']:6.3f}  R2={m['r2']:+.3f}  r={m['pearson_r']:+.3f}  bias={m['bias']:+.3f}\")
"
