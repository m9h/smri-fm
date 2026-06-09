#!/usr/bin/env bash
# Waits for the new-arm parquets, ridges each (CV + fixed-TEST), renders the table.
set -uo pipefail
cd /home/mhough/dev/smri-fm-fomo26/experiments/fomo26_fm_benchmark/brainage_probe
PY=/home/mhough/dev/smri-fm/.venv/bin/python

# out_dir/parquet_name : tool : out_stem
ARMS=(
  "fomo60k_contrastive_regular_embed/fomo60k_embeddings.parquet:fomo60k_embed:fomo60k_contrastive_regular"
  "fomo60k_combined_modality_embed/fomo60k_embeddings.parquet:fomo60k_embed:fomo60k_combined_modality"
  "fomo60k_contrastive_modality_embed/fomo60k_embeddings.parquet:fomo60k_embed:fomo60k_contrastive_modality"
  "simclr3d_embed/simclr3d_embeddings.parquet:simclr3d_embed:simclr3d"
  "brainiac_embed/brainiac_embeddings.parquet:brainiac_embed:brainiac"
  "siam_embed/siam_embeddings.parquet:siam_embed:siam"
)
for entry in "${ARMS[@]}"; do
  IFS=':' read -r rel tool stem <<< "$entry"
  pq="results/$rel"
  for i in $(seq 1 480); do [ -f "$pq" ] && break; sleep 10; done
  if [ -f "$pq" ]; then
    $PY scripts/fit_ridge_baseline.py --features "$pq" --tool "$tool" --value-col value \
      --participants data/participants.tsv --cv-mode groupkfold5 \
      --out "results/ridge_${stem}_cv.json" >/dev/null 2>&1 && \
    $PY scripts/fit_ridge_fixed_split.py --features "$pq" --tool "$tool" --value-col value \
      --participants data/participants.tsv --test-subjects data/test_subjects.json \
      --out "results/ridge_${stem}_fixedtest.json" >/dev/null 2>&1 && echo "ridged $stem" || echo "RIDGE FAIL $stem"
  else
    echo "TIMEOUT $stem ($pq)"
  fi
done
echo; echo "================ FULL TABLE ================"
$PY scripts/compare_fomo26_t3.py
