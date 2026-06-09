#!/usr/bin/env bash
# Waits for the bridge-arm parquets, ridges each (CV + fixed-TEST), renders the table.
set -uo pipefail
cd /home/mhough/dev/smri-fm-fomo26/experiments/fomo26_fm_benchmark/brainage_probe
PY=/home/mhough/dev/smri-fm/.venv/bin/python

# arm: parquet_path : tool : out_stem
ARMS=(
  "results/fomo60k_combined_regular_embed/fomo60k_embeddings.parquet:fomo60k_embed:fomo60k_combined_regular"
  "results/anatcl_embed/anatcl_embeddings.parquet:anatcl_embed:anatcl"
  "results/triad_embed/triad_embeddings.parquet:triad_embed:triad"
)

ridge_one() {
  local pq="$1" tool="$2" stem="$3"
  $PY scripts/fit_ridge_baseline.py --features "$pq" --tool "$tool" --value-col value \
    --participants data/participants.tsv --cv-mode groupkfold5 \
    --out "results/ridge_${stem}_cv.json" >/dev/null 2>&1
  $PY scripts/fit_ridge_fixed_split.py --features "$pq" --tool "$tool" --value-col value \
    --participants data/participants.tsv --test-subjects data/test_subjects.json \
    --out "results/ridge_${stem}_fixedtest.json" >/dev/null 2>&1
  echo "ridged $stem"
}

for entry in "${ARMS[@]}"; do
  IFS=':' read -r pq tool stem <<< "$entry"
  # wait up to 30 min for this parquet
  for i in $(seq 1 360); do [ -f "$pq" ] && break; sleep 5; done
  if [ -f "$pq" ]; then ridge_one "$pq" "$tool" "$stem"; else echo "TIMEOUT waiting $pq"; fi
done

echo; echo "================ FINAL TABLE ================"
$PY scripts/compare_fomo26_t3.py
