#!/usr/bin/env bash
set -uo pipefail
cd /home/mhough/dev/smri-fm-fomo26/experiments/fomo26_fm_benchmark/brainage_probe
PY=/home/mhough/dev/smri-fm/.venv/bin/python
PQ=results/amaes_embed_160/fomo25_embeddings.parquet
for i in $(seq 1 600); do [ -f "$PQ" ] && break; sleep 10; done
[ -f "$PQ" ] || { echo "TIMEOUT: no $PQ"; exit 1; }
# ridge the native-resolution AMAES; overwrite the primary AMAES row
$PY scripts/fit_ridge_baseline.py --features "$PQ" --tool fomo25_embed --value-col value \
  --participants data/participants.tsv --cv-mode groupkfold5 --out results/ridge_amaes_cv.json >/dev/null 2>&1
$PY scripts/fit_ridge_fixed_split.py --features "$PQ" --tool fomo25_embed --value-col value \
  --participants data/participants.tsv --test-subjects data/test_subjects.json \
  --out results/ridge_amaes_fixedtest.json >/dev/null 2>&1
echo "=== AMAES 160^3 ridged ==="
echo "================ FINAL TABLE (AMAES @ native 160^3) ================"
$PY scripts/compare_fomo26_t3.py
