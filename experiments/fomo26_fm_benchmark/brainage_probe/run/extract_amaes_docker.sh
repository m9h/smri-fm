#!/usr/bin/env bash
# Host launcher: extract frozen AMAES embeddings in the NGC PyTorch container.
# Mirrors scripts/docker_finetune_siam.sh mounts; also binds /data/datasets/fomo26
# at the SAME path so the symlink farm (absolute targets) resolves in-container.
#   bash extract_amaes_docker.sh                 # all 494
#   SMOKE=5 bash extract_amaes_docker.sh         # first 5 (smoke)
set -euo pipefail

REPO=/home/mhough/dev/smri-fm-fomo26
PROBE=$REPO/experiments/fomo26_fm_benchmark/brainage_probe
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.04-py3}"

# Optional smoke subset: first N pat_ids into a temp csv + out dir
INPUT_CSV=$PROBE/data/input_csv.csv
OUT_SUB=amaes_embed
if [ -n "${SMOKE:-}" ]; then
  head -n $(( SMOKE + 1 )) "$INPUT_CSV" > "$PROBE/data/input_csv_smoke.csv"
  INPUT_CSV=$PROBE/data/input_csv_smoke.csv
  OUT_SUB=amaes_embed_smoke
  echo ">>> SMOKE: $SMOKE subjects"
fi

exec docker run --rm --gpus all --shm-size=16g \
  -e INPUT_CSV="/workspace/smri-fm/experiments/fomo26_fm_benchmark/brainage_probe/${INPUT_CSV#$PROBE/}" \
  -e OUT_DIR="/workspace/smri-fm/experiments/fomo26_fm_benchmark/brainage_probe/results/${OUT_SUB}" \
  -v "$REPO":/workspace/smri-fm \
  -v /data/datasets/fomo26:/data/datasets/fomo26 \
  -w /workspace/smri-fm \
  "$IMAGE" \
  bash /workspace/smri-fm/experiments/fomo26_fm_benchmark/brainage_probe/run/extract_amaes_incontainer.sh
