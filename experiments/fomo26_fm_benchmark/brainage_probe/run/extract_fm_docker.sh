#!/usr/bin/env bash
# Generic host launcher: run an in-container FM embedding extractor in the NGC
# PyTorch container with GPU. Binds /data/datasets/fomo26 at the same path so the
# symlink farm resolves.  Arg 1 = in-container driver (relative to probe/run/).
#   bash extract_fm_docker.sh extract_mmunetvae_incontainer.sh
#   SMOKE=5 bash extract_fm_docker.sh extract_mmunetvae_incontainer.sh
set -euo pipefail

REPO=/home/mhough/dev/smri-fm-fomo26
PROBE=$REPO/experiments/fomo26_fm_benchmark/brainage_probe
CPROBE=/workspace/smri-fm/experiments/fomo26_fm_benchmark/brainage_probe
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.04-py3}"
DRIVER="${1:?usage: extract_fm_docker.sh <incontainer_driver.sh>}"
ARM="$(basename "$DRIVER" .sh | sed 's/^extract_//;s/_incontainer$//')"

ENVARGS=()
if [ -n "${SMOKE:-}" ]; then
  head -n $(( SMOKE + 1 )) "$PROBE/data/input_csv.csv" > "$PROBE/data/input_csv_smoke.csv"
  ENVARGS+=(-e INPUT_CSV="$CPROBE/data/input_csv_smoke.csv"
            -e OUT_DIR="$CPROBE/results/${ARM}_embed_smoke")
  echo ">>> SMOKE $SMOKE subjects ($ARM)"
fi

exec docker run --rm --gpus all --shm-size=16g \
  "${ENVARGS[@]}" \
  -v "$REPO":/workspace/smri-fm \
  -v /data/datasets/fomo26:/data/datasets/fomo26 \
  -w /workspace/smri-fm \
  "$IMAGE" \
  bash "$CPROBE/run/$DRIVER"
