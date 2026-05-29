#!/usr/bin/env bash
# Host-side launcher: run a SIAM FOMO26 finetune inside the NGC PyTorch container
# with GPU access. First arg = in-container driver script (default CLS002).
#   bash scripts/docker_finetune_siam.sh                                   # CLS002 smoke
#   DEBUG=0 bash scripts/docker_finetune_siam.sh scripts/finetune_siam_regr002.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
DEBUG=${DEBUG:-1}
DRIVER=${1:-scripts/finetune_siam_cls002.sh}

exec docker run --rm --gpus all \
  --shm-size=16g \
  -e DEBUG="$DEBUG" \
  -v "$REPO":/workspace/smri-fm \
  -v /data/datasets/fomo26:/fomo26 \
  -v /home/mhough/siam_params:/siam_params:ro \
  -w /workspace/smri-fm \
  "$IMAGE" \
  bash "$DRIVER"
