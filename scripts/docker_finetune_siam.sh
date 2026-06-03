#!/usr/bin/env bash
# Host-side launcher: run a FOMO26 FM finetune inside the NGC PyTorch container
# with GPU access. First arg = in-container driver script (default SIAM CLS002).
#   bash scripts/docker_finetune_siam.sh                                       # SIAM CLS002 smoke
#   DEBUG=0 bash scripts/docker_finetune_siam.sh scripts/finetune_siam_regr002.sh
#   DEBUG=0 bash scripts/docker_finetune_siam.sh scripts/finetune_fomo60k_regr002.sh
#   DEBUG=0 bash scripts/docker_finetune_siam.sh scripts/finetune_brainiac_regr002.sh
# Mounts SIAM params and the BrainIAC repo read-only when those host dirs exist,
# so the same launcher serves every bridge (each driver uses only what it needs).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
DEBUG=${DEBUG:-1}
DRIVER=${1:-scripts/finetune_siam_cls002.sh}

MOUNTS=(-v "$REPO":/workspace/smri-fm -v /data/datasets/fomo26:/fomo26)
# Optional bridge sources: bind read-only at their host path only if present
# (BrainIAC's bridge imports model.py via its absolute /home/mhough/dev path).
[ -d /home/mhough/siam_params ] && MOUNTS+=(-v /home/mhough/siam_params:/siam_params:ro)
[ -d /home/mhough/dev/BrainIAC ] && MOUNTS+=(-v /home/mhough/dev/BrainIAC:/home/mhough/dev/BrainIAC:ro)

exec docker run --rm --gpus all \
  --shm-size=16g \
  -e DEBUG="$DEBUG" \
  "${MOUNTS[@]}" \
  -w /workspace/smri-fm \
  "$IMAGE" \
  bash "$DRIVER"
