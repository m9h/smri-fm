#!/usr/bin/env bash
# Host launcher for the CLS002 diffusion-classification sweep. Runs the
# in-container orchestrator (scripts/finetune_cls002_all.sh) with GPU + all
# bridge mounts. Pass-through env (ARMS/TASKS/FOLDS/DEBUG/EPOCHS) selects scope.
#   DEBUG=0 ARMS="smri_siam" TASKS="CLS002_FOMO26_Infarct" FOLDS="0" \
#     bash scripts/docker_cls002_sweep.sh           # single validation run
#   bash scripts/docker_cls002_sweep.sh             # full 7-arm x 2-task x 5-fold
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=nvcr.io/nvidia/pytorch:26.04-py3

MOUNTS=(-v "$REPO":/workspace/smri-fm -v /data/datasets/fomo26:/fomo26)
[ -d /home/mhough/siam_params ] && MOUNTS+=(-v /home/mhough/siam_params:/siam_params:ro)
[ -d /home/mhough/dev/BrainIAC ] && MOUNTS+=(-v /home/mhough/dev/BrainIAC:/home/mhough/dev/BrainIAC:ro)

PASS=()
for v in ARMS TASKS FOLDS DEBUG EPOCHS BATCH SUMMARY; do
  [ -n "${!v:-}" ] && PASS+=(-e "$v=${!v}")
done

exec docker run --rm --gpus all --shm-size=16g \
  "${PASS[@]}" "${MOUNTS[@]}" \
  -w /workspace/smri-fm "$IMAGE" \
  bash scripts/finetune_cls002_all.sh
