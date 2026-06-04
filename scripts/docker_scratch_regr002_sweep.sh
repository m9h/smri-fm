#!/usr/bin/env bash
# Host-side launcher: run the three FROM-SCRATCH control REGR002 brain-age runs
# (scratch_swinvit, scratch_vitb, scratch_nnunet) inside one NGC PyTorch
# container — deps installed once, then each arm via finetune_scratch_regr002.sh.
# Outputs land in /data/datasets/fomo26/models/regr002_scratch_* (outside repo).
#   DEBUG=1 bash scripts/docker_scratch_regr002_sweep.sh   # quick smoke (default)
#   DEBUG=0 bash scripts/docker_scratch_regr002_sweep.sh   # real 50-epoch runs
#   ARMS="scratch_nnunet" bash scripts/docker_scratch_regr002_sweep.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
DEBUG=${DEBUG:-1}
ARMS=${ARMS:-"scratch_swinvit scratch_vitb scratch_nnunet"}

MOUNTS=(-v "$REPO":/workspace/smri-fm -v /data/datasets/fomo26:/fomo26)
# scratch_nnunet needs SIAM plans.json (topology only); mount read-only if present
[ -d /home/mhough/siam_params ] && MOUNTS+=(-v /home/mhough/siam_params:/siam_params:ro)

# in-container loop: one deps install (guarded in the driver), then each arm
INNER='set -e; ok=0; fail=0;
for arm in '"$ARMS"'; do
  echo "########## REGR002 control: $arm ##########";
  if ARM=$arm DEBUG='"$DEBUG"' bash scripts/finetune_scratch_regr002.sh; then
    ok=$((ok+1)); else fail=$((fail+1)); echo "[FAILED] $arm"; fi;
done;
echo ">>> scratch REGR002 sweep done: $ok ok, $fail failed"'

exec docker run --rm --gpus all --shm-size=16g \
  -e DEBUG="$DEBUG" \
  "${MOUNTS[@]}" \
  -w /workspace/smri-fm \
  "$IMAGE" \
  bash -c "$INNER"
