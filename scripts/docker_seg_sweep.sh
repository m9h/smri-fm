#!/usr/bin/env bash
# Host-side launcher: run the SEG009 + SEG010 sweep over SIAM + mmunetvae inside
# one NGC PyTorch container (deps installed once, then 4 finetune_seg runs).
#   DEBUG=1 bash scripts/docker_seg_sweep.sh        # quick pipeline smoke (default)
#   DEBUG=0 bash scripts/docker_seg_sweep.sh        # real 200-epoch runs
#   ARMS="smri_mmunetvae" TASKS="SEG010_FOMO26_TrigeminalNeuralgia" bash scripts/docker_seg_sweep.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
DEBUG=${DEBUG:-1}
ARMS=${ARMS:-"smri_siam smri_mmunetvae"}
TASKS=${TASKS:-"SEG009_FOMO26_Meningioma SEG010_FOMO26_TrigeminalNeuralgia"}

MOUNTS=(-v "$REPO":/workspace/smri-fm -v /data/datasets/fomo26:/fomo26)
[ -d /home/mhough/siam_params ] && MOUNTS+=(-v /home/mhough/siam_params:/siam_params:ro)

# in-container loop: one deps install, then each (arm,task) via the driver
INNER='set -e; ok=0; fail=0;
for arm in '"$ARMS"'; do for task in '"$TASKS"'; do
  echo "########## $arm / $task ##########";
  if ARM=$arm SEG_TASK=$task DEBUG='"$DEBUG"' bash scripts/finetune_seg_arm.sh; then
    ok=$((ok+1)); else fail=$((fail+1)); echo "[FAILED] $arm / $task"; fi;
done; done;
echo ">>> seg sweep done: $ok ok, $fail failed"'

exec docker run --rm --gpus all --shm-size=16g \
  -e DEBUG="$DEBUG" \
  "${MOUNTS[@]}" \
  -w /workspace/smri-fm \
  "$IMAGE" \
  bash -c "$INNER"
