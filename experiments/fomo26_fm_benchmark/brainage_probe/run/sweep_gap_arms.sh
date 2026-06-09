#!/usr/bin/env bash
# Extract the 3 gap FM arms (simclr3d, brainiac, siam) at native 96^3.
# Each needs different host mounts/env (brainiac repo; siam params + SIAM_MODEL_DIR).
set -uo pipefail
REPO=/home/mhough/dev/smri-fm-fomo26
CPROBE=/workspace/smri-fm/experiments/fomo26_fm_benchmark/brainage_probe
IMG=nvcr.io/nvidia/pytorch:26.04-py3
COMMON=(-v "$REPO":/workspace/smri-fm -v /data/datasets/fomo26:/data/datasets/fomo26 -w /workspace/smri-fm)

echo "===== simclr3d ====="
docker run --rm --gpus all --shm-size=16g \
  -e ARM=simclr3d -e TARGET=96 \
  -e CKPT=/data/datasets/fomo26/weights/simclr3d/simclr_3d_brain_foundation.tar \
  -e OUT_DIR="$CPROBE/results/simclr3d_embed" \
  "${COMMON[@]}" "$IMG" bash "$CPROBE/run/extract_bridge_incontainer.sh" || echo "simclr3d FAILED"

echo "===== brainiac ====="
docker run --rm --gpus all --shm-size=16g \
  -e ARM=brainiac -e TARGET=96 \
  -e CKPT=/home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt \
  -e OUT_DIR="$CPROBE/results/brainiac_embed" \
  -v /home/mhough/dev/BrainIAC:/home/mhough/dev/BrainIAC:ro \
  "${COMMON[@]}" "$IMG" bash "$CPROBE/run/extract_bridge_incontainer.sh" || echo "brainiac FAILED"

echo "===== siam ====="
docker run --rm --gpus all --shm-size=16g \
  -e ARM=siam -e TARGET=96 \
  -e SIAM_MODEL_DIR=/siam_params/v0.3/pred_DS108_LcsfP_Ano \
  -e CKPT=/siam_params/v0.3/pred_DS108_LcsfP_Ano/fold_0/checkpoint_final.pth \
  -e OUT_DIR="$CPROBE/results/siam_embed" \
  -v /home/mhough/siam_params:/siam_params:ro \
  "${COMMON[@]}" "$IMG" bash "$CPROBE/run/extract_bridge_incontainer.sh" || echo "siam FAILED"

echo "===== gap sweep done ====="
ls results/simclr3d_embed/*.parquet results/brainiac_embed/*.parquet results/siam_embed/*.parquet 2>/dev/null
