#!/usr/bin/env bash
# Driver: run extract_fomo25_embeddings.py inside the Grace Blackwell-tuned
# fomo25-arm container on DGX Spark. Inputs and outputs on /data NFS so
# the parquet is visible from anywhere on the network.
#
# The image bakes the AMAES_resenc_b checkpoint at build time, so this
# starts with no network round-trip.
set -euo pipefail

IMG="${IMG:-fomo25-arm:latest}"
SCRIPT_DIR="${SCRIPT_DIR:-/home/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/scripts}"
CSV="${CSV:-/home/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/results/brainiac_dlbs_csv.csv}"
SCANS="${SCANS:-/data/datasets/smri-fm-cmp/brainiac-preproc/ds004856}"
OUT="${OUT:-/data/datasets/smri-fm-cmp/fomo-embeds/ds004856}"
DEVICE="${DEVICE:-cuda}"
CKPT="${CKPT:-/opt/checkpoints/resenc_unet_b.ckpt}"

mkdir -p "${OUT}"

echo "Starting FOMO25 extraction on Spark (IMG: ${IMG})"
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "${SCRIPT_DIR}/extract_fomo25_embeddings.py:/work/extract.py:ro" \
  -v "${CSV}:/csv/cohort.csv:ro" \
  -v "${SCANS}:/scans:ro" \
  -v "${OUT}:/out:rw" \
  --entrypoint python \
  "${IMG}" \
  /work/extract.py \
    --input_csv /csv/cohort.csv \
    --root_dir /scans \
    --out_dir /out \
    --checkpoint "${CKPT}" \
    --device "${DEVICE}" \
    "${@}"

echo
echo "outputs:"
ls -la "${OUT}"/
