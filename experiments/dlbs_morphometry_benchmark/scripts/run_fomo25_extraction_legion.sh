#!/usr/bin/env bash
# Driver: run extract_fomo25_embeddings.py inside jbanusco/sslmmunetave:1.0.0
# on Legion (amd64). Inputs and outputs all on /data NFS so Spark can read
# the resulting parquet for ridge.
#
# Invoked manually on Legion via the prompt at
# /data/mhough/dev/comparison/docs/legion_fomo25_extraction_prompt.md
set -euo pipefail

IMG="${IMG:-jbanusco/sslmmunetave:1.0.0}"
SCRIPT_DIR="${SCRIPT_DIR:-/data/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/scripts}"
CSV="${CSV:-/data/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/results/brainiac_dlbs_csv.csv}"
SCANS="${SCANS:-/data/datasets/smri-fm-cmp/brainiac-preproc/ds004856}"
OUT="${OUT:-/data/datasets/smri-fm-cmp/fomo-embeds/ds004856}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT}"

podman run --rm \
  --gpus all \
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
    --device "${DEVICE}"
