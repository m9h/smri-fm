#!/usr/bin/env bash
# Minimal T1Prep longitudinal runner that bypasses upstream
# process_longitudinal.sh (which has bugs around readonly mounts, TTY-only
# variables, and shell-line parsing). We do the two steps ourselves:
#
#   1) Copy the T1s into a writable scratch dir.
#   2) Run realign_longitudinal.py directly on them (--save-resampled writes
#      r<stem>.nii.gz next to the inputs in /scratch).
#   3) For each timepoint, call T1Prep with --long-data pointing at the
#      scratch dir so it picks up the realigned image.
#
# Usage:
#   t1prep_longitudinal.sh DATASET SUBJECT
#
# Example:
#   t1prep_longitudinal.sh ds004856 sub-1003
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 DATASET SUBJECT" >&2
  exit 1
fi

DATASET="$1"
SUBJECT="$2"
SRC="/data/raw/openneuro/${DATASET}/${SUBJECT}"
SCRATCH="/data/datasets/smri-fm-cmp/scratch-long/${SUBJECT}"
OUT_ROOT="/data/datasets/smri-fm-cmp/t1prep-long/${DATASET}"
LOG="/home/mhough/dev/T1Prep/comparison/logs/t1prep_long_${SUBJECT}.log"

rm -rf "$SCRATCH" && mkdir -p "$SCRATCH" "$OUT_ROOT"
{
  echo "=== t1prep_longitudinal ${SUBJECT} $(date -Iseconds) ==="

  # Stage T1s into flat scratch dir.
  mapfile -t T1S < <(find "$SRC" -name "${SUBJECT}_ses-wave*_acq-MPRAGE_run-1_T1w.nii.gz" | sort)
  echo "timepoints: ${#T1S[@]}"
  CPATHS=()
  for p in "${T1S[@]}"; do
    b=$(basename "$p")
    cp "$p" "${SCRATCH}/${b}"
    CPATHS+=("/scratch/${b}")
  done

  # Step 1: rigid realignment (--save-resampled writes r<stem>.nii.gz into /scratch)
  # The t1prep package isn't pip-installed in the venv; it sits under
  # /opt/T1Prep/src. Setting PYTHONPATH explicitly bypasses the need to
  # activate T1Prep's venv via its bash wrapper.
  echo ">>> realign_longitudinal"
  docker run --gpus all --rm \
    --user "$(id -u):$(id -g)" \
    --entrypoint /opt/T1Prep/env/bin/python \
    -e PYTHONPATH=/opt/T1Prep/src \
    -v "${SCRATCH}:/scratch" \
    t1prep-grace:latest \
    -m t1prep.realign_longitudinal \
      --inputs "${CPATHS[@]}" \
      --out-dir /scratch \
      --save-resampled 2>&1 | tail -10

  echo "---scratch contents after realign---"
  ls "${SCRATCH}" 2>&1

  # Step 2: T1Prep --long-data per timepoint.
  for p in "${T1S[@]}"; do
    SES=$(basename "$(dirname "$(dirname "$p")")")          # ses-waveN
    SID="${SUBJECT}_${SES}"
    B=$(basename "$p")
    OUT="${OUT_ROOT}/${SID}"
    mkdir -p "$OUT"

    # T1Prep's --long-data flag, despite the help text implying a directory
    # search, is consumed by scripts/T1Prep as `input="$long_data"` — i.e.
    # it must be a FILE path to the realigned volume. Point it at the
    # concrete _desc-realigned.nii.gz file.
    REALIGNED_NAME="${B%.nii.gz}_desc-realigned.nii.gz"
    echo ">>> T1Prep long ${SID} (realigned=${REALIGNED_NAME})"
    docker run --gpus all --rm \
      --user "$(id -u):$(id -g)" \
      -v "${SCRATCH}:/long_data:ro" \
      -v "${OUT}:/output" \
      t1prep-grace:latest \
        --out-dir /output --gz \
        --long-data "/long_data/${REALIGNED_NAME}" \
        "/long_data/${B}" 2>&1 | tail -5
  done

  echo "=== t1prep_longitudinal ${SUBJECT} done $(date -Iseconds) ==="
} 2>&1 | tee "$LOG"
