#!/usr/bin/env bash
# Single-subject smoke runner: FastSurfer seg_only + T1Prep across all sessions
# for one subject. Arm64 host, docker.
#
# Usage:
#   smoke_all_subject.sh DATASET SUBJECT [--include-long]
#
# Examples:
#   smoke_all_subject.sh ds004856 sub-1003
#   smoke_all_subject.sh ds004856 sub-1007 --include-long
#
# Assumptions:
#   - Raw data at /data/raw/openneuro/<dataset>/<subject>/ses-*/anat/*_T1w.nii.gz
#   - Output root /data/datasets/smri-fm-cmp/<tool>/<dataset>/<subject>_<session>
#   - fastsurfer-arm and t1prep-arm images built locally
#     (published at ghcr.io/m9h/{fastsurfer,t1prep}-arm:latest)
#   - FreeSurfer license at ~/license.txt; FastSurfer checkpoints at ~/fs_checkpoints
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 DATASET SUBJECT [--include-long]" >&2
  exit 1
fi

DATASET="$1"
SUBJECT="$2"
INCLUDE_LONG=0
shift 2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-long) INCLUDE_LONG=1 ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
  shift
done

ROOT_IN="/data/raw/openneuro/${DATASET}/${SUBJECT}"
ROOT_FS="/data/datasets/smri-fm-cmp/fastsurfer/${DATASET}"
ROOT_T1="/data/datasets/smri-fm-cmp/t1prep/${DATASET}"
ROOT_T1L="/data/datasets/smri-fm-cmp/t1prep-long/${DATASET}/${SUBJECT}"
mkdir -p "${ROOT_FS}" "${ROOT_T1}" "${ROOT_T1L}"

LOG="/home/mhough/dev/T1Prep/comparison/logs/smoke_${SUBJECT}.log"
{
  echo "=== smoke_all_subject ${DATASET} ${SUBJECT} start $(date -Iseconds) ==="

  # Discover T1s across all sessions (MPRAGE preferred; run-1 first if multiple)
  mapfile -t T1S < <(find "${ROOT_IN}" -name "${SUBJECT}_ses-wave*_acq-MPRAGE_*T1w.nii.gz" 2>/dev/null | sort)
  echo "found ${#T1S[@]} T1s"
  if (( ${#T1S[@]} == 0 )); then
    echo "ERROR: no T1s found under ${ROOT_IN} — is the subject synced?"
    exit 2
  fi

  # FastSurfer seg_only per session
  for T1 in "${T1S[@]}"; do
    SES="$(basename "$(dirname "$(dirname "${T1}")")")"    # ses-waveN
    SID="${SUBJECT}_${SES}"
    T1_BASE="$(basename "${T1}")"
    ANAT_DIR="$(dirname "${T1}")"
    echo ">>> FastSurfer ${SID}"
    docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm \
      --user "$(id -u):$(id -g)" \
      -v "${ANAT_DIR}:/data:ro" \
      -v "${ROOT_FS}:/output" \
      -v "${HOME}/license.txt:/opt/FastSurfer/license.txt:ro" \
      -v "${HOME}/fs_checkpoints:/opt/FastSurfer/checkpoints:ro" \
      fastsurfer-arm:latest \
      ./run_fastsurfer.sh \
        --t1 "/data/${T1_BASE}" --sid "${SID}" --sd /output \
        --seg_only --parallel \
        --fs_license /opt/FastSurfer/license.txt 2>&1 | tail -3
  done

  # T1Prep cross-sectional per session
  for T1 in "${T1S[@]}"; do
    SES="$(basename "$(dirname "$(dirname "${T1}")")")"
    SID="${SUBJECT}_${SES}"
    T1_BASE="$(basename "${T1}")"
    ANAT_DIR="$(dirname "${T1}")"
    OUT="${ROOT_T1}/${SID}"
    mkdir -p "${OUT}"
    echo ">>> T1Prep ${SID}"
    docker run --gpus all --rm \
      --user "$(id -u):$(id -g)" \
      -v "${ANAT_DIR}:/input:ro" \
      -v "${OUT}:/output" \
      t1prep-arm:latest \
        --out-dir /output --gz "/input/${T1_BASE}" 2>&1 | tail -3
  done

  # Optional: T1Prep longitudinal (realignment + --long-data per timepoint)
  if [[ "${INCLUDE_LONG}" -eq 1 ]]; then
    echo ">>> T1Prep longitudinal ${SUBJECT}"
    CPATHS=()
    for p in "${T1S[@]}"; do
      rel=${p#${ROOT_IN}/}
      CPATHS+=("/input/${rel}")
    done
    docker run --gpus all --rm -t \
      --user "$(id -u):$(id -g)" \
      --entrypoint /opt/T1Prep/scripts/process_longitudinal.sh \
      -v "${ROOT_IN}:/input:ro" \
      -v "${ROOT_T1L}:/output" \
      t1prep-arm:latest \
      --out-dir /output --t1prep-arg "--gz" "${CPATHS[@]}" 2>&1 | tail -10
  fi

  echo "=== smoke_all_subject ${DATASET} ${SUBJECT} finished $(date -Iseconds) ==="
} 2>&1 | tee -a "${LOG}"
