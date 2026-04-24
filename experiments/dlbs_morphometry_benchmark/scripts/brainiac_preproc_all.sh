#!/usr/bin/env bash
# Drive brainiac-preproc-arm across all DLBS subjects that already have
# FastSurfer/T1Prep outputs under /data/datasets/smri-fm-cmp/fastsurfer.
# This gives us the same 60-scan cohort for BrainIAC embedding extraction
# as rung 1–4 of the morphometry benchmark.
#
# Output: /data/datasets/smri-fm-cmp/brainiac-preproc/ds004856/<subject>_<session>.nii.gz
set -euo pipefail

DATASET="${DATASET:-ds004856}"
RAW_ROOT="/data/raw/openneuro/${DATASET}"
FS_ROOT="/data/datasets/smri-fm-cmp/fastsurfer/${DATASET}"
OUT_ROOT="/data/datasets/smri-fm-cmp/brainiac-preproc/${DATASET}"
IMG="brainiac-preproc-arm:dev"

mkdir -p "${OUT_ROOT}"

# Discover target scans from FastSurfer output names (<subject>_<session> dirs)
mapfile -t SIDS < <(ls "${FS_ROOT}" 2>/dev/null | sort)
echo "found ${#SIDS[@]} fastsurfer sessions → brainiac-preproc target list"

done_count=0
skipped_count=0
fail_count=0
for SID in "${SIDS[@]}"; do
    # SID = sub-XXXX_ses-waveN
    SUBJECT="${SID%_ses-*}"
    SESSION="ses-${SID##*_ses-}"
    OUT_FILE="${OUT_ROOT}/${SID}.nii.gz"

    if [[ -s "${OUT_FILE}" ]]; then
        ((skipped_count++)) || true
        continue
    fi

    # Find raw T1 (MPRAGE, run-1)
    RAW_T1="$(find "${RAW_ROOT}/${SUBJECT}/${SESSION}/anat" \
        -maxdepth 1 -name "${SID}_acq-MPRAGE_run-1_T1w.nii.gz" 2>/dev/null \
        | head -1)"
    if [[ -z "${RAW_T1}" ]]; then
        echo "WARN: no raw T1 for ${SID} — skipping"
        ((fail_count++)) || true
        continue
    fi

    ANAT_DIR="$(dirname "${RAW_T1}")"
    T1_BASE="$(basename "${RAW_T1}")"

    echo ">>> brainiac-preproc ${SID}"
    if docker run --gpus all --rm \
        --user "$(id -u):$(id -g)" \
        -v "${ANAT_DIR}:/input:ro" \
        -v "${OUT_ROOT}:/output" \
        -e HOME=/tmp \
        "${IMG}" \
            --input "/input/${T1_BASE}" \
            --pat_id "${SID}" \
            --out_dir /output \
            --temp_img /opt/brainiac_preproc/temp_head.nii.gz 2>&1 | tail -3; then
        ((done_count++)) || true
    else
        echo "FAIL: ${SID}"
        ((fail_count++)) || true
    fi
done

echo "=== brainiac_preproc_all done: ${done_count} new, ${skipped_count} skipped, ${fail_count} failed ==="
