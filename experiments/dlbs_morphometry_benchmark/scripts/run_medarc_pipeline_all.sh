#!/usr/bin/env bash
# Run MedARC smri-fm pipeline.py (SynthStrip + rigid MNI + SynthSeg) across
# all 23 DLBS subjects. Builds a T1-only BIDS subset under /tmp so
# pipeline.py skips the FLAIR/T2w paths that we don't need for SynthSeg
# volumes → ridge. Runs inside ghcr.io/m9h/medarc-smri-fm:latest.
#
# Output structure (matches sub-1003's existing layout):
#   /data/datasets/smri-fm-cmp/medarc-smri-fm/ds004856/<sub>/<sub>/<ses>/anat/
#     <sub>_<ses>_acq-MPRAGE_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
#     ... _desc-brain_mask_T1w.nii.gz
#   /data/datasets/smri-fm-cmp/medarc-smri-fm/ds004856/<sub>/synthseg/<sub>/<ses>/anat/
#     *_desc-synthseg_volumes.tsv  <- what ridge reads
set -euo pipefail

DATASET="${DATASET:-ds004856}"
RAW_ROOT="/data/raw/openneuro/${DATASET}"
OUT_ROOT="/data/datasets/smri-fm-cmp/medarc-smri-fm/${DATASET}"
BIDS_T1_ONLY="/tmp/medarc_t1only_bids_${DATASET}"
IMG="ghcr.io/m9h/medarc-smri-fm:latest"
LOG_ROOT="/data/mhough/tmp/medarc_pipeline"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# Build a T1-only BIDS symlink tree (dataset_description.json + participants.tsv + every subject's
# ses-*/anat/*T1w.* symlinked). Skip T2/FLAIR so pipeline.py doesn't waste CPU on them.
build_t1_only_bids() {
    rm -rf "${BIDS_T1_ONLY}"
    mkdir -p "${BIDS_T1_ONLY}"
    # Required BIDS top-level files
    for f in dataset_description.json participants.tsv participants.json README CHANGES; do
        if [[ -e "${RAW_ROOT}/${f}" ]]; then
            ln -sf "${RAW_ROOT}/${f}" "${BIDS_T1_ONLY}/${f}"
        fi
    done
    # Per-subject anat T1 symlinks (MPRAGE run-1 T1w only + .json sidecar)
    local subs
    subs=$(ls "${RAW_ROOT}" | grep -E '^sub-' | sort)
    for sub in ${subs}; do
        for ses_dir in "${RAW_ROOT}/${sub}"/ses-*; do
            [[ -d "${ses_dir}" ]] || continue
            ses=$(basename "${ses_dir}")
            anat="${ses_dir}/anat"
            [[ -d "${anat}" ]] || continue
            mkdir -p "${BIDS_T1_ONLY}/${sub}/${ses}/anat"
            for ext in nii.gz json; do
                t1="${anat}/${sub}_${ses}_acq-MPRAGE_run-1_T1w.${ext}"
                if [[ -e "${t1}" ]]; then
                    ln -sf "${t1}" "${BIDS_T1_ONLY}/${sub}/${ses}/anat/"
                fi
            done
        done
    done
}

run_one_subject() {
    local sub="$1"
    local log="${LOG_ROOT}/${sub}.log"
    # Skip if a SynthSeg volumes.tsv already exists for this subject
    local done_marker
    done_marker=$(find "${OUT_ROOT}/${sub}/synthseg" -name "*_desc-synthseg_volumes.tsv" 2>/dev/null | head -1 || true)
    if [[ -n "${done_marker}" ]]; then
        local n
        n=$(find "${OUT_ROOT}/${sub}/synthseg" -name "*_desc-synthseg_volumes.tsv" 2>/dev/null | wc -l)
        # Expect ~1 per session (T1-only); 3 sessions = 3 TSVs
        if [[ "${n}" -ge 3 ]]; then
            echo "skip ${sub} (already has ${n} synthseg volumes TSVs)"
            return 0
        fi
    fi

    echo ">>> medarc pipeline.py ${sub} $(date -Iseconds)" | tee -a "${log}"
    # Pre-create host output dirs with user ownership so docker doesn't root-own them
    mkdir -p "${OUT_ROOT}/${sub}" "${OUT_ROOT}/${sub}/logs" "${OUT_ROOT}/${sub}/synthseg"
    docker run --gpus all --rm \
      --user "$(id -u):$(id -g)" \
      -v "${BIDS_T1_ONLY}:/bids:ro" \
      -v "${RAW_ROOT}:/raw:ro" \
      -v "${OUT_ROOT}/${sub}:/output:rw" \
      -v "${HOME}/license.txt:/usr/lib/freesurfer/license.txt:ro" \
      -e HOME=/tmp \
      --entrypoint python3 \
      "${IMG}" /app/smri-fm/preprocessing/pipeline.py \
        --bids /bids \
        --subject "${sub}" \
        --output /output \
        --log_dir /output/logs \
        --n_workers 4 \
        --itk_threads 4 \
        --synthseg \
        --synthseg_output /output/synthseg \
        --synthseg_workers 1 2>&1 | tee -a "${log}" | tail -5
    echo "<<< done ${sub} $(date -Iseconds)" | tee -a "${log}"
}

build_t1_only_bids
echo "T1-only BIDS tree: ${BIDS_T1_ONLY}"
find "${BIDS_T1_ONLY}" -name "*_T1w.nii.gz" | wc -l | xargs echo "T1 files:"

SUBJECTS=(
    sub-1003 sub-1007 sub-1013 sub-1022 sub-1023 sub-103 sub-1031
    sub-1045 sub-1054 sub-1058 sub-1084 sub-1093 sub-1139 sub-1141
    sub-1146 sub-1149 sub-1153 sub-1157 sub-1172 sub-1175 sub-1183
    sub-1200 sub-1220
)

for sub in "${SUBJECTS[@]}"; do
    run_one_subject "${sub}"
done

echo "=== medarc pipeline.py ALL done $(date -Iseconds) ==="
echo "synthseg volumes tsvs: $(find ${OUT_ROOT} -name '*_desc-synthseg_volumes.tsv' | wc -l)"
