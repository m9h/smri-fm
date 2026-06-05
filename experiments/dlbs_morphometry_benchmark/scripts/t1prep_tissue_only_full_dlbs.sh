#!/usr/bin/env bash
# T1Prep --no-surf on every DLBS T1 — tissue segmentation only, no surface
# reconstruction. ~3-5 min/scan vs ~30 min for full T1Prep. Targets the
# "T1Prep tissue ratios vs FOMO25" question at full DLBS scale.
#
# Skips:
#   - The 42 subjects already through full T1Prep (kept their existing dir)
#   - Subjects whose tissue marker already exists (idempotent re-run safe)
#
# Output: /data/datasets/smri-fm-cmp/t1prep_tissue/ds004856/<sub>_<ses>/
#         (CAT12-style, just mri/ + report/, no surf/)
#
# Resource caps: --cpus 6, no nice (HIGH priority), PARALLEL=4 → uses
# ~24 cores at peak (oversubscribed Linux gets scheduled fairly).
set -uo pipefail

RAW_ROOT="/data/raw/openneuro/ds004856"
DERIV="/data/datasets/smri-fm-cmp/t1prep_tissue/ds004856"
LOG_DIR="/data/datasets/smri-fm-cmp/_logs/t1prep_tissue_only"
EXISTING_FULL="/data/datasets/smri-fm-cmp/t1prep/ds004856"
mkdir -p "${DERIV}" "${LOG_DIR}"

CPU_LIMIT="${CPU_LIMIT:-6}"
MEM_LIMIT="${MEM_LIMIT:-30g}"
PARALLEL="${PARALLEL:-4}"
IMG="ghcr.io/m9h/t1prep-arm:latest"

log() { printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "${LOG_DIR}/queue.log" >&2; }

# Build the queue: every (subject, session) with a T1, minus those that have
# either (a) full T1Prep already done, or (b) tissue-only output already done.
QUEUE=()
total=0
skip_existing=0
skip_done=0
for sub_dir in $(ls -d "${RAW_ROOT}"/sub-* 2>/dev/null | sort); do
    sub=$(basename "${sub_dir}")
    for ses_dir in "${sub_dir}"/ses-*; do
        [ -d "${ses_dir}" ] || continue
        ses=$(basename "${ses_dir}")
        sid="${sub}_${ses}"
        t1=$(find "${ses_dir}/anat" -name "*acq-MPRAGE_run-1_T1w.nii.gz" 2>/dev/null | head -1)
        [ -z "${t1}" ] && continue
        total=$((total + 1))
        # skip if full T1Prep already wrote a p0 segmentation for this scan,
        # checking BOTH layouts: nested (sub/ses/, used by extend_cohort_pipeline)
        # and flat (sub_ses/, used by the original 23-subject runs).
        already_done=0
        for path in "${EXISTING_FULL}/${sub}/${ses}" "${EXISTING_FULL}/${sid}"; do
            if [ -d "${path}" ] && find "${path}" -name 'p0*.nii.gz' 2>/dev/null | grep -q . ; then
                already_done=1
                break
            fi
        done
        if [ "${already_done}" -eq 1 ]; then
            skip_existing=$((skip_existing + 1))
            continue
        fi
        # skip if tissue-only run already produced p0
        if find "${DERIV}/${sid}" -name 'p0*.nii.gz' 2>/dev/null | grep -q . ; then
            skip_done=$((skip_done + 1))
            continue
        fi
        QUEUE+=("${sub}|${ses}|${t1}")
    done
done
log "queue: total=${total} skip_existing=${skip_existing} skip_done=${skip_done} to_run=${#QUEUE[@]}"

run_one() {
    local sub="$1" ses="$2" t1="$3"
    local sid="${sub}_${ses}"
    local out_dir="${DERIV}/${sid}"
    local anat_dir; anat_dir="$(dirname "${t1}")"
    local t1_base; t1_base="$(basename "${t1}")"
    mkdir -p "${out_dir}"
    log "  ${sid}: T1Prep --no-surf"
    docker run --rm --gpus all \
        --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
        --user "$(id -u):$(id -g)" \
        -v "${anat_dir}:/input:ro" \
        -v "${out_dir}:/output:rw" \
        "${IMG}" \
            --out-dir /output --gz --no-surf "/input/${t1_base}" \
        >> "${LOG_DIR}/${sid}.log" 2>&1 \
        && log "    ${sid} done" \
        || log "    ${sid} FAILED — see ${LOG_DIR}/${sid}.log"
}

# Cohort dispatch with PARALLEL bound
running=0
for entry in "${QUEUE[@]}"; do
    IFS='|' read -r sub ses t1 <<< "${entry}"
    run_one "${sub}" "${ses}" "${t1}" &
    running=$((running + 1))
    if [ "${running}" -ge "${PARALLEL}" ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait
log "=== all queued subjects done ==="
log "next: extract tissue parquet via extract_t1prep_features.py + add to v3 ridge"
