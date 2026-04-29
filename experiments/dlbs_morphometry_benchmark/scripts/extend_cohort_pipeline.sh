#!/usr/bin/env bash
# Resource-bounded extraction pipeline for additional DLBS subjects.
#
# Adds the next batch of multi-wave DLBS subjects to the morphometry
# benchmark — runs each through 5 stages idempotently:
#   A. BrainIAC preproc       (GPU, ~5 min/scan)
#   B. MedARC SynthSeg        (GPU, ~5 min/scan, batched per subject)
#   C. T1Prep                 (CPU mostly, ~30 min/scan @ 6 cores)
#   D. FastSurfer --seg_only  (GPU, ~10-30 min/scan)
#   E. FOMO25 AMAES batch     (GPU, ~30 sec/scan, post-loop)
#
# Each docker run is capped at --cpus 6 --memory 30g + nice/ionice so
# Spark stays usable for other work.
#
# Stage-level idempotence: skips any subject/session that already has
# the marker output. Ctrl-C-safe and resumable.
#
# Usage:
#   bash extend_cohort_pipeline.sh                  # next 23 multi-wave subjects
#   COHORT_SIZE=46 bash extend_cohort_pipeline.sh   # next 46
#   bash extend_cohort_pipeline.sh sub-12 sub-1226  # explicit subject list
set -uo pipefail
# NOTE: deliberately NOT using `set -e` — individual stages handle their
# own failures via `|| log "FAILED..."` and we want the loop to continue
# to the next stage / subject even if one fails.

EXISTING_23=(sub-1003 sub-1007 sub-1013 sub-1022 sub-1023 sub-103 sub-1031
             sub-1045 sub-1054 sub-1058 sub-1084 sub-1093 sub-1139 sub-1141
             sub-1146 sub-1149 sub-1153 sub-1157 sub-1172 sub-1175 sub-1183
             sub-1200 sub-1220)

RAW_ROOT="/data/raw/openneuro/ds004856"
DERIV="/data/datasets/smri-fm-cmp"
LOG_DIR="${DERIV}/_logs/extend_cohort"
LICENSE="${HOME}/license.txt"
FS_CHECKPOINTS="${HOME}/fs_checkpoints"
mkdir -p "${LOG_DIR}"

CPU_LIMIT="${CPU_LIMIT:-6}"
MEM_LIMIT="${MEM_LIMIT:-30g}"
NICE_PREFIX="nice -n 19 ionice -c3"

log() { printf '[%s] %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "${LOG_DIR}/queue.log" >&2; }

# 1) Pick the cohort
if [ "$#" -gt 0 ]; then
    COHORT=("$@")
    log "explicit cohort: ${#COHORT[@]} subjects"
else
    COHORT_SIZE="${COHORT_SIZE:-23}"
    log "discovering next ${COHORT_SIZE} multi-wave subjects..."
    COHORT=()
    for s in $(ls "${RAW_ROOT}" | grep '^sub-' | sort); do
        in_existing=0
        for e in "${EXISTING_23[@]}"; do [ "$s" = "$e" ] && in_existing=1 && break; done
        [ "${in_existing}" -eq 1 ] && continue
        t1s=$(find "${RAW_ROOT}/${s}" -name '*acq-MPRAGE_run-1_T1w.nii.gz' 2>/dev/null | wc -l)
        [ "$t1s" -lt 3 ] && continue
        COHORT+=("$s")
        [ "${#COHORT[@]}" -ge "${COHORT_SIZE}" ] && break
    done
fi
log "cohort: ${COHORT[*]}"

# 2) Per-stage helpers, each idempotent

run_brainiac() {
    local sub="$1" ses="$2"
    local sid="${sub}_${ses}"
    local out_dir="${DERIV}/brainiac-preproc/ds004856"
    local out_file="${out_dir}/${sid}.nii.gz"
    [ -f "${out_file}" ] && return 0

    local t1
    t1=$(find "${RAW_ROOT}/${sub}/${ses}/anat" -name '*acq-MPRAGE_run-1_T1w.nii.gz' 2>/dev/null | head -1)
    [ -z "${t1}" ] && return 0

    local anat_dir t1_base
    anat_dir="$(dirname "${t1}")"
    t1_base="$(basename "${t1}")"

    log "  ${sid}: BrainIAC preproc (GPU)"
    mkdir -p "${out_dir}"
    ${NICE_PREFIX} docker run --rm --gpus all \
        --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
        --user "$(id -u):$(id -g)" \
        -v "${anat_dir}:/input:ro" \
        -v "${out_dir}:/output:rw" \
        -e HOME=/tmp \
        ghcr.io/m9h/brainiac-preproc-arm:latest \
            --input "/input/${t1_base}" \
            --pat_id "${sid}" \
            --out_dir /output \
            --temp_img /opt/brainiac_preproc/temp_head.nii.gz \
        >> "${LOG_DIR}/${sid}.brainiac.log" 2>&1 \
        || log "    BrainIAC FAILED — see ${LOG_DIR}/${sid}.brainiac.log"
}

run_medarc_synthseg() {
    local sub="$1"
    local sub_dir="${DERIV}/medarc-smri-fm/ds004856/${sub}"
    mkdir -p "${sub_dir}/synthseg" "${sub_dir}/logs"

    local synthseg_n=0
    if [ -d "${sub_dir}/synthseg" ]; then
        synthseg_n=$(find "${sub_dir}/synthseg" -name '*_desc-synthseg_volumes.tsv' 2>/dev/null | wc -l || echo 0)
    fi
    [ "${synthseg_n}" -ge 3 ] && return 0

    log "  ${sub}: MedARC SynthSeg (GPU)"
    local t1bids="/tmp/medarc_t1only_${sub}"
    rm -rf "${t1bids}"
    mkdir -p "${t1bids}/${sub}"
    for f in dataset_description.json participants.tsv participants.json README; do
        [ -e "${RAW_ROOT}/${f}" ] && ln -sf "${RAW_ROOT}/${f}" "${t1bids}/${f}"
    done
    for ses_dir in "${RAW_ROOT}/${sub}"/ses-*; do
        [ -d "${ses_dir}" ] || continue
        local sesname; sesname="$(basename "${ses_dir}")"
        mkdir -p "${t1bids}/${sub}/${sesname}/anat"
        for ext in nii.gz json; do
            local t1f="${ses_dir}/anat/${sub}_${sesname}_acq-MPRAGE_run-1_T1w.${ext}"
            [ -e "${t1f}" ] && ln -sf "${t1f}" "${t1bids}/${sub}/${sesname}/anat/"
        done
    done

    # NB: the container's bundled pipeline.py looks up `/opt/venv/bin/python`
    # for the SynthSeg subprocess call, but that path doesn't exist in the
    # currently-published image (regression vs the build the original 23
    # were processed with). Mount the patched pipeline.py over it. The
    # patched file lives at /tmp/pipeline_patched.py (saved from the
    # original 23-subject run) and was tested successfully then.
    local patched_pipeline="/tmp/pipeline_patched.py"
    local patch_args=()
    if [ -f "${patched_pipeline}" ]; then
        patch_args+=(-v "${patched_pipeline}:/app/smri-fm/preprocessing/pipeline.py:ro")
    else
        log "    WARN: ${patched_pipeline} not found; SynthSeg may hit /opt/venv/bin/python error"
    fi

    ${NICE_PREFIX} docker run --rm --gpus all \
        --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
        --user "$(id -u):$(id -g)" \
        -v "${t1bids}:/bids:ro" \
        -v "${RAW_ROOT}:${RAW_ROOT}:ro" \
        -v "${sub_dir}:/output:rw" \
        -v "${LICENSE}:/usr/lib/freesurfer/license.txt:ro" \
        "${patch_args[@]}" \
        -e HOME=/tmp -e TF_USE_LEGACY_KERAS=1 \
        --entrypoint python3 \
        ghcr.io/m9h/medarc-smri-fm:latest \
        /app/smri-fm/preprocessing/pipeline.py \
            --bids /bids --subject "${sub}" \
            --output /output --log_dir /output/logs \
            --n_workers 1 --itk_threads 4 \
            --synthseg --synthseg_output /output/synthseg --synthseg_workers 1 \
        >> "${LOG_DIR}/${sub}.synthseg.log" 2>&1 \
        || log "    SynthSeg FAILED — see ${LOG_DIR}/${sub}.synthseg.log"
}

run_t1prep() {
    local sub="$1" ses="$2"
    local sid="${sub}_${ses}"
    local out_dir="${DERIV}/t1prep/ds004856/${sub}/${ses}"
    local marker="${out_dir}/done.txt"
    [ -f "${marker}" ] && return 0

    local t1
    t1=$(find "${RAW_ROOT}/${sub}/${ses}/anat" -name '*acq-MPRAGE_run-1_T1w.nii.gz' 2>/dev/null | head -1)
    [ -z "${t1}" ] && return 0

    local anat_dir t1_base
    anat_dir="$(dirname "${t1}")"
    t1_base="$(basename "${t1}")"

    log "  ${sid}: T1Prep (CPU)"
    mkdir -p "${out_dir}"
    ${NICE_PREFIX} docker run --rm --gpus all \
        --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
        --user "$(id -u):$(id -g)" \
        -v "${anat_dir}:/input:ro" \
        -v "${out_dir}:/output:rw" \
        ghcr.io/m9h/t1prep-arm:latest \
            --out-dir /output --gz "/input/${t1_base}" \
        >> "${LOG_DIR}/${sid}.t1prep.log" 2>&1 \
        && touch "${marker}" \
        || log "    T1Prep FAILED — see ${LOG_DIR}/${sid}.t1prep.log"
}

run_fastsurfer() {
    local sub="$1" ses="$2"
    local sid="${sub}_${ses}"
    local fs_root="${DERIV}/fastsurfer/ds004856"
    local marker="${fs_root}/${sid}/mri/aparc.DKTatlas+aseg.deep.mgz"
    [ -f "${marker}" ] && return 0

    local t1
    t1=$(find "${RAW_ROOT}/${sub}/${ses}/anat" -name '*acq-MPRAGE_run-1_T1w.nii.gz' 2>/dev/null | head -1)
    [ -z "${t1}" ] && return 0

    local anat_dir t1_base
    anat_dir="$(dirname "${t1}")"
    t1_base="$(basename "${t1}")"

    log "  ${sid}: FastSurfer --seg_only (GPU)"
    mkdir -p "${fs_root}"
    ${NICE_PREFIX} docker run --rm --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
        --user "$(id -u):$(id -g)" \
        -v "${anat_dir}:/data:ro" \
        -v "${fs_root}:/output:rw" \
        -v "${LICENSE}:/opt/FastSurfer/license.txt:ro" \
        -v "${FS_CHECKPOINTS}:/opt/FastSurfer/checkpoints:ro" \
        ghcr.io/m9h/fastsurfer-arm:latest \
        ./run_fastsurfer.sh \
            --t1 "/data/${t1_base}" --sid "${sid}" --sd /output \
            --seg_only --parallel \
            --fs_license /opt/FastSurfer/license.txt \
        >> "${LOG_DIR}/${sid}.fs.log" 2>&1 \
        || log "    FastSurfer FAILED — see ${LOG_DIR}/${sid}.fs.log"
}

# 3) Per-subject orchestration
process_subject() {
    local sub="$1"
    log "=== ${sub} ==="
    for ses in ses-wave1 ses-wave2 ses-wave3; do
        run_brainiac "${sub}" "${ses}"
    done
    run_medarc_synthseg "${sub}"
    for ses in ses-wave1 ses-wave2 ses-wave3; do
        run_t1prep "${sub}" "${ses}"
        run_fastsurfer "${sub}" "${ses}"
    done
    log "  ${sub} all stages attempted"
}

# 4) Cohort loop
for sub in "${COHORT[@]}"; do
    process_subject "${sub}"
done

# 5) Batch FOMO25 AMAES at the end (one container call covers all 60+ scans)
log "=== batch FOMO25 AMAES extraction ==="
EXTENDED_CSV="/tmp/dlbs_extended_cohort.csv"
echo 'pat_id,label,dataset' > "${EXTENDED_CSV}"
for sub in "${COHORT[@]}"; do
    for ses in ses-wave1 ses-wave2 ses-wave3; do
        if [ -f "${DERIV}/brainiac-preproc/ds004856/${sub}_${ses}.nii.gz" ]; then
            wave="${ses#ses-}"          # ses-wave1 → wave1
            wave_upper="${wave^}"        # wave1 → Wave1 (oddly the participants.tsv uses W1, W2, W3)
            # Map ses-waveN → AgeMRI_W{N}
            n="${ses#ses-wave}"
            col="AgeMRI_W${n}"
            age=$(awk -v s="${sub}" -v c="${col}" '
                BEGIN{FS="\t"; getline hdr; n=split(hdr,h,"\t"); for(i=1;i<=n;i++) col[h[i]]=i}
                $col["participant_id"]==s {print $col[c]}' \
                "${RAW_ROOT}/participants.tsv" 2>/dev/null | head -1)
            [ -n "${age}" ] && [ "${age}" != "n/a" ] && \
                echo "${sub}_${ses},${age},DLBS" >> "${EXTENDED_CSV}"
        fi
    done
done
log "extended CSV: $(wc -l < "${EXTENDED_CSV}") rows"

CSV="${EXTENDED_CSV}" \
OUT="${DERIV}/fomo-embeds/ds004856_extended" \
${NICE_PREFIX} bash /home/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/scripts/run_fomo25_extraction_spark.sh \
    >> "${LOG_DIR}/fomo25.log" 2>&1 \
    || log "  FOMO25 batch FAILED — see ${LOG_DIR}/fomo25.log"

log "=== pipeline complete ==="
log "next: extract SynthSeg parquet + run framing-2 ridge"
