#!/usr/bin/env bash
# Resource-bounded extraction pipeline for additional DLBS subjects.
#
# Adds the next batch of multi-wave DLBS subjects to the morphometry
# benchmark — runs BrainIAC preproc → SynthSeg (MedARC pipeline.py) →
# FOMO25 AMAES → T1Prep → FastSurfer per scan, idempotently. Designed
# to leave Spark usable for other work: caps containers at 6 of 20
# cores + 30 GB RAM, runs at nice -n 19 / ionice -c3, processes
# subjects serially.
#
# Stops cleanly at any time (Ctrl-C or kill); stage-level outputs are
# checked before re-running so resumption is automatic.
#
# Usage:
#   bash extend_cohort_pipeline.sh                    # next 23 subjects
#   COHORT_SIZE=46 bash extend_cohort_pipeline.sh     # next 46
#   bash extend_cohort_pipeline.sh sub-12 sub-1226    # explicit subject list
#
# Outputs (NFS, downstream-readable):
#   /data/datasets/smri-fm-cmp/brainiac-preproc/ds004856/<sub>_<ses>.nii.gz
#   /data/datasets/smri-fm-cmp/medarc-smri-fm/ds004856/<sub>/...synthseg.vol.csv
#   /data/datasets/smri-fm-cmp/fomo-embeds/ds004856/extended/...
#   /data/datasets/smri-fm-cmp/t1prep/ds004856/<sub>/<ses>/...
#   /data/datasets/smri-fm-cmp/fastsurfer/ds004856/<sub>_<ses>/...
set -euo pipefail

EXISTING_23=(sub-1003 sub-1007 sub-1013 sub-1022 sub-1023 sub-103 sub-1031
             sub-1045 sub-1054 sub-1058 sub-1084 sub-1093 sub-1139 sub-1141
             sub-1146 sub-1149 sub-1153 sub-1157 sub-1172 sub-1175 sub-1183
             sub-1200 sub-1220)

RAW_ROOT="/data/raw/openneuro/ds004856"
DERIV="/data/datasets/smri-fm-cmp"
LOG_DIR="${DERIV}/_logs/extend_cohort"
mkdir -p "${LOG_DIR}"

# Resource caps for any docker run we issue.
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

# 2) Per-subject pipeline. Each stage is idempotent.
process_subject() {
    local sub="$1"
    log "=== ${sub} ==="

    for ses in ses-wave1 ses-wave2 ses-wave3; do
        local sid="${sub}_${ses}"
        local t1
        t1=$(find "${RAW_ROOT}/${sub}/${ses}/anat" -name '*acq-MPRAGE_run-1_T1w.nii.gz' 2>/dev/null | head -1)
        if [ -z "${t1}" ]; then
            log "  ${sid}: no T1, skip"
            continue
        fi

        # Stage A — BrainIAC preproc (GPU, ~5 min/scan)
        local preproc_out="${DERIV}/brainiac-preproc/ds004856/${sid}.nii.gz"
        if [ ! -f "${preproc_out}" ]; then
            log "  ${sid}: BrainIAC preproc (GPU)"
            mkdir -p "${DERIV}/brainiac-preproc/ds004856"
            ${NICE_PREFIX} docker run --rm --gpus all \
                --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
                --user "$(id -u):$(id -g)" \
                -v "$(dirname "${t1}"):/in:ro" \
                -v "${DERIV}/brainiac-preproc/ds004856:/out:rw" \
                ghcr.io/m9h/brainiac-preproc-arm:latest \
                "/in/$(basename "${t1}")" "/out/${sid}.nii.gz" \
                >> "${LOG_DIR}/${sid}.log" 2>&1 || \
                log "    BrainIAC preproc FAILED — see ${LOG_DIR}/${sid}.log"
        fi

        # Stage B — FOMO25 AMAES extraction (GPU, ~30 sec/scan)
        # Batched at end-of-cohort, not per-subject — see post-loop.

        # Stage B' — MedARC pipeline.py for SynthSeg volumes (GPU, ~5 min/scan)
        # Done at the end of this subject's loop, once across all 3 sessions.
    done

    # Run MedARC pipeline.py for this subject (3 sessions in one container call)
    local synthseg_n
    synthseg_n=$(find "${DERIV}/medarc-smri-fm/ds004856/${sub}/synthseg" \
        -name '*_desc-synthseg_volumes.tsv' 2>/dev/null | wc -l)
    if [ "${synthseg_n}" -lt 3 ]; then
        log "  ${sub}: MedARC pipeline.py SynthSeg (GPU)"
        mkdir -p "${DERIV}/medarc-smri-fm/ds004856/${sub}/synthseg" \
                 "${DERIV}/medarc-smri-fm/ds004856/${sub}/logs"
        # Build T1-only BIDS for this subject (the pattern used by
        # run_medarc_pipeline_all.sh; pipeline.py expects a BIDS-shaped tree).
        local t1bids="/tmp/medarc_t1only_${sub}"
        rm -rf "${t1bids}"
        mkdir -p "${t1bids}/${sub}"
        for f in dataset_description.json participants.tsv participants.json README; do
            [ -e "${RAW_ROOT}/${f}" ] && ln -sf "${RAW_ROOT}/${f}" "${t1bids}/${f}"
        done
        for ses_dir in "${RAW_ROOT}/${sub}"/ses-*; do
            [ -d "${ses_dir}" ] || continue
            local sesname
            sesname="$(basename "${ses_dir}")"
            mkdir -p "${t1bids}/${sub}/${sesname}/anat"
            for ext in nii.gz json; do
                local t1f="${ses_dir}/anat/${sub}_${sesname}_acq-MPRAGE_run-1_T1w.${ext}"
                [ -e "${t1f}" ] && ln -sf "${t1f}" "${t1bids}/${sub}/${sesname}/anat/"
            done
        done

        ${NICE_PREFIX} docker run --rm --gpus all \
            --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
            --user "$(id -u):$(id -g)" \
            -v "${t1bids}:/bids:ro" \
            -v "${RAW_ROOT}:${RAW_ROOT}:ro" \
            -v "${DERIV}/medarc-smri-fm/ds004856/${sub}:/output:rw" \
            -v "${HOME}/license.txt:/usr/lib/freesurfer/license.txt:ro" \
            -e HOME=/tmp -e TF_USE_LEGACY_KERAS=1 \
            --entrypoint python3 \
            ghcr.io/m9h/medarc-smri-fm:latest \
            /app/smri-fm/preprocessing/pipeline.py \
                --bids /bids --subject "${sub}" \
                --output /output --log_dir /output/logs \
                --n_workers 1 --itk_threads 4 \
                --synthseg --synthseg_output /output/synthseg --synthseg_workers 1 \
            >> "${LOG_DIR}/${sub}.synthseg.log" 2>&1 || \
            log "    MedARC SynthSeg FAILED — see ${LOG_DIR}/${sub}.synthseg.log"
    fi

    for ses in ses-wave1 ses-wave2 ses-wave3; do
        local sid="${sub}_${ses}"
        local t1
        t1=$(find "${RAW_ROOT}/${sub}/${ses}/anat" -name '*acq-MPRAGE_run-1_T1w.nii.gz' 2>/dev/null | head -1)
        [ -z "${t1}" ] && continue

        # Stage C — T1Prep (CPU, ~30 min/scan @ 6 cores)
        local t1prep_done="${DERIV}/t1prep/ds004856/${sub}/${ses}/done.txt"
        if [ ! -f "${t1prep_done}" ]; then
            log "  ${sid}: T1Prep (CPU)"
            mkdir -p "${DERIV}/t1prep/ds004856/${sub}/${ses}"
            ${NICE_PREFIX} docker run --rm --gpus all \
                --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
                --user "$(id -u):$(id -g)" \
                -v "$(dirname "${t1}"):/in:ro" \
                -v "${DERIV}/t1prep/ds004856/${sub}/${ses}:/out:rw" \
                ghcr.io/m9h/t1prep-arm:latest \
                T1Prep --out-dir /out "/in/$(basename "${t1}")" \
                >> "${LOG_DIR}/${sid}.t1prep.log" 2>&1 \
                && touch "${t1prep_done}" \
                || log "    T1Prep FAILED — see ${LOG_DIR}/${sid}.t1prep.log"
        fi

        # Stage D — FastSurfer (--seg_only is fastest, sufficient for our ridge)
        local fs_done="${DERIV}/fastsurfer/ds004856/${sid}/scripts/recon-surf.done"
        local fs_seg="${DERIV}/fastsurfer/ds004856/${sid}/mri/aparc.DKTatlas+aseg.deep.mgz"
        if [ ! -f "${fs_seg}" ]; then
            log "  ${sid}: FastSurfer --seg_only (CPU+GPU)"
            mkdir -p "${DERIV}/fastsurfer/ds004856"
            ${NICE_PREFIX} docker run --rm --gpus all \
                --cpus "${CPU_LIMIT}" --memory "${MEM_LIMIT}" \
                --user "$(id -u):$(id -g)" \
                -v "$(dirname "${t1}"):/in:ro" \
                -v "${DERIV}/fastsurfer/ds004856:/out:rw" \
                -v "${HOME}/license.txt:/fs_license.txt:ro" \
                ghcr.io/m9h/fastsurfer-arm:latest \
                --t1 "/in/$(basename "${t1}")" --sid "${sid}" --sd /out \
                --seg_only --fs_license /fs_license.txt --threads "${CPU_LIMIT}" \
                >> "${LOG_DIR}/${sid}.fs.log" 2>&1 || \
                log "    FastSurfer FAILED — see ${LOG_DIR}/${sid}.fs.log"
        fi
    done
    log "  ${sub} all sessions done"
}

# 3) Run the cohort serially
for sub in "${COHORT[@]}"; do
    process_subject "${sub}"
done

# 4) Batch FOMO25 AMAES at the end (one container invocation handles all 60+ scans)
log "=== batch FOMO25 AMAES extraction ==="
EXTENDED_CSV="/tmp/dlbs_extended_cohort.csv"
echo 'pat_id,label,dataset' > "${EXTENDED_CSV}"
for sub in "${COHORT[@]}"; do
    for ses in ses-wave1 ses-wave2 ses-wave3; do
        if [ -f "${DERIV}/brainiac-preproc/ds004856/${sub}_${ses}.nii.gz" ]; then
            age=$(awk -v s="${sub}" -v c="AgeMRI_${ses#ses-}" '
                BEGIN{FS="\t"; getline; for(i=1;i<=NF;i++) col[$i]=i}
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
    >> "${LOG_DIR}/fomo25.log" 2>&1 || \
    log "  FOMO25 batch FAILED — see ${LOG_DIR}/fomo25.log"

log "=== pipeline complete ==="
log "next steps: extract SynthSeg volumes from MedARC pipeline outputs,"
log "  then run framing-2 ridge: train on original 60, test on new ~69."
