#!/usr/bin/env bash
# T1Prep longitudinal runner, three-stage so the surface step actually
# works. Upstream process_longitudinal.sh has too many quirks (BOLD unbound
# under set -u, readonly-mount mishandling, shell line-parsing) — we do it
# ourselves.
#
# Stages:
#   1) Stage all T1s for the subject into a writable scratch dir.
#   2) Run realign_longitudinal.py (--save-resampled writes
#      <stem>_desc-realigned.nii.gz alongside the inputs).
#   3) Run T1Prep cross-sectional on wave 1 ONLY (so CAT-Surface produces
#      a valid lh/rh.central.*.gii to reuse in stage 4).
#   4) Run T1Prep --long-data on each wave, with --initial-surface pointing
#      at the wave-1 cross-sectional surface. This is what T1Prep
#      `scripts/T1Prep:459` expects for longitudinal: "Use initial surface
#      and skip Marching Cubes for longitudinal processing". Without this
#      the surface step fails every time.
#
# Usage:
#   t1prep_longitudinal.sh DATASET SUBJECT
#
# Example:
#   t1prep_longitudinal.sh ds004856 sub-1003
set -uo pipefail
# NB: not set -e. Surface step failures on a single wave should not kill
# the whole batch; we record and continue.

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

OUT_XS_ROOT="/data/datasets/smri-fm-cmp/t1prep/ds004856"  # cross-sectional already here
rm -rf "$SCRATCH" && mkdir -p "$SCRATCH" "$OUT_ROOT"
{
  echo "=== t1prep_longitudinal ${SUBJECT} $(date -Iseconds) ==="

  # Stage T1s into a flat scratch dir (we need a writable mount; T1Prep
  # writes alongside the inputs regardless of --out-dir).
  mapfile -t T1S < <(find "$SRC" -name "${SUBJECT}_ses-wave*_acq-MPRAGE_run-1_T1w.nii.gz" | sort)
  echo "timepoints: ${#T1S[@]}"
  CPATHS=()
  for p in "${T1S[@]}"; do
    b=$(basename "$p")
    cp "$p" "${SCRATCH}/${b}"
    CPATHS+=("/scratch/${b}")
  done

  # Stage 2: rigid realignment.
  # t1prep package not pip-installed in the venv — it's under
  # /opt/T1Prep/src. Use PYTHONPATH to make `python -m t1prep.X` work.
  echo ">>> realign_longitudinal"
  docker run --gpus all --rm \
    --user "$(id -u):$(id -g)" \
    --entrypoint /opt/T1Prep/env/bin/python \
    -e PYTHONPATH=/opt/T1Prep/src \
    -v "${SCRATCH}:/scratch" \
    t1prep-arm:latest \
    -m t1prep.realign_longitudinal \
      --inputs "${CPATHS[@]}" \
      --out-dir /scratch \
      --save-resampled 2>&1 | tail -10

  echo "---scratch after realign---"
  ls "${SCRATCH}" 2>&1

  # Stage 3: reuse a pre-computed cross-sectional wave-1 surface if
  # available (from the smoke_all_subject.sh cross-sectional runs). If not,
  # run T1Prep cross-sectional on wave 1 now to produce lh/rh.central.*.gii.
  # This is what T1Prep's scripts/T1Prep:459 expects for --long-data: a
  # valid initial surface, otherwise Marching Cubes runs on the realigned
  # volume and fails.
  FIRST_T1="${T1S[0]}"
  FIRST_SES=$(basename "$(dirname "$(dirname "${FIRST_T1}")")")   # ses-wave1
  FIRST_SID="${SUBJECT}_${FIRST_SES}"
  FIRST_B=$(basename "${FIRST_T1}")
  XS_SURF_DIR="${OUT_XS_ROOT}/${FIRST_SID}/surf"
  if [[ ! -f "${XS_SURF_DIR}/lh.central.${FIRST_B%.nii.gz}.gii" ]]; then
    echo ">>> T1Prep cross-sectional wave 1 (needed for initial surface)"
    mkdir -p "${OUT_XS_ROOT}/${FIRST_SID}"
    ANAT_DIR=$(dirname "${FIRST_T1}")
    docker run --gpus all --rm \
      --user "$(id -u):$(id -g)" \
      -v "${ANAT_DIR}:/input:ro" \
      -v "${OUT_XS_ROOT}/${FIRST_SID}:/output" \
      t1prep-arm:latest \
        --out-dir /output --gz "/input/${FIRST_B}" 2>&1 | tail -3
  else
    echo ">>> reusing cross-sectional wave-1 surface at ${XS_SURF_DIR}"
  fi

  INIT_LH="${XS_SURF_DIR}/lh.central.${FIRST_B%.nii.gz}.gii"
  INIT_RH="${XS_SURF_DIR}/rh.central.${FIRST_B%.nii.gz}.gii"
  if [[ ! -f "${INIT_LH}" ]]; then
    echo "WARN: wave-1 surface still missing, skipping --initial-surface"
    INIT_LH=""
  fi

  # Stage 4: T1Prep --long-data per timepoint with --initial-surface.
  for p in "${T1S[@]}"; do
    SES=$(basename "$(dirname "$(dirname "$p")")")
    SID="${SUBJECT}_${SES}"
    B=$(basename "$p")
    OUT="${OUT_ROOT}/${SID}"
    mkdir -p "$OUT"

    # --long-data takes a FILE path (help text is misleading; see
    # scripts/T1Prep:867 `input="$long_data"`).
    REALIGNED_NAME="${B%.nii.gz}_desc-realigned.nii.gz"
    echo ">>> T1Prep long ${SID}"

    DOCKER_ARGS=(
      --gpus all --rm
      --user "$(id -u):$(id -g)"
      -v "${SCRATCH}:/long_data:ro"
      -v "${OUT}:/output"
    )
    T1PREP_ARGS=(
      --out-dir /output --gz
      --long-data "/long_data/${REALIGNED_NAME}"
    )
    if [[ -n "${INIT_LH}" ]]; then
      DOCKER_ARGS+=(-v "${XS_SURF_DIR}:/xs_surf:ro")
      T1PREP_ARGS+=(--initial-surface "/xs_surf/$(basename "${INIT_LH}")")
    fi

    docker run "${DOCKER_ARGS[@]}" t1prep-arm:latest \
      "${T1PREP_ARGS[@]}" "/long_data/${B}" 2>&1 | tail -5
  done

  echo "=== t1prep_longitudinal ${SUBJECT} done $(date -Iseconds) ==="
} 2>&1 | tee "$LOG" 