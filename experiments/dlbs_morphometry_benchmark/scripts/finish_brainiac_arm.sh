#!/usr/bin/env bash
# Run this once brainiac-preproc has emitted all 60 <sid>.nii.gz files.
# It:
#   1) regenerates the pat_id/label CSV from whatever is present in the preproc dir
#   2) extracts BrainIAC 768-d embeddings inside brainiac-preproc-arm:dev
#   3) fits BrainIAC-only ridge (rung: FM-alone)
#   4) fits FS + T1Prep + BrainIAC concat ridge (rung: complementarity test)
#   5) re-weaves main.pnw → main.pdf
#
# Idempotent — if a step has already written its output, sklearn/pweave will
# still re-run it from scratch.
set -euo pipefail

BENCH=/home/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark
PREPROC_DIR=/data/datasets/smri-fm-cmp/brainiac-preproc/ds004856
BRAINIAC_SRC=/home/mhough/dev/BrainIAC/src
PAPER=/home/mhough/dev/T1Prep/comparison/paper
IMG=brainiac-preproc-arm:dev

echo "=== finish_brainiac_arm $(date -Iseconds) ==="
echo ">>> 1/5 make CSV from preproc dir"
python3 "${BENCH}/scripts/make_brainiac_csv.py" \
    --root_dir "${PREPROC_DIR}" \
    --out     "${BENCH}/results/brainiac_dlbs_csv.csv"

CSV_ROWS=$(($(wc -l < "${BENCH}/results/brainiac_dlbs_csv.csv") - 1))
echo "csv rows: ${CSV_ROWS}"
if [[ "${CSV_ROWS}" -lt 30 ]]; then
    echo "WARN: only ${CSV_ROWS} rows — was preproc interrupted?"
fi

echo ">>> 2/5 extract BrainIAC 768-d embeddings"
docker run --gpus all --rm \
  --user "$(id -u):$(id -g)" \
  -v "${BRAINIAC_SRC%/src}:/brainiac:ro" \
  -v "${BENCH}:/benchmark:rw" \
  -v "${PREPROC_DIR}:/scans:ro" \
  -e HOME=/tmp \
  --entrypoint python3 \
  "${IMG}" \
    /benchmark/scripts/extract_brainiac_embeddings.py \
    --brainiac_src /brainiac/src \
    --input_csv /benchmark/results/brainiac_dlbs_csv.csv \
    --root_dir /scans \
    --simclr_checkpoint /brainiac/src/checkpoints/BrainIAC.ckpt \
    --out_dir /benchmark/results \
    --batch_size 1

echo ">>> 3/5 ridge: BrainIAC 768-d alone (MedARC GroupKFold5)"
python3 "${BENCH}/scripts/fit_ridge_baseline.py" \
    --features "${BENCH}/results/brainiac_embeddings.parquet" \
    --tool brainiac_embed --value-col value \
    --cv-mode groupkfold5 \
    --out "${BENCH}/results/ridge_brainiac_embed.json"

echo ">>> 4/5 ridge: FS + T1Prep + BrainIAC concat (MedARC GroupKFold5)"
python3 "${BENCH}/scripts/fit_ridge_concat.py" \
    --source "${BENCH}/results/fastsurfer_features.parquet:aseg+DKT.VINN:volume_mm3" \
    --source "${BENCH}/results/t1prep_features.parquet:t1prep_thickness:thickness_mm" \
    --source "${BENCH}/results/brainiac_embeddings.parquet:brainiac_embed:value" \
    --cv-mode groupkfold5 \
    --out "${BENCH}/results/ridge_concat_fs_t1prep_brainiac.json"

echo ">>> 5/5 re-weave paper"
cd "${PAPER}" && make main.pdf

echo "=== DONE $(date -Iseconds) ==="
echo "results:"
for f in ridge_fs_asegdkt ridge_t1prep_thickness ridge_brainiac_embed \
         ridge_concat_fs_t1prep ridge_concat_fs_t1prep_brainiac; do
    p="${BENCH}/results/${f}.json"
    if [[ -f "${p}" ]]; then
        mae=$(python3 -c "import json; d=json.load(open('${p}')); print(f\"{d['zhang']['mae']:.2f}\")")
        r=$(python3 -c "import json; d=json.load(open('${p}')); print(f\"{d['zhang']['pearson_r']:.2f}\")")
        echo "  ${f}: MAE(Zhang)=${mae} yr  r=${r}"
    fi
done
