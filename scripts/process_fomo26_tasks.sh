#!/usr/bin/env bash
# Convert the downloaded FOMO26 task data into Asparagus .pt tensors, run inside
# an NVIDIA PyTorch (NGC) container — sidesteps the aarch64 uv throttling and the
# editable-install/setuptools_scm trap (we use PYTHONPATH, not `pip install -e`).
#
# Expects these to be set (the docker wrapper passes them):
#   ASPARAGUS_SOURCE  /fomo26/source     (Task_1..4 already unzipped as Task_N/Task_N; PPMR zip + Task_5_extract.py present)
#   ASPARAGUS_DATA    /fomo26/processed
#   ASPARAGUS_RAW_LABELS /fomo26/raw_labels
#   PYTHONPATH includes .../third_party/asparagus_preprocessing
#
# asp_process is idempotent (skips cases whose .pt already exists) and its
# postprocess step writes split_80_10_10.json + TEST_80_10_10.json, so no
# separate asp_split is needed for the finetune configs.
set -euo pipefail

echo ">>> installing preprocessing deps (torch+numpy already in image)"
pip install -q --no-cache-dir --root-user-action=ignore \
  nibabel scikit-image scikit-learn pandas python-dotenv openpyxl xlrd pillow scipy 2>&1 | tail -2 || true

proc() {  # proc <PREFIX> <human label>
  echo ">>> asp_process --dataset $1  ($2)"
  python -m asparagus_preprocessing.scripts.process --dataset "$1" --save_as_tensor --num_workers 8
}

# --- Task 1 / CLS002 infarct -------------------------------------------------
# Two modules match prefix "CLS002" (canonical + a MedARC _CUSTOM modality
# variant) and find_module's os.walk order is nondeterministic, so invoke the
# canonical module directly to guarantee output lands in CLS002_FOMO26_Infarct
# (the task name the finetune config expects). _CUSTOM is only for modality
# experiments; remove any stray output from an ambiguous earlier run.
if [ -d "$ASPARAGUS_SOURCE/Task_1/Task_1" ]; then
  rm -rf "$ASPARAGUS_DATA/CLS002_FOMO26_Infarct_CUSTOM" "$ASPARAGUS_RAW_LABELS/CLS002_FOMO26_Infarct_CUSTOM"
  echo ">>> CLS002 infarct (canonical module, direct import)"
  python -c "from asparagus_preprocessing.datasets_classification.CLS002_FOMO26_Infarct import main; main(processes=8, save_as_tensor=True)"
fi

# --- Task 2 / SEG009 meningioma ---------------------------------------------
[ -d "$ASPARAGUS_SOURCE/Task_2/Task_2" ] && proc SEG009 "Task 2 meningioma seg"

# --- Task 3 / REGR002 brain age ---------------------------------------------
[ -d "$ASPARAGUS_SOURCE/Task_3/Task_3" ] && proc REGR002 "Task 3 brain-age reg"

# --- Task 4 / SEG010 trigeminal neuralgia -----------------------------------
[ -d "$ASPARAGUS_SOURCE/Task_4" ] && proc SEG010 "Task 4 trigeminal-neuralgia seg"

# --- Task 5 / CLS003 polymicrogyria -----------------------------------------
# Build Task_5 from the Zhang PPMR archive first (coronal JPG -> NIfTI), then process.
if [ ! -d "$ASPARAGUS_SOURCE/Task_5" ] && [ -f "$ASPARAGUS_SOURCE/Zhang_Lingfeng_2022_PPMR_Dataset.zip" ]; then
  echo ">>> building Task_5 from Zhang PPMR (Task_5_extract.py)"
  ( cd "$ASPARAGUS_SOURCE" && python Task_5_extract.py )
fi
[ -d "$ASPARAGUS_SOURCE/Task_5" ] && proc CLS003 "Task 5 polymicrogyria cls"

echo
echo ">>> SUMMARY (.pt per processed dataset)"
for d in "$ASPARAGUS_DATA"/*/; do
  [ -d "$d" ] || continue
  n=$(find "$d" -name "*.pt" 2>/dev/null | wc -l)
  s=$( [ -f "$d/split_80_10_10.json" ] && echo "split:yes" || echo "split:NO" )
  echo "    $(basename "$d"): ${n} .pt  ${s}"
done
echo ">>> done"
