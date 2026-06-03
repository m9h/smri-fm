#!/usr/bin/env bash
# Preprocess the FOMO26 Task_1 cohort as a DWI-ONLY (1-channel) variant, so the
# diffusion-modality probe is isolated from FLAIR/ADC. Uses the MedARC _CUSTOM
# dataset module with modalities=['dwi'] (dwi maps to dwi_b1000.nii.gz), writing
# a separate task dir CLS002_FOMO26_Infarct_DWI with per-subject dwi.pt tensors.
#
# Runs inside the NGC PyTorch container (same env contract as
# process_fomo26_tasks.sh). Expected mounts (set by the docker wrapper):
#   /workspace/smri-fm  <- repo root
#   /fomo26             <- /data/datasets/fomo26
set -euo pipefail

REPO=/workspace/smri-fm
export ASPARAGUS_SOURCE=/fomo26/source
export ASPARAGUS_DATA=/fomo26/processed
export ASPARAGUS_RAW_LABELS=/fomo26/raw_labels
export PYTHONPATH=$REPO/third_party/asparagus_preprocessing:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$ASPARAGUS_DATA" "$ASPARAGUS_RAW_LABELS"

echo ">>> installing preprocessing deps"
pip install -q --no-cache-dir --root-user-action=ignore \
  nibabel scikit-image scikit-learn pandas python-dotenv openpyxl xlrd pillow scipy 2>&1 | tail -2 || true

echo ">>> preprocessing CLS002 DWI-only (modalities=['dwi'] -> CLS002_FOMO26_Infarct_DWI)"
python -c "from asparagus_preprocessing.datasets_classification.CLS002_FOMO26_Infarct_CUSTOM import main; main(modalities=['dwi'], task_name='CLS002_FOMO26_Infarct_DWI', processes=8, save_as_tensor=True)"

echo ">>> SUMMARY"
d=$ASPARAGUS_DATA/CLS002_FOMO26_Infarct_DWI
n=$(find "$d" -name '*.pt' 2>/dev/null | wc -l)
echo "    CLS002_FOMO26_Infarct_DWI: ${n} .pt  $( [ -f "$d/split_80_10_10.json" ] && echo split:yes || echo split:NO )"
echo ">>> done"
