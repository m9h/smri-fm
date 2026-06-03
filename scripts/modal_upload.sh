#!/usr/bin/env bash
# One-time upload of the CLS002 cohorts + FM checkpoints to the Modal Volume
# `fomo26-cls002`, matching the paths scripts/modal_cls002_sweep.py expects.
set -euo pipefail
V=fomo26-cls002
P=/data/datasets/fomo26
W=$P/weights

put() { echo ">>> $2"; modal volume put -f "$V" "$1" "$2"; }

# --- preprocessed cohorts (incl. split_cv5 + TEST_cv*.json) ------------------
put "$P/processed/CLS002_FOMO26_Infarct"     /processed/CLS002_FOMO26_Infarct
put "$P/processed/CLS002_FOMO26_Infarct_DWI" /processed/CLS002_FOMO26_Infarct_DWI

# --- FM checkpoints (only the files the converters read) ---------------------
put "$W/fomo60k_pkoutsouvelis/combined_regular-step=200000.ckpt" /weights/fomo60k_pkoutsouvelis/combined_regular-step=200000.ckpt
put "$W/triad/Triad-SwinB-MAE.pth"            /weights/triad/Triad-SwinB-MAE.pth
put "$W/anatcl/anatcl_global_fold0.pth"       /weights/anatcl/anatcl_global_fold0.pth
put "$W/mmunetvae/fomo25_mmunetvae_pretrained.ckpt" /weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt
put "$W/simclr3d/simclr_3d_brain_foundation.tar"    /weights/simclr3d/simclr_3d_brain_foundation.tar
put /home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt /weights/brainiac/BrainIAC.ckpt

# --- SIAM model dir (converter needs plans.json + fold_0/checkpoint_final) ---
S=/home/mhough/siam_params/v0.3/pred_DS108_LcsfP_Ano
D=/siam_params/pred_DS108_LcsfP_Ano
put "$S/plans.json"                  "$D/plans.json"
put "$S/dataset.json"                "$D/dataset.json"
put "$S/dataset_fingerprint.json"    "$D/dataset_fingerprint.json"
put "$S/fold_0/checkpoint_final.pth" "$D/fold_0/checkpoint_final.pth"

echo ">>> upload complete"
modal volume ls "$V" /
