#!/usr/bin/env bash
# In-container orchestrator for the FOMO26 CLS002 (infarct) diffusion-classification
# leaderboard: every FM arm x {full flair+adc+dwi stack, dwi-only} x rotating CV fold.
# Installs deps once and converts each arm's checkpoint once, then loops the runs.
# Per-run failures are caught and logged so one bad arm doesn't sink the sweep.
#
# Configurable via env (space-separated lists):
#   ARMS   default: all 7
#   TASKS  default: "CLS002_FOMO26_Infarct CLS002_FOMO26_Infarct_DWI"
#   FOLDS  default: "0 1 2 3 4"
#   DEBUG  default: 0 (real). 1 = smoke.
#   EPOCHS default: 50
set -uo pipefail

REPO=/workspace/smri-fm
export ASPARAGUS_CONFIGS=$REPO/third_party/asparagus/configs
export ASPARAGUS_FINETUNE_CONFIGS=$REPO/src/asparagus_bridge/configs
export ASPARAGUS_DATA=/fomo26/processed
export ASPARAGUS_RAW_LABELS=/fomo26/raw_labels
export ASPARAGUS_MODELS=/fomo26/models
export ASPARAGUS_RESULTS=/fomo26/results
export PYTHONPATH=$REPO/third_party/asparagus:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=offline WANDB_DISABLED=true HYDRA_FULL_ERROR=1
# SIAM bridge reads this at model-instantiation time (not just ckpt conversion).
export SIAM_MODEL_DIR=${SIAM_MODEL_DIR:-/siam_params/v0.3/pred_DS108_LcsfP_Ano}
mkdir -p "$ASPARAGUS_MODELS" "$ASPARAGUS_RESULTS" "$ASPARAGUS_RAW_LABELS"

ARMS=${ARMS:-"smri_siam smri_fomo60k smri_anatcl smri_simclr3d smri_triad smri_brainiac smri_mmunetvae"}
TASKS=${TASKS:-"CLS002_FOMO26_Infarct CLS002_FOMO26_Infarct_DWI"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
DEBUG=${DEBUG:-0}
EPOCHS=${EPOCHS:-50}
BATCH=${BATCH:-1}
SUMMARY=${SUMMARY:-/fomo26/results/cls002_sweep_summary.tsv}

echo ">>> torch/cuda check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

CONS=/tmp/constraints.txt
python - <<'PY' > "$CONS"
import torch
print(f"torch=={torch.__version__}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__}")
except Exception:
    pass
PY
echo ">>> installing asparagus runtime deps once"
pip install -q --no-cache-dir --root-user-action=ignore -c "$CONS" \
  gardening_tools 'monai==1.5.2' 'lightning==2.5.0' torchmetrics \
  hydra-core omegaconf python-dotenv wandb nnunetv2 \
  nibabel scikit-image scikit-learn pandas scipy einops 2>&1 | tail -3 || true

convert_ckpt() {  # arm -> writes /fomo26/<arm>_cls002_asparagus.ckpt; echoes path
  local arm=$1 ckpt=/fomo26/${1#smri_}_cls002_asparagus.ckpt
  case "$arm" in
    smri_siam)      python -c "from asparagus_bridge.checkpoint import convert_checkpoint; convert_checkpoint('smri_siam', '${SIAM_MODEL_DIR:-/siam_params/v0.3/pred_DS108_LcsfP_Ano}/fold_0/checkpoint_final.pth', '$ckpt')" ;;
    smri_fomo60k)   python -c "from asparagus_bridge.models_smri_fomo60k import convert_fomo60k_checkpoint; convert_fomo60k_checkpoint('${FOMO60K_CKPT:-/fomo26/weights/fomo60k_pkoutsouvelis/combined_regular-step=200000.ckpt}', '$ckpt')" ;;
    smri_brainiac)  python -c "from asparagus_bridge.models_smri_brainiac import convert_brainiac_checkpoint; convert_brainiac_checkpoint('${BRAINIAC_CKPT:-/home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt}', '$ckpt')" ;;
    smri_triad)     python -c "from asparagus_bridge.models_smri_triad import convert_triad_checkpoint; convert_triad_checkpoint('${TRIAD_CKPT:-/fomo26/weights/triad/Triad-SwinB-MAE.pth}', '$ckpt')" ;;
    smri_anatcl)    python -c "from asparagus_bridge.models_smri_anatcl import convert_anatcl_checkpoint; convert_anatcl_checkpoint('${ANATCL_CKPT:-/fomo26/weights/anatcl/anatcl_global_fold0.pth}', '$ckpt')" ;;
    smri_mmunetvae) python -c "from asparagus_bridge.models_smri_mmunetvae import convert_mmunetvae_checkpoint; convert_mmunetvae_checkpoint('${MMUNETVAE_CKPT:-/fomo26/weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt}', '$ckpt')" ;;
    smri_simclr3d)  python -c "from asparagus_bridge.models_smri_simclr3d import convert_simclr3d_checkpoint; convert_simclr3d_checkpoint('${SIMCLR_CKPT:-/fomo26/weights/simclr3d/simclr_3d_brain_foundation.tar}', '$ckpt')" ;;
    *) return 2 ;;
  esac
  echo "$ckpt"
}

if [ "$DEBUG" = "1" ]; then
  EXTRA=(training.epochs=5 training.limit_train_batches=10 training.limit_val_batches=5
         training.target_size=[96,96,96] training.warmup_epochs=1)
else
  EXTRA=(training.epochs=$EPOCHS training.target_size=[128,128,128] training.warmup_epochs=5)
fi

[ -f "$SUMMARY" ] || echo -e "arm\ttask\tfold\trundir\tstatus" > "$SUMMARY"
echo "=== sweep: ARMS=[$ARMS] TASKS=[$TASKS] FOLDS=[$FOLDS] DEBUG=$DEBUG ==="

for ARM in $ARMS; do
  echo "################ converting $ARM ################"
  if ! CKPT=$(convert_ckpt "$ARM"); then echo "!! convert failed $ARM"; continue; fi
  echo ">>> $ARM ckpt -> $CKPT"
  for TASK in $TASKS; do
    suffix=${TASK#CLS002_FOMO26_Infarct}; suffix=${suffix:-_FULL}
    for FOLD in $FOLDS; do
      RUNDIR=$ASPARAGUS_MODELS/cls002_${ARM#smri_}${suffix}_cv${FOLD}
      echo "==== RUN arm=$ARM task=$TASK fold=$FOLD -> $RUNDIR ===="
      if python -m asparagus.pipeline.run.finetune_cls \
          task="$TASK" +model="$ARM" checkpoint_path="$CKPT" \
          data.train_split=split_cv5 data.test_split=TEST_cv${FOLD} data.fold=${FOLD} \
          training.batch_size=${BATCH} training.load_decoder=False \
          hardware.num_workers=8 logger.wandb_logging=False \
          hydra.run.dir="$RUNDIR" "${EXTRA[@]}"; then
        st=OK
      else
        st=FAIL
      fi
      echo -e "${ARM}\t${TASK}\t${FOLD}\t${RUNDIR}\t${st}" >> "$SUMMARY"
      echo "==== $st arm=$ARM task=$TASK fold=$FOLD ===="
    done
  done
done

echo ">>> sweep complete"; cat "$SUMMARY"
