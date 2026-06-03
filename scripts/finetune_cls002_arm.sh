#!/usr/bin/env bash
# Generalized FOMO26 CLS002 (infarct) finetune driver — one arm, one CV fold.
# Parameterizes the 7 FM arms over the per-arm checkpoint-conversion logic that
# each finetune_<arm>_regr002.sh already encodes, so the diffusion-classification
# leaderboard reuses the exact same pretrained weights as the REGR002 leaderboard.
#
# Env contract (set by the caller / orchestrator):
#   ARM         smri_siam|smri_fomo60k|smri_brainiac|smri_triad|smri_anatcl|smri_mmunetvae|smri_simclr3d
#   TASK        CLS002_FOMO26_Infarct (3ch flair+adc+dwi) | CLS002_FOMO26_Infarct_DWI (1ch dwi)
#   TRAIN_SPLIT split file stem (default split_cv5)
#   TEST_SPLIT  test file stem   (e.g. TEST_cv0)
#   FOLD        fold index into TRAIN_SPLIT (e.g. 0)
#   RUNDIR      hydra run dir (default derived)
#   DEBUG       1=smoke, 0=real (default 0 here; this driver is for real runs)
#
# Expected mounts (docker wrapper): /workspace/smri-fm, /fomo26, optional
# /siam_params (SIAM) and /home/mhough/dev/BrainIAC (BrainIAC).
set -euo pipefail

REPO=/workspace/smri-fm
export ASPARAGUS_CONFIGS=$REPO/third_party/asparagus/configs
export ASPARAGUS_FINETUNE_CONFIGS=$REPO/src/asparagus_bridge/configs
export ASPARAGUS_DATA=/fomo26/processed
export ASPARAGUS_RAW_LABELS=/fomo26/raw_labels
export ASPARAGUS_MODELS=/fomo26/models
export ASPARAGUS_RESULTS=/fomo26/results
export PYTHONPATH=$REPO/third_party/asparagus:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=offline
export WANDB_DISABLED=true
export HYDRA_FULL_ERROR=1
mkdir -p "$ASPARAGUS_MODELS" "$ASPARAGUS_RESULTS" "$ASPARAGUS_RAW_LABELS"

ARM=${ARM:?set ARM}
TASK=${TASK:-CLS002_FOMO26_Infarct}
TRAIN_SPLIT=${TRAIN_SPLIT:-split_cv5}
TEST_SPLIT=${TEST_SPLIT:?set TEST_SPLIT}
FOLD=${FOLD:?set FOLD}
DEBUG=${DEBUG:-0}
RUNDIR=${RUNDIR:-$ASPARAGUS_MODELS/cls002_${ARM#smri_}_${TASK#CLS002_FOMO26_Infarct}_${TEST_SPLIT}}

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

echo ">>> installing asparagus runtime deps (constraint-pinned)"
pip install -q --no-cache-dir --root-user-action=ignore -c "$CONS" \
  gardening_tools 'monai==1.5.2' 'lightning==2.5.0' torchmetrics \
  hydra-core omegaconf python-dotenv wandb nnunetv2 \
  nibabel scikit-image scikit-learn pandas scipy einops 2>&1 | tail -3 || true

CKPT=/fomo26/${ARM#smri_}_cls002_asparagus.ckpt
echo ">>> converting $ARM checkpoint -> $CKPT"
case "$ARM" in
  smri_siam)
    SRC=${SIAM_MODEL_DIR:-/siam_params/v0.3/pred_DS108_LcsfP_Ano}/fold_0/checkpoint_final.pth
    python -c "from asparagus_bridge.checkpoint import convert_checkpoint; convert_checkpoint('smri_siam', '$SRC', '$CKPT')" ;;
  smri_fomo60k)
    SRC=${FOMO60K_CKPT:-/fomo26/weights/fomo60k_pkoutsouvelis/combined_regular-step=200000.ckpt}
    python -c "from asparagus_bridge.models_smri_fomo60k import convert_fomo60k_checkpoint; convert_fomo60k_checkpoint('$SRC', '$CKPT')" ;;
  smri_brainiac)
    SRC=${BRAINIAC_CKPT:-/home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt}
    python -c "from asparagus_bridge.models_smri_brainiac import convert_brainiac_checkpoint; convert_brainiac_checkpoint('$SRC', '$CKPT')" ;;
  smri_triad)
    SRC=${TRIAD_CKPT:-/fomo26/weights/triad/Triad-SwinB-MAE.pth}
    python -c "from asparagus_bridge.models_smri_triad import convert_triad_checkpoint; convert_triad_checkpoint('$SRC', '$CKPT')" ;;
  smri_anatcl)
    SRC=${ANATCL_CKPT:-/fomo26/weights/anatcl/anatcl_global_fold0.pth}
    python -c "from asparagus_bridge.models_smri_anatcl import convert_anatcl_checkpoint; convert_anatcl_checkpoint('$SRC', '$CKPT')" ;;
  smri_mmunetvae)
    SRC=${MMUNETVAE_CKPT:-/fomo26/weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt}
    python -c "from asparagus_bridge.models_smri_mmunetvae import convert_mmunetvae_checkpoint; convert_mmunetvae_checkpoint('$SRC', '$CKPT')" ;;
  smri_simclr3d)
    SRC=${SIMCLR_CKPT:-/fomo26/weights/simclr3d/simclr_3d_brain_foundation.tar}
    python -c "from asparagus_bridge.models_smri_simclr3d import convert_simclr3d_checkpoint; convert_simclr3d_checkpoint('$SRC', '$CKPT')" ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
echo ">>> wrote $CKPT"

if [ "$DEBUG" = "1" ]; then
  echo ">>> DEBUG smoke (5 epochs, 96^3)"
  EXTRA=(training.epochs=5 training.limit_train_batches=10 training.limit_val_batches=5
         training.target_size=[96,96,96] training.warmup_epochs=1 training.check_val_every_n_epoch=1)
else
  echo ">>> REAL finetune (50 epochs, 128^3)"
  EXTRA=(training.epochs=50 training.target_size=[128,128,128] training.warmup_epochs=5)
fi

set -x
python -m asparagus.pipeline.run.finetune_cls \
  task="$TASK" \
  +model="$ARM" \
  checkpoint_path="$CKPT" \
  data.train_split="$TRAIN_SPLIT" \
  data.test_split="$TEST_SPLIT" \
  data.fold="$FOLD" \
  training.batch_size=1 \
  training.load_decoder=False \
  hardware.num_workers=8 \
  logger.wandb_logging=False \
  hydra.run.dir="$RUNDIR" \
  "${EXTRA[@]}"
set +x
echo ">>> done ARM=$ARM TASK=$TASK FOLD=$FOLD TEST_SPLIT=$TEST_SPLIT"
