#!/usr/bin/env bash
# Finetune the 3D-Neuro-SimCLR (MONAI 3D-ResNet-18, raw-T1 SimCLR contrastive)
# FM on FOMO26 Task 3 (REGR002 brain-age regression), inside an NVIDIA PyTorch
# (NGC) container with --gpus all. Mirrors the other bridge drivers. The MONAI
# ResNet uses adaptive GAP so any crop size works; we use 128^3/batch 2 to match
# the other arms. DOMAIN: pretrained on raw T1 (turboprep/MNI152) -> closer to
# FOMO26 raw T1 than the VBM-pretrained AnatCL arm (see models_smri_simclr3d).
#
# Expected mounts (set by the docker wrapper):
#   /workspace/smri-fm   <- repo root
#   /fomo26              <- /data/datasets/fomo26   (also holds the SimCLR ckpt)
#
# DEBUG=1 (default) = short smoke; DEBUG=0 = real 50-epoch finetune.
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

SIMCLR_CKPT=${SIMCLR_CKPT:-/fomo26/weights/simclr3d/simclr_3d_brain_foundation.tar}

echo ">>> torch/cuda check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo ">>> pinning NGC torch/torchvision in a constraints file (no downgrade)"
CONS=/tmp/constraints.txt
python - <<'PY' > "$CONS"
import torch
print(f"torch=={torch.__version__}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__}")
except Exception:
    pass
PY
cat "$CONS"

echo ">>> installing asparagus runtime deps (constraint-pinned)"
pip install -q --no-cache-dir --root-user-action=ignore -c "$CONS" \
  gardening_tools 'monai==1.5.2' 'lightning==2.5.0' torchmetrics \
  hydra-core omegaconf python-dotenv wandb nnunetv2 \
  nibabel scikit-image scikit-learn pandas scipy einops 2>&1 | tail -5 || true

echo ">>> converting SimCLR checkpoint -> asparagus format"
CKPT=/fomo26/simclr3d_regr002_asparagus.ckpt
python -c "from asparagus_bridge.models_smri_simclr3d import convert_simclr3d_checkpoint; convert_simclr3d_checkpoint('$SIMCLR_CKPT', '$CKPT'); print('wrote', '$CKPT')"

DEBUG=${DEBUG:-1}
if [ "$DEBUG" = "1" ]; then
  echo ">>> DEBUG smoke finetune (5 epochs, 96^3 crop, limited batches)"
  RUNDIR=$ASPARAGUS_MODELS/regr002_simclr3d_debug
  EXTRA=(training.epochs=5 training.limit_train_batches=10 training.limit_val_batches=5
         training.target_size=[96,96,96] training.batch_size=1 training.check_val_every_n_epoch=1)
else
  echo ">>> REAL finetune (50 epochs, 128^3 crop, full 494-case cohort)"
  RUNDIR=$ASPARAGUS_MODELS/regr002_simclr3d
  EXTRA=(training.target_size=[128,128,128] training.batch_size=2)
fi

set -x
python -m asparagus.pipeline.run.finetune_reg \
  task=REGR002_FOMO26_BrainAge \
  +model=smri_simclr3d \
  checkpoint_path="$CKPT" \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10 \
  training.load_decoder=False \
  hardware.num_workers=8 \
  logger.wandb_logging=False \
  hydra.run.dir="$RUNDIR" \
  "${EXTRA[@]}"
set +x
echo ">>> done"
