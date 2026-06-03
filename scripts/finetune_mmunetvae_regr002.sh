#!/usr/bin/env bash
# Finetune the FOMO25-winning Multi-Modal U-Net VAE (mmunetvae, SSL on FOMO60K
# raw MRI) on FOMO26 Task 3 (REGR002 brain-age regression), inside an NVIDIA
# PyTorch (NGC) container with --gpus all. Mirrors the other bridge drivers.
# The U-Net encoder pools to a VAE bottleneck so any crop size works; we use
# 128^3/batch 2 to match the other arms. DOMAIN: pretrained on raw FOMO60K MRI
# -> same domain as FOMO26 (most task-aligned arm). See models_smri_mmunetvae.
#
# The bridge imports the vendored FOMO25 `src` tree (extracted from the amd64
# docker image jbanusco/sslmmunetave:1.0.0, which cannot run on the GB10) plus a
# yucca_stub, both under $REPO/third_party/fomo25_mmunetvae; the bridge module
# puts them on sys.path itself, so no extra PYTHONPATH entry is required here.
#
# Expected mounts (set by the docker wrapper):
#   /workspace/smri-fm   <- repo root (includes third_party/fomo25_mmunetvae)
#   /fomo26              <- /data/datasets/fomo26   (also holds the mmunetvae ckpt)
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

MMUNETVAE_CKPT=${MMUNETVAE_CKPT:-/fomo26/weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt}

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

echo ">>> converting mmunetvae checkpoint -> asparagus format"
CKPT=/fomo26/mmunetvae_regr002_asparagus.ckpt
python -c "from asparagus_bridge.models_smri_mmunetvae import convert_mmunetvae_checkpoint; convert_mmunetvae_checkpoint('$MMUNETVAE_CKPT', '$CKPT'); print('wrote', '$CKPT')"

DEBUG=${DEBUG:-1}
if [ "$DEBUG" = "1" ]; then
  echo ">>> DEBUG smoke finetune (5 epochs, 64^3 crop, limited batches)"
  RUNDIR=$ASPARAGUS_MODELS/regr002_mmunetvae_debug
  EXTRA=(training.epochs=5 training.limit_train_batches=10 training.limit_val_batches=5
         training.target_size=[64,64,64] training.batch_size=1 training.check_val_every_n_epoch=1)
else
  echo ">>> REAL finetune (50 epochs, 128^3 crop, full 494-case cohort)"
  RUNDIR=$ASPARAGUS_MODELS/regr002_mmunetvae
  EXTRA=(training.target_size=[128,128,128] training.batch_size=2)
fi

set -x
python -m asparagus.pipeline.run.finetune_reg \
  task=REGR002_FOMO26_BrainAge \
  +model=smri_mmunetvae \
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
