#!/usr/bin/env bash
# In-container: test-only siam on the saved ISLES22 best.ckpt (with the tiling fix).
set -euo pipefail
REPO=/workspace/smri-fm
export ASPARAGUS_CONFIGS=$REPO/third_party/asparagus/configs
export ASPARAGUS_FINETUNE_CONFIGS=$REPO/src/asparagus_bridge/configs
export ASPARAGUS_DATA=/fomo26/processed ASPARAGUS_RAW_LABELS=/fomo26/raw_labels
export ASPARAGUS_MODELS=/fomo26/models ASPARAGUS_RESULTS=/fomo26/results
export SIAM_MODEL_DIR=/siam_params/v0.3/pred_DS108_LcsfP_Ano
export PYTHONPATH=$REPO/third_party/asparagus:$REPO/src
export WANDB_MODE=offline WANDB_DISABLED=true HYDRA_FULL_ERROR=1
if ! python -c "import nnunetv2, monai, lightning" 2>/dev/null; then
  CONS=/tmp/c.txt; python -c "import torch;print('torch=='+torch.__version__)" > $CONS
  pip install -q --no-cache-dir --root-user-action=ignore -c $CONS \
    gardening_tools 'monai==1.5.2' 'lightning==2.5.0' torchmetrics hydra-core omegaconf \
    python-dotenv wandb nnunetv2 nibabel scikit-image scikit-learn pandas scipy einops 2>&1 | tail -2 || true
fi
CKPT=/fomo26/siam_seg_asparagus.ckpt
python -c "from asparagus_bridge.checkpoint import convert_checkpoint; convert_checkpoint('smri_siam','$SIAM_MODEL_DIR/fold_0/checkpoint_final.pth','$CKPT')"
export TEST_CKPT=/fomo26/models/seg_siam_seg011_isles22_ischstroke/checkpoints/best.ckpt
python -m asparagus.pipeline.run.predict_seg \
  task=SEG011_ISLES22_IschStroke +model=smri_siam \
  checkpoint_path=$CKPT training.load_decoder=False \
  data.train_split=split_80_10_10 data.test_split=TEST_80_10_10 \
  hardware.num_workers=8 logger.wandb_logging=False \
  hydra.run.dir=$ASPARAGUS_MODELS/seg_siam_seg011_isles22_ischstroke_predict \
  training.epochs=200 training.patch_size=[128,128,128] training.batch_size=2 \
  training.train_batches_per_epoch_per_device=50 training.val_batches_per_epoch_per_device=20 \
  training.warmup_epochs=10 training.decoder_warmup_epochs=10 training.check_val_every_n_epoch=5
echo ">>> predict done"
