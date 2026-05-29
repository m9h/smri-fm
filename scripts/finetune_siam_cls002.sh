#!/usr/bin/env bash
# Finetune the SIAM FM on FOMO26 Task 1 (CLS002 infarct), inside an NVIDIA
# PyTorch (NGC) container with --gpus all. Runs asparagus via PYTHONPATH (NOT
# `pip install asparagus`, which strict-pins torch==2.6.0 and trips the aarch64
# uv/wheel footgun) — we only pip-install asparagus' pure runtime deps, with a
# constraints file that freezes the NGC-provided torch/torchvision so nothing
# downgrades them. gardening_tools needs only torch>=2.6.0, satisfied by NGC.
#
# Expected mounts (set by the docker wrapper):
#   /workspace/smri-fm   <- repo root
#   /fomo26              <- /data/datasets/fomo26  (processed/, models/, results/)
#   /siam_params         <- ~/siam_params          (SIAM result folders)
#
# DEBUG=1 (default) runs a short smoke (few epochs, small crop). DEBUG=0 runs the
# real 50-epoch finetune.
set -euo pipefail

REPO=/workspace/smri-fm
export ASPARAGUS_CONFIGS=$REPO/third_party/asparagus/configs
export ASPARAGUS_FINETUNE_CONFIGS=$REPO/src/asparagus_bridge/configs
export ASPARAGUS_DATA=/fomo26/processed
export ASPARAGUS_RAW_LABELS=/fomo26/raw_labels
export ASPARAGUS_MODELS=/fomo26/models
export ASPARAGUS_RESULTS=/fomo26/results
export SIAM_MODEL_DIR=/siam_params/v0.3/pred_DS108_LcsfP_Ano
export PYTHONPATH=$REPO/third_party/asparagus:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=offline
export WANDB_DISABLED=true
export HYDRA_FULL_ERROR=1
mkdir -p "$ASPARAGUS_MODELS" "$ASPARAGUS_RESULTS" "$ASPARAGUS_RAW_LABELS"

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

echo ">>> converting SIAM fold_0 checkpoint -> asparagus format"
CKPT=/fomo26/siam_cls002_asparagus.ckpt
python -c "from asparagus_bridge.checkpoint import convert_checkpoint; convert_checkpoint('smri_siam', '$SIAM_MODEL_DIR/fold_0/checkpoint_final.pth', '$CKPT'); print('wrote', '$CKPT')"

DEBUG=${DEBUG:-1}
if [ "$DEBUG" = "1" ]; then
  echo ">>> DEBUG smoke finetune (5 epochs, 96^3 crop, limited batches)"
  RUNDIR=$ASPARAGUS_MODELS/cls002_siam_debug
  EXTRA=(training.epochs=5 training.limit_train_batches=10 training.limit_val_batches=5
         training.target_size=[96,96,96] training.warmup_epochs=1 training.check_val_every_n_epoch=1)
else
  echo ">>> REAL finetune (50 epochs, 128^3 crop)"
  RUNDIR=$ASPARAGUS_MODELS/cls002_siam
  EXTRA=(training.target_size=[128,128,128])
fi

# Pin a short hydra run dir: the default embeds every CLI override in a single
# path component and blows past the 255-char filename limit.
set -x
python -m asparagus.pipeline.run.finetune_cls \
  task=CLS002_FOMO26_Infarct \
  +model=smri_siam \
  checkpoint_path="$CKPT" \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10 \
  training.batch_size=1 \
  training.load_decoder=False \
  hardware.num_workers=8 \
  logger.wandb_logging=False \
  hydra.run.dir="$RUNDIR" \
  "${EXTRA[@]}"
set +x
echo ">>> done"
