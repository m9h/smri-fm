#!/usr/bin/env bash
# Finetune one FM arm on one FOMO26 segmentation task (SEG009 Meningioma /
# SEG010 TrigeminalNeuralgia) inside the NGC PyTorch container with --gpus all.
# Parametrized over ARM (smri_siam | smri_mmunetvae) and SEG_TASK; both arms have
# a real seg decoder (SIAM nnU-Net; mmunetvae native segmentation mode with the
# pretrained encoder + VAE bottleneck feeding a fresh UNetDecoder). The other 5
# roster arms are encoder-only and excluded (would need bolt-on decoders).
#
# Expected mounts (set by scripts/docker_seg_sweep.sh):
#   /workspace/smri-fm  <- repo root (incl. third_party/fomo25_mmunetvae)
#   /fomo26             <- /data/datasets/fomo26  (processed cohorts + weights)
#   /siam_params        <- ~/siam_params  (SIAM only)
#
# Env: ARM, SEG_TASK required. DEBUG=1 (default) short smoke; DEBUG=0 real run.
set -euo pipefail

ARM=${ARM:?set ARM=smri_siam|smri_mmunetvae|scratch_nnunet}
SEG_TASK=${SEG_TASK:?set SEG_TASK=SEG009_FOMO26_Meningioma|SEG010_FOMO26_TrigeminalNeuralgia}

REPO=/workspace/smri-fm
export ASPARAGUS_CONFIGS=$REPO/third_party/asparagus/configs
export ASPARAGUS_FINETUNE_CONFIGS=$REPO/src/asparagus_bridge/configs
export ASPARAGUS_DATA=/fomo26/processed
export ASPARAGUS_RAW_LABELS=/fomo26/raw_labels
export ASPARAGUS_MODELS=/fomo26/models
export ASPARAGUS_RESULTS=/fomo26/results
export SIAM_MODEL_DIR=/siam_params/v0.3/pred_DS108_LcsfP_Ano
export PYTHONPATH=$REPO/third_party/asparagus:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export WANDB_MODE=offline WANDB_DISABLED=true HYDRA_FULL_ERROR=1
mkdir -p "$ASPARAGUS_MODELS" "$ASPARAGUS_RESULTS" "$ASPARAGUS_RAW_LABELS"

# deps once per container; skip silently if already present
if ! python -c "import nnunetv2, monai" 2>/dev/null; then
  CONS=/tmp/constraints.txt
  python - > "$CONS" <<'PY'
import torch; print(f"torch=={torch.__version__}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__}")
except Exception: pass
PY
  pip install -q --no-cache-dir --root-user-action=ignore -c "$CONS" \
    gardening_tools 'monai==1.5.2' 'lightning==2.5.0' torchmetrics \
    hydra-core omegaconf python-dotenv wandb nnunetv2 \
    nibabel scikit-image scikit-learn pandas scipy einops 2>&1 | tail -3 || true
fi

# per-arm checkpoint conversion -> asparagus state_dict. The from-scratch control
# (scratch_nnunet) loads NO checkpoint: CKPT_ARG stays empty so resolve_checkpoint
# returns None and the SIAM-topology nnU-Net trains from random init.
CKPT=/fomo26/${ARM#smri_}_seg_asparagus.ckpt
CKPT_ARG=(checkpoint_path="$CKPT" training.load_decoder=False)
case "$ARM" in
  smri_siam)
    python -c "from asparagus_bridge.checkpoint import convert_checkpoint; convert_checkpoint('smri_siam', '$SIAM_MODEL_DIR/fold_0/checkpoint_final.pth', '$CKPT'); print('wrote', '$CKPT')" ;;
  smri_mmunetvae)
    SRC=${MMUNETVAE_CKPT:-/fomo26/weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt}
    python -c "from asparagus_bridge.models_smri_mmunetvae import convert_mmunetvae_checkpoint; convert_mmunetvae_checkpoint('$SRC', '$CKPT'); print('wrote', '$CKPT')" ;;
  scratch_nnunet)
    CKPT_ARG=()  # from-scratch: random init, no weights to convert or load
    echo ">>> FROM-SCRATCH control (scratch_nnunet): no checkpoint, random init" ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac

TAG=${ARM#smri_}_$(echo "$SEG_TASK" | sed -E 's/_FOMO26.*//' | tr 'A-Z' 'a-z')
DEBUG=${DEBUG:-1}
if [ "$DEBUG" = "1" ]; then
  echo ">>> DEBUG smoke seg ($ARM / $SEG_TASK): 6 epochs, 64^3, few batches"
  # epochs must exceed warmup+decoder_warmup so the cosine window (and thus the
  # scheduler T_max) is > 0 — otherwise CosineAnnealingLR divides by zero.
  RUNDIR=$ASPARAGUS_MODELS/seg_${TAG}_debug
  EXTRA=(training.epochs=6 training.patch_size=[64,64,64] training.batch_size=1
         training.train_batches_per_epoch_per_device=5
         training.val_batches_per_epoch_per_device=3
         training.warmup_epochs=1 training.decoder_warmup_epochs=1
         training.check_val_every_n_epoch=1)
else
  echo ">>> REAL seg finetune ($ARM / $SEG_TASK): 200 epochs, 128^3"
  RUNDIR=$ASPARAGUS_MODELS/seg_${TAG}
  EXTRA=(training.epochs=200 training.patch_size=[128,128,128] training.batch_size=2
         training.train_batches_per_epoch_per_device=50
         training.val_batches_per_epoch_per_device=20
         training.warmup_epochs=10 training.decoder_warmup_epochs=10
         training.check_val_every_n_epoch=5)
fi

set -x
python -m asparagus.pipeline.run.finetune_seg \
  task="$SEG_TASK" \
  +model="$ARM" \
  "${CKPT_ARG[@]}" \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10 \
  hardware.num_workers=8 \
  logger.wandb_logging=False \
  hydra.run.dir="$RUNDIR" \
  "${EXTRA[@]}"
set +x
echo ">>> done $ARM / $SEG_TASK -> $RUNDIR"
