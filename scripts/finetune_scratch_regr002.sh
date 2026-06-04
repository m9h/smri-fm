#!/usr/bin/env bash
# Finetune a FROM-SCRATCH CONTROL arm on FOMO26 Task 3 (REGR002 brain-age),
# inside the NGC PyTorch container with --gpus all. Same architecture as the
# matched FM arm but RANDOM init: no checkpoint conversion, no checkpoint_path,
# so asparagus' resolve_checkpoint returns None and BaseModule leaves the net at
# fresh init. These rows give the "no-pretraining floor" the FM arms are read
# against (alongside the morphometry ridge baseline).
#
#   ARM=scratch_swinvit  -> SwinViT-V2 (matches smri_fomo60k / smri_triad)
#   ARM=scratch_vitb     -> ViT-B      (matches smri_brainiac)
#   ARM=scratch_nnunet   -> nnU-Net    (matches smri_siam; needs SIAM_MODEL_DIR
#                                       for plans.json TOPOLOGY only — no weights)
#
# Expected mounts (set by the docker wrapper):
#   /workspace/smri-fm   <- repo root
#   /fomo26              <- /data/datasets/fomo26
#   /siam_params         <- ~/siam_params   (scratch_nnunet only)
#
# DEBUG=1 (default) = short smoke; DEBUG=0 = real 50-epoch finetune.
set -euo pipefail

ARM=${ARM:?set ARM=scratch_swinvit|scratch_vitb|scratch_nnunet}

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
# nnU-Net control needs the SIAM plans.json to define topology (no weights read).
[ "$ARM" = "scratch_nnunet" ] && export SIAM_MODEL_DIR=${SIAM_MODEL_DIR:-/siam_params/v0.3/pred_DS108_LcsfP_Ano}
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

echo ">>> FROM-SCRATCH control ($ARM): no checkpoint, random init"

DEBUG=${DEBUG:-1}
if [ "$DEBUG" = "1" ]; then
  echo ">>> DEBUG smoke finetune (5 epochs, 96^3 crop, limited batches)"
  RUNDIR=$ASPARAGUS_MODELS/regr002_${ARM}_debug
  EXTRA=(training.epochs=5 training.limit_train_batches=10 training.limit_val_batches=5
         training.target_size=[96,96,96] training.batch_size=1 training.check_val_every_n_epoch=1)
else
  echo ">>> REAL finetune (50 epochs, 128^3 crop, full 494-case cohort)"
  RUNDIR=$ASPARAGUS_MODELS/regr002_${ARM}
  EXTRA=(training.target_size=[128,128,128] training.batch_size=2)
fi

# No checkpoint_path override: resolve_checkpoint -> None -> fresh random init.
set -x
python -m asparagus.pipeline.run.finetune_reg \
  task=REGR002_FOMO26_BrainAge \
  +model="$ARM" \
  data.train_split=split_80_10_10 \
  data.test_split=TEST_80_10_10 \
  hardware.num_workers=8 \
  logger.wandb_logging=False \
  hydra.run.dir="$RUNDIR" \
  "${EXTRA[@]}"
set +x
echo ">>> done $ARM -> $RUNDIR"
