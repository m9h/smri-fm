#!/usr/bin/env bash
# Runs INSIDE the NGC pytorch container (see extract_amaes_docker.sh).
# Extracts frozen AMAES resenc_b embeddings over the FOMO26 Task-3 symlink farm.
set -euo pipefail

REPO=/workspace/smri-fm
PROBE=$REPO/experiments/fomo26_fm_benchmark/brainage_probe
export PYTHONPATH=$REPO/third_party/asparagus:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/data/datasets/fomo26/weights/hf_cache

echo ">>> torch/cuda"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo ">>> pin NGC torch in a constraints file (no downgrade)"
CONS=/tmp/cons.txt
python - <<'PY' > "$CONS"
import torch; print(f"torch=={torch.__version__}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__}")
except Exception: pass
PY

echo ">>> install AMAES-path deps (constraint-pinned)"
pip install -q --no-cache-dir --root-user-action=ignore -c "$CONS" \
  gardening_tools 'monai==1.5.2' einops nibabel scikit-image scikit-learn \
  pandas scipy tqdm huggingface_hub 2>&1 | tail -3 || true

CKPT="${CKPT:-/data/datasets/fomo26/weights/amaes/resenc_unet_b.ckpt}"
INPUT_CSV="${INPUT_CSV:-$PROBE/data/input_csv.csv}"
ROOT_DIR="${ROOT_DIR:-$PROBE/data/t3_flat}"
OUT_DIR="${OUT_DIR:-$PROBE/results/amaes_embed}"
TARGET="${TARGET:-96}"

echo ">>> extracting AMAES embeddings (target ${TARGET}): $(wc -l < "$INPUT_CSV") rows -> $OUT_DIR"
python "$PROBE/scripts/extract_fomo25_embeddings.py" \
  --input_csv "$INPUT_CSV" --root_dir "$ROOT_DIR" \
  --checkpoint "$CKPT" --out_dir "$OUT_DIR" --target "$TARGET"
echo ">>> done"
