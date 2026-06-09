#!/usr/bin/env bash
# Runs INSIDE the NGC pytorch container. Frozen mmunetvae (FOMO25 winner) embeddings
# over the FOMO26 Task-3 symlink farm. Uses the vendored fomo25 src in the repo.
set -euo pipefail

REPO=/workspace/smri-fm
PROBE=$REPO/experiments/fomo26_fm_benchmark/brainage_probe
export PYTHONPATH=$REPO/third_party/asparagus:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export FOMO25_MMUNETVAE_SRC=$REPO/third_party/fomo25_mmunetvae

python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

CONS=/tmp/cons.txt
python - <<'PY' > "$CONS"
import torch; print(f"torch=={torch.__version__}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__}")
except Exception: pass
PY
echo ">>> install mmunetvae-path deps (constraint-pinned)"
pip install -q --no-cache-dir --root-user-action=ignore -c "$CONS" \
  monai einops nibabel scikit-image scikit-learn pandas scipy tqdm 2>&1 | tail -3 || true

CKPT="${CKPT:-/data/datasets/fomo26/weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt}"
INPUT_CSV="${INPUT_CSV:-$PROBE/data/input_csv.csv}"
ROOT_DIR="${ROOT_DIR:-$PROBE/data/t3_flat}"
OUT_DIR="${OUT_DIR:-$PROBE/results/mmunetvae_embed}"

echo ">>> extracting mmunetvae embeddings: $(wc -l < "$INPUT_CSV") rows -> $OUT_DIR"
python "$PROBE/scripts/extract_fomo25_mmunetvae.py" \
  --input_csv "$INPUT_CSV" --root_dir "$ROOT_DIR" \
  --checkpoint "$CKPT" --out_dir "$OUT_DIR" --target 64
echo ">>> done"
