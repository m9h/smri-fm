#!/usr/bin/env bash
# Runs INSIDE the NGC pytorch container. Frozen bridge-FM embeddings (fomo60k/anatcl/triad).
# Driven by env: ARM, CKPT, TARGET, (optional) INPUT_CSV / OUT_DIR.
set -euo pipefail

REPO=/workspace/smri-fm
PROBE=$REPO/experiments/fomo26_fm_benchmark/brainage_probe
export PYTHONPATH=$REPO/third_party/asparagus:$REPO/src${PYTHONPATH:+:$PYTHONPATH}

python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

CONS=/tmp/cons.txt
python - <<'PY' > "$CONS"
import torch; print(f"torch=={torch.__version__}")
try:
    import torchvision; print(f"torchvision=={torchvision.__version__}")
except Exception: pass
PY
echo ">>> install bridge deps (constraint-pinned)"
pip install -q --no-cache-dir --root-user-action=ignore -c "$CONS" \
  gardening_tools nnunetv2 'monai==1.5.2' einops nibabel scikit-image scikit-learn pandas scipy tqdm 2>&1 | tail -3 || true

ARM="${ARM:?set ARM}"
CKPT="${CKPT:?set CKPT}"
TARGET="${TARGET:-96}"
INPUT_CSV="${INPUT_CSV:-$PROBE/data/input_csv.csv}"
OUT_DIR="${OUT_DIR:-$PROBE/results/${ARM}_embed}"

echo ">>> extracting $ARM embeddings: $(wc -l < "$INPUT_CSV") rows -> $OUT_DIR"
python "$PROBE/scripts/extract_bridge_embeddings.py" \
  --arm "$ARM" --checkpoint "$CKPT" \
  --input_csv "$INPUT_CSV" --root_dir "$PROBE/data/t3_flat" \
  --out_dir "$OUT_DIR" --target "$TARGET"
echo ">>> done"
