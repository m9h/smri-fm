"""Data-free smoke test of the SIAM -> asparagus cls/reg bridge.

Validates the part of the FOMO26 SIAM finetune path that does NOT need the
(not-yet-downloaded) FOMO26 task data: convert SIAM's real pretrained
checkpoint, load it into SmriSiamClsRegBackbone exactly as asparagus'
BaseModule would (strict=False), and run a forward pass on a synthetic batch.

Confirms:
  - SIAM_MODEL_DIR plans.json + fold_0/checkpoint_final.pth are readable
  - the converter emits model.encoder.* keys that land in the wrapper encoder
    (missing keys = only the fresh task head; unexpected = only decoder.*)
  - the cls/reg forward produces [B, num_classes] with real weights

Usage:
    PYTHONPATH=/workspace/weightwatcher:/workspace/smri-fm/src \
      SIAM_MODEL_DIR=/siam_params/v0.3/pred_DS108_LcsfP_Ano \
      python scripts/smoke_siam_checkpoint_load.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

from asparagus_bridge.checkpoint import convert_smri_siam_checkpoint
from asparagus_bridge.models_smri_siam import SmriSiamClsRegBackbone


NUM_CLASSES = 2  # pretend a binary cls task (e.g. FOMO26 infarct)
INPUT_SHAPE = (1, 1, 96, 96, 96)


def main() -> int:
    model_dir = os.environ.get("SIAM_MODEL_DIR")
    if not model_dir:
        print("ERROR: SIAM_MODEL_DIR unset", file=sys.stderr)
        return 2
    fold0 = Path(model_dir) / "fold_0" / "checkpoint_final.pth"
    if not fold0.exists():
        print(f"ERROR: {fold0} not found", file=sys.stderr)
        return 2

    print(f">>> converting SIAM checkpoint: {fold0}")
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "siam_asparagus.pth"
        convert_smri_siam_checkpoint(fold0, dst)
        blob = torch.load(dst, map_location="cpu", weights_only=False)
    sd_full = blob["state_dict"]
    print(f"    converted state_dict: {len(sd_full)} tensors  epoch={blob.get('epoch')}")

    print(f">>> building SmriSiamClsRegBackbone(in=1, out={NUM_CLASSES})")
    model = SmriSiamClsRegBackbone(input_channels=1, output_channels=NUM_CLASSES)

    # Mirror asparagus BaseModule: strip the leading "model." and load strict=False.
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd_full.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    enc_loaded = sum(1 for k in sd if k.startswith("encoder."))
    miss_non_head = [k for k in missing if not k.startswith("head.")]
    unexp_non_decoder = [k for k in unexpected if not k.startswith("decoder.")]
    print(f"    encoder tensors offered: {enc_loaded}")
    print(f"    missing (expect head.* only): {len(missing)}  non-head: {len(miss_non_head)}")
    print(f"    unexpected (expect decoder.* only): {len(unexpected)}  non-decoder: {len(unexp_non_decoder)}")
    if miss_non_head:
        print(f"      WARN missing non-head keys: {miss_non_head[:6]}")
    if unexp_non_decoder:
        print(f"      WARN unexpected non-decoder keys: {unexp_non_decoder[:6]}")

    print(f">>> forward on synthetic {INPUT_SHAPE}")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(*INPUT_SHAPE))
    print(f"    output shape: {tuple(out.shape)}  (expect ({INPUT_SHAPE[0]}, {NUM_CLASSES}))")

    ok = (out.shape == (INPUT_SHAPE[0], NUM_CLASSES)) and not miss_non_head and not unexp_non_decoder
    print()
    print("PASS: SIAM weights convert + load into the cls/reg bridge and forward cleanly."
          if ok else "FAIL: see warnings above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
