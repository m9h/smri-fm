"""Data-free smoke test of the 3D-Neuro-SimCLR -> asparagus cls/reg bridge.

Converts SimCLR's pretrained checkpoint, loads it into
SmriSimclr3dClsRegBackbone exactly as asparagus' BaseModule would (strip leading
"model.", strict=False), and runs a forward pass on a synthetic batch.

Confirms:
  - the SimCLR checkpoint is readable and the converter emits model.encoder.*
    keys (projector.* dropped, module. prefix stripped)
  - those keys land in the wrapper's MONAI ResNet-18 3D encoder
    (missing keys = only the fresh task head.*; unexpected = empty)
  - the cls/reg forward produces [B, num_classes] with real weights

Usage:
    PYTHONPATH=/home/mhough/dev/smri-fm-fomo26/src \
      /tmp/bridge_venv/bin/python scripts/smoke_simclr3d_checkpoint_load.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from asparagus_bridge.models_smri_simclr3d import (
    SmriSimclr3dClsRegBackbone,
    convert_simclr3d_checkpoint,
)


CKPT = Path("/data/datasets/fomo26/weights/simclr3d/simclr_3d_brain_foundation.tar")
NUM_CLASSES = 1  # brain-age regression is the target task
INPUT_SHAPE = (1, 1, 96, 96, 96)


def main() -> int:
    if not CKPT.exists():
        print(f"ERROR: {CKPT} not found", file=sys.stderr)
        return 2

    print(f">>> converting SimCLR checkpoint: {CKPT}")
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "simclr3d_asparagus.pth"
        convert_simclr3d_checkpoint(CKPT, dst)
        blob = torch.load(dst, map_location="cpu", weights_only=False)
    sd_full = blob["state_dict"]
    print(f"    converted state_dict: {len(sd_full)} tensors  epoch={blob.get('epoch')}")

    print(f">>> building SmriSimclr3dClsRegBackbone(in=1, out={NUM_CLASSES})")
    model = SmriSimclr3dClsRegBackbone(input_channels=1, output_channels=NUM_CLASSES)

    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd_full.items()}
    offered_enc = sum(1 for k in sd if k.startswith("encoder."))

    own = model.state_dict()
    matched = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    shape_mismatch = [k for k in sd if k in own and own[k].shape != sd[k].shape]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss_non_head = [k for k in missing if not k.startswith("head.")]

    print(f"    encoder tensors offered: {offered_enc}")
    print(f"    matched (name+shape) and loaded: {len(matched)}")
    if shape_mismatch:
        print(f"    SHAPE MISMATCH ({len(shape_mismatch)}): {shape_mismatch[:8]}")
    print(f"    missing (expect head.* only): {len(missing)}  non-head: {len(miss_non_head)}")
    print(f"    unexpected (expect empty): {len(unexpected)}")
    if miss_non_head:
        print(f"      WARN missing non-head keys: {miss_non_head[:8]}")
    if unexpected:
        print(f"      WARN unexpected keys: {unexpected[:8]}")

    print(f">>> forward on synthetic {INPUT_SHAPE}")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(*INPUT_SHAPE))
    print(f"    output shape: {tuple(out.shape)}  (expect ({INPUT_SHAPE[0]}, {NUM_CLASSES}))")

    ok = (
        out.shape == (INPUT_SHAPE[0], NUM_CLASSES)
        and not miss_non_head
        and not unexpected
        and not shape_mismatch
        and len(matched) == offered_enc
    )
    print()
    print(
        "PASS: SimCLR weights convert + load into the cls/reg bridge and forward cleanly."
        if ok else "FAIL: see warnings above."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
