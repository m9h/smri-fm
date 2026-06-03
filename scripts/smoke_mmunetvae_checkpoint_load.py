"""Data-free smoke test of the FOMO25 mmunetvae -> asparagus cls/reg bridge.

Converts the FOMO25 Lightning checkpoint, loads it into
SmriMmunetvaeClsRegBackbone exactly as asparagus' BaseModule would (strip
leading "model.", strict=False), and runs a forward pass on a synthetic batch.

Confirms:
  - the Lightning checkpoint is readable and the converter re-emits its 74
    `model.*` tensors as `model.net.*`
  - those keys land in the wrapper's MultiModalUNetVAE (regression mode):
    encoder + reconstruction decoder + 4 VAE conv projections; missing keys =
    only the fresh task head (net.decoder_task.*); unexpected = empty
  - the regression forward produces [B, num_classes] with real weights

Usage:
    PYTHONPATH=/home/mhough/dev/smri-fm-fomo26/src \
      /tmp/bridge_venv/bin/python scripts/smoke_mmunetvae_checkpoint_load.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from asparagus_bridge.models_smri_mmunetvae import (
    SmriMmunetvaeClsRegBackbone,
    convert_mmunetvae_checkpoint,
)


CKPT = Path("/data/datasets/fomo26/weights/mmunetvae/fomo25_mmunetvae_pretrained.ckpt")
NUM_CLASSES = 1  # brain-age regression is the target task
INPUT_SHAPE = (1, 1, 64, 64, 64)  # pretrain patch_size


def main() -> int:
    if not CKPT.exists():
        print(f"ERROR: {CKPT} not found", file=sys.stderr)
        return 2

    print(f">>> converting mmunetvae checkpoint: {CKPT}")
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "mmunetvae_asparagus.pth"
        convert_mmunetvae_checkpoint(CKPT, dst)
        blob = torch.load(dst, map_location="cpu", weights_only=False)
    sd_full = blob["state_dict"]
    print(f"    converted state_dict: {len(sd_full)} tensors  epoch={blob.get('epoch')}")

    print(f">>> building SmriMmunetvaeClsRegBackbone(in=1, out={NUM_CLASSES})")
    model = SmriMmunetvaeClsRegBackbone(input_channels=1, output_channels=NUM_CLASSES)

    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd_full.items()}
    offered = len(sd)

    own = model.state_dict()
    matched = [k for k in sd if k in own and own[k].shape == sd[k].shape]
    shape_mismatch = [k for k in sd if k in own and own[k].shape != sd[k].shape]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss_non_head = [k for k in missing if "decoder_task" not in k]

    print(f"    tensors offered: {offered}")
    print(f"    matched (name+shape) and loaded: {len(matched)}")
    if shape_mismatch:
        print(f"    SHAPE MISMATCH ({len(shape_mismatch)}): {shape_mismatch[:8]}")
    print(f"    missing (expect decoder_task.* only): {len(missing)}  non-head: {len(miss_non_head)}")
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
        and len(matched) == offered
    )
    print()
    print(
        "PASS: mmunetvae weights convert + load into the cls/reg bridge and forward cleanly."
        if ok else "FAIL: see warnings above."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
