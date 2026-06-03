"""Data-free smoke test of the BrainIAC -> asparagus cls/reg bridge.

Validates the part of the FOMO26 BrainIAC finetune path that does NOT need the
(not-yet-downloaded) FOMO26 task data: convert BrainIAC's real pretrained
checkpoint, load it into SmriBrainiacClsRegBackbone exactly as asparagus'
BaseModule would (strict=False), and run a forward pass on a synthetic batch.

Confirms:
  - BrainIAC.ckpt is readable and the converter emits model.encoder.* keys
  - those keys land in the wrapper's MONAI ViT encoder (missing keys = only the
    fresh task head.*; no unexpected keys)
  - the cls/reg forward produces [B, num_classes] with the real weights

Requires MONAI (BrainIAC's dependency). The intended interpreter is the one
weightwatcher_brainiac.py was run under; if MONAI is absent the import will
fail and that should be reported rather than worked around.

Usage:
    PYTHONPATH=/home/mhough/dev/smri-fm-fomo26/src \
      [BRAINIAC_CKPT=/home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt] \
      python scripts/smoke_brainiac_checkpoint_load.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

from asparagus_bridge.models_smri_brainiac import (
    SmriBrainiacClsRegBackbone,
    convert_brainiac_checkpoint,
)


DEFAULT_CKPT = "/home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt"
NUM_CLASSES = 1  # FOMO26 brain-age regression: single output
INPUT_SHAPE = (1, 1, 96, 96, 96)  # BrainIAC's MONAI ViT img_size = 96^3


def main() -> int:
    ckpt = os.environ.get("BRAINIAC_CKPT", DEFAULT_CKPT)
    if not Path(ckpt).exists():
        print(f"ERROR: BrainIAC checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    print(f">>> converting BrainIAC checkpoint: {ckpt}")
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "brainiac_asparagus.pth"
        convert_brainiac_checkpoint(ckpt, dst)
        blob = torch.load(dst, map_location="cpu", weights_only=False)
    sd_full = blob["state_dict"]
    print(f"    converted state_dict: {len(sd_full)} tensors  epoch={blob.get('epoch')}")

    print(f">>> building SmriBrainiacClsRegBackbone(in=1, out={NUM_CLASSES})")
    model = SmriBrainiacClsRegBackbone(input_channels=1, output_channels=NUM_CLASSES)

    # Mirror asparagus BaseModule: strip the leading "model." and load strict=False.
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd_full.items()}

    # Report how many offered encoder tensors match the model's encoder shapes.
    model_sd = model.state_dict()
    offered_enc = [k for k in sd if k.startswith("encoder.")]
    shape_match = sum(1 for k in offered_enc if k in model_sd and model_sd[k].shape == sd[k].shape)
    shape_mismatch = [k for k in offered_enc if k in model_sd and model_sd[k].shape != sd[k].shape]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # MONAI 1.5.2's TransformerBlock unconditionally instantiates cross-attention
    # submodules (cross_attn.*, norm_cross_attn.*), but ViT never enables them
    # (with_cross_attention defaults False), so they are never exercised in the
    # forward pass. The pretrained BrainIAC ViT predates them, so they show up as
    # "missing" — inert MONAI-version artifacts, not a real load gap.
    def _inert(k: str) -> bool:
        return ".cross_attn." in k or ".norm_cross_attn." in k
    miss_non_head = [k for k in missing if not k.startswith("head.") and not _inert(k)]
    inert_missing = [k for k in missing if _inert(k)]
    print(f"    encoder tensors offered: {len(offered_enc)}")
    print(f"    encoder tensors with matching shapes: {shape_match}")
    if shape_mismatch:
        print(f"      WARN shape-mismatched encoder keys ({len(shape_mismatch)}): {shape_mismatch[:6]}")
    print(f"    missing: {len(missing)}  (head.*: {len(missing) - len(miss_non_head) - len(inert_missing)}, "
          f"inert cross-attn: {len(inert_missing)}, other non-head: {len(miss_non_head)})")
    print(f"    unexpected (expect none): {len(unexpected)}")
    if miss_non_head:
        print(f"      WARN missing non-head keys: {miss_non_head[:6]}")
    if unexpected:
        print(f"      WARN unexpected keys: {unexpected[:6]}")

    print(f">>> forward on synthetic {INPUT_SHAPE}")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(*INPUT_SHAPE))
    print(f"    output shape: {tuple(out.shape)}  (expect ({INPUT_SHAPE[0]}, {NUM_CLASSES}))")

    ok = (
        tuple(out.shape) == (INPUT_SHAPE[0], NUM_CLASSES)
        and not miss_non_head
        and not unexpected
        and not shape_mismatch
        and shape_match == len(offered_enc)
    )
    print()
    print("PASS: BrainIAC weights convert + load into the cls/reg bridge and forward cleanly."
          if ok else "FAIL: see warnings above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
