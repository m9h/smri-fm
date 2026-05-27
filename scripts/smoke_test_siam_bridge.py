"""Smoke-test the SIAM asparagus bridge.

End-to-end verification (no training):
  1. Read $SIAM_MODEL_DIR plans + ckpt.
  2. Run the registered converter -> asparagus.ckpt.
  3. Build SmriSiamClsRegBackbone(input_channels=1, output_channels=2).
  4. load_state_dict(strict=False); report missing/unexpected keys.
  5. Forward a random tensor at SIAM's expected patch size; report shape.
  6. Repeat for SmriSiamSegBackbone(input_channels=1, output_channels=5).

Run:
    SIAM_MODEL_DIR=~/siam_params/v0.3/pred_DS108_LcsfP_Ano \
      .venv/bin/python scripts/smoke_test_siam_bridge.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from asparagus_bridge.checkpoint import convert_checkpoint
from asparagus_bridge.models_smri_siam import (
    SmriSiamClsRegBackbone,
    SmriSiamSegBackbone,
)


def _summarize_keys(missing: list[str], unexpected: list[str], head_prefixes: tuple[str, ...]) -> None:
    head_missing = [k for k in missing if any(k.startswith(p) for p in head_prefixes)]
    rest_missing = [k for k in missing if k not in head_missing]
    print(f"    missing keys:    total={len(missing)}  head/init-allowed={len(head_missing)}  unexpected-missing={len(rest_missing)}")
    if rest_missing:
        print(f"    first 5 unexpected-missing: {rest_missing[:5]}")
    print(f"    unexpected keys: total={len(unexpected)}")
    if unexpected:
        print(f"    first 5 unexpected: {unexpected[:5]}")


def main() -> int:
    siam_model_dir = os.environ.get("SIAM_MODEL_DIR")
    if not siam_model_dir:
        print("ERROR: SIAM_MODEL_DIR is not set", file=sys.stderr)
        return 2

    src_ckpt = Path(siam_model_dir) / "fold_0" / "checkpoint_final.pth"
    if not src_ckpt.exists():
        print(f"ERROR: missing SIAM checkpoint at {src_ckpt}", file=sys.stderr)
        return 2

    plans_path = Path(siam_model_dir) / "plans.json"
    plans = json.loads(plans_path.read_text())
    patch_size = (
        plans["configurations"]
        [os.environ.get("SIAM_CONFIG", "3d_fullres")]
        ["patch_size"]
    )
    print(f"plans patch_size = {patch_size}")

    with tempfile.TemporaryDirectory() as tmp:
        dst_ckpt = Path(tmp) / "siam_asparagus.ckpt"
        print(f"\n>>> converting {src_ckpt}")
        convert_checkpoint("smri_siam", src_ckpt, dst_ckpt)
        converted = torch.load(dst_ckpt, map_location="cpu", weights_only=False)
        n_keys = len(converted["state_dict"])
        prefixes = sorted({k.split(".", 2)[1] for k in converted["state_dict"] if "." in k})[:6]
        print(f"    converted state_dict: {n_keys} keys; top-level prefixes after 'model.': {prefixes}")

        # The converter writes keys with the asparagus BaseModule prefix
        # (`model.<wrapper_attr>...`). When loading directly into a wrapper
        # outside BaseModule, strip that prefix.
        wrapper_sd = {
            k[len("model."):]: v for k, v in converted["state_dict"].items()
            if k.startswith("model.")
        }

        # cls/reg head
        print("\n>>> building SmriSiamClsRegBackbone(in=1, out=2)")
        cls = SmriSiamClsRegBackbone(input_channels=1, output_channels=2)
        miss, unexp = cls.load_state_dict(wrapper_sd, strict=False)
        _summarize_keys(list(miss), list(unexp), head_prefixes=("head.",))
        cls.eval()
        with torch.no_grad():
            x = torch.randn(1, 1, *patch_size)
            y = cls(x)
        print(f"    cls forward: input {tuple(x.shape)} -> output {tuple(y.shape)} (expected (1, 2))")

        # seg head
        print("\n>>> building SmriSiamSegBackbone(in=1, out=5)")
        seg = SmriSiamSegBackbone(input_channels=1, output_channels=5)
        miss, unexp = seg.load_state_dict(wrapper_sd, strict=False)
        _summarize_keys(
            list(miss),
            list(unexp),
            head_prefixes=("decoder.seg_layers.",),
        )
        seg.eval()
        with torch.no_grad():
            x = torch.randn(1, 1, *patch_size)
            y = seg(x)
            y_shape = tuple(y.shape) if torch.is_tensor(y) else [tuple(t.shape) for t in y]
        print(f"    seg forward: input {tuple(x.shape)} -> output {y_shape}")

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
