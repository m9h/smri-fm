"""Data-free smoke test for the FROM-SCRATCH control arms.

Confirms each control backbone constructs at RANDOM init (no checkpoint) and runs
a forward pass — the same architecture as the matched FM arm, just without loaded
weights. This is exactly what asparagus does for these arms (resolve_checkpoint
-> None -> weights never loaded), so a green run here means the +model=scratch_*
finetune path will instantiate.

  scratch_swinvit -> SmriFomo60kClsRegBackbone   (MONAI SwinViT-V2)
  scratch_vitb    -> SmriBrainiacClsRegBackbone  (MONAI ViT-B)
  scratch_nnunet  -> SmriSiam{ClsReg,Seg}Backbone (nnU-Net; needs SIAM_MODEL_DIR
                     for plans.json topology — no weights are read)

Usage:
    PYTHONPATH=/home/mhough/dev/smri-fm-fomo26/src \
      python scripts/smoke_scratch_arms.py
"""
from __future__ import annotations

import os
import sys

import torch

from asparagus_bridge.models_smri_fomo60k import SmriFomo60kClsRegBackbone
from asparagus_bridge.models_smri_brainiac import SmriBrainiacClsRegBackbone


def _check_clsreg(name: str, model, in_shape) -> bool:
    model.eval()
    x = torch.randn(*in_shape)
    with torch.no_grad():
        y = model(x)
    ok = tuple(y.shape) == (in_shape[0], model.num_classes)
    n = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] forward {tuple(x.shape)} -> {tuple(y.shape)}  "
          f"params={n/1e6:.1f}M  {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = []

    print(">>> scratch_swinvit (SwinViT-V2, random init)")
    results.append(_check_clsreg(
        "scratch_swinvit",
        SmriFomo60kClsRegBackbone(input_channels=1, output_channels=1),
        (1, 1, 96, 96, 96),
    ))

    print(">>> scratch_vitb (ViT-B, random init)")
    results.append(_check_clsreg(
        "scratch_vitb",
        SmriBrainiacClsRegBackbone(input_channels=1, output_channels=1),
        (1, 1, 96, 96, 96),
    ))

    print(">>> scratch_nnunet (nnU-Net, random init)")
    if not os.environ.get("SIAM_MODEL_DIR"):
        print("  SKIP: SIAM_MODEL_DIR unset (plans.json topology unavailable)")
    else:
        from asparagus_bridge.models_smri_siam import (
            SmriSiamClsRegBackbone,
            SmriSiamSegBackbone,
        )
        results.append(_check_clsreg(
            "scratch_nnunet/reg",
            SmriSiamClsRegBackbone(input_channels=1, output_channels=1),
            (1, 1, 64, 64, 64),
        ))
        seg = SmriSiamSegBackbone(input_channels=1, output_channels=2).eval()
        with torch.no_grad():
            yo = seg(torch.randn(1, 1, 64, 64, 64))
        out = yo[0] if isinstance(yo, (list, tuple)) else yo
        seg_ok = out.shape[1] == 2
        print(f"  [scratch_nnunet/seg] forward (1,1,64,64,64) -> {tuple(out.shape)}  "
              f"{'OK' if seg_ok else 'FAIL'}")
        results.append(seg_ok)

    ok = all(results)
    print(f"\n{'ALL OK' if ok else 'FAILURES PRESENT'} ({sum(results)}/{len(results)} passed)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
