"""Asparagus-compatible wrapper around the BrainIAC ViT backbone.

BrainIAC (AIM-KannLab) is a MONAI ViT-B (img 96^3, patch 16, hidden 768, 12
layers, 12 heads) pretrained with SimCLR on single-channel 3D MRI. We wrap its
encoder (the MONAI ViT) and feed the CLS token to a fresh linear head so
asparagus can use it as a +model= overlay alongside smri_mae / smri_siam.

Mirrors BrainIAC/src/model.py::ViTBackboneNet but builds the bare MONAI ViT
WITHOUT loading weights at __init__ (asparagus instantiates with no weights then
load_state_dict(strict=False)). convert_brainiac_checkpoint emits model.encoder.*
keys so the 137 pretrained tensors land in self.encoder; self.head is the fresh
task head.

Note: MONAI 1.5.2's TransformerBlock unconditionally instantiates cross-attention
submodules (blocks.*.cross_attn.*, blocks.*.norm_cross_attn.*), but ViT leaves
with_cross_attention=False so they are never invoked in forward. The pretrained
BrainIAC ViT predates them, so they stay at init — inert dead weight, never
exercised, zero gradient (not reached in forward). All substantive pretrained
weights (patch_embed, self-attn qkv/proj, MLP, norms, CLS/pos embeds) load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

_BRAINIAC_SRC = "/home/mhough/dev/BrainIAC/src"
if _BRAINIAC_SRC not in sys.path:
    sys.path.insert(0, _BRAINIAC_SRC)

BRAINIAC_VIT_KWARGS = dict(
    in_channels=1,
    img_size=(96, 96, 96),
    patch_size=(16, 16, 16),
    hidden_size=768,
    mlp_dim=3072,
    num_layers=12,
    num_heads=12,
    save_attn=True,
)
BRAINIAC_HIDDEN_SIZE = 768


def _build_brainiac_vit(in_channels: int) -> nn.Module:
    from monai.networks.nets import ViT  # BrainIAC dep; imported lazily

    kwargs = dict(BRAINIAC_VIT_KWARGS)
    kwargs["in_channels"] = in_channels
    return ViT(**kwargs)


class SmriBrainiacClsRegBackbone(nn.Module):
    def __init__(self, input_channels, output_channels, dimensions="3D", **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        self.encoder = _build_brainiac_vit(in_channels=input_channels)
        self.head = nn.Linear(BRAINIAC_HIDDEN_SIZE, output_channels)

    def _features(self, x: Tensor) -> Tensor:
        out = self.encoder(x)  # MONAI ViT -> (x, hidden_states_out)
        tokens = out[0] if isinstance(out, (list, tuple)) else out
        return tokens[:, 0]  # CLS token, [B, 768]

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self._features(x))

    def _encode(self, x: Tensor) -> Tensor:
        feat = self._features(x)
        return feat[:, :, None, None, None]


def convert_brainiac_checkpoint(src_path, dst_path) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(raw, dict):
        raise ValueError(f"unrecognized BrainIAC checkpoint structure at {src_path}")
    prefix = "backbone."
    state_dict = {
        f"model.encoder.{k[len(prefix):]}": v
        for k, v in raw.items()
        if k.startswith(prefix)
    }
    if not state_dict:
        raise ValueError(
            f"no `backbone.*` weights found in {src_path}; got keys like {list(raw)[:5]}"
        )
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "epoch": epoch}, dst_path)
