"""Asparagus-compatible wrapper around the 3D-Neuro-SimCLR ResNet-18 backbone.

3D-Neuro-SimCLR (Kaczmarek et al., arXiv 2509.10620) is a MONAI 3D ResNet-18
(spatial_dims=3, n_input_channels=1) pretrained with the standard SimCLR
NT-Xent contrastive objective on RAW T1w brain MRI (turboprep / MNI152
preprocessing). DOMAIN MATCH: unlike AnatCL (VBM GM maps), this FM is
pretrained on raw T1 like FOMO26, so it is a closer-domain transfer arm.

We build the encoder from `monai.networks.nets.resnet18` with exactly the args
the pretrain repo used (`get_resnet("resnet18")` -> resnet18(spatial_dims=3,
n_input_channels=1, num_classes=0)) and, like the repo's SimCLR wrapper, replace
the final `fc` with Identity so the encoder emits the 512-d GAP feature. A fresh
linear head is attached for asparagus cls/reg tasks; the SimCLR projector MLP is
dropped.

Checkpoint layout (GitHub release `simclr_3d_brain_foundation.tar`) is a SimCLR
training checkpoint: `{'epoch', 'model_state_dict', 'optimizer_state_dict'}`.
The `model_state_dict` carries `encoder.*` (the ResNet) + `projector.*` (the
2-layer MLP, dropped) and may have a leading `module.` from DataParallel.
convert_simclr3d_checkpoint strips `module.`, keeps only tensor `encoder.*`
weights, and re-emits them as `model.encoder.*`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from torch import Tensor

ENCODER_DIM = 512


def _build_simclr3d_encoder(in_channels: int) -> nn.Module:
    enc = resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=0)
    enc.fc = nn.Identity()  # mirror SimCLR: GAP feature, no fc
    return enc


class SmriSimclr3dClsRegBackbone(nn.Module):
    """3D-Neuro-SimCLR ResNet-18 encoder + linear head for asparagus cls + reg."""

    def __init__(self, input_channels, output_channels, dimensions="3D", **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        self.encoder = _build_simclr3d_encoder(in_channels=input_channels)
        self.head = nn.Linear(ENCODER_DIM, output_channels)

    def _features(self, x: Tensor) -> Tensor:
        return self.encoder(x)  # [B, 512]

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self._features(x))

    def _encode(self, x: Tensor) -> Tensor:
        feat = self._features(x)
        return feat[:, :, None, None, None]


def convert_simclr3d_checkpoint(src_path, dst_path) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(raw, dict):
        raise ValueError(f"unrecognized SimCLR checkpoint structure at {src_path}")

    def strip_module(k: str) -> str:
        return k[len("module."):] if k.startswith("module.") else k

    state_dict = {
        f"model.{strip_module(k)}": v
        for k, v in raw.items()
        if strip_module(k).startswith("encoder.") and torch.is_tensor(v)
    }
    if not state_dict:
        raise ValueError(
            f"no `encoder.*` tensors found in {src_path}; got keys like {list(raw)[:5]}"
        )
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "epoch": epoch}, dst_path)


from .seg_inference import SlidingWindowSegMixin  # noqa: E402
from .seg_decoders import ResNetUNetDecoder  # noqa: E402


class SmriSimclr3dSegBackbone(SlidingWindowSegMixin, nn.Module):
    """3D-Neuro-SimCLR ResNet-18 (MONAI) encoder + ResNet-UNet decoder for seg."""

    stem_weight_name = "encoder.conv1.weight"

    def __init__(self, input_channels, output_channels, dimensions="3D",
                 deep_supervision=False, **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        self.encoder = _build_simclr3d_encoder(in_channels=input_channels)
        self.decoder = ResNetUNetDecoder(output_channels)

    def forward(self, x):
        e = self.encoder  # MONAI resnet18: conv1 -> bn1 -> act -> maxpool -> layer1..4
        s0 = e.act(e.bn1(e.conv1(x)))    # /2,  64
        s1 = e.layer1(e.maxpool(s0))     # /4,  64
        s2 = e.layer2(s1)                # /8,  128
        s3 = e.layer3(s2)                # /16, 256
        s4 = e.layer4(s3)                # /32, 512
        return self.decoder([s0, s1, s2, s3, s4], x.shape[2:])
