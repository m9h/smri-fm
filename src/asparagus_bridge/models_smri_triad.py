"""Asparagus-compatible wrapper around the Triad SwinViT-V2 backbone.

Triad (wangshansong1, arXiv 2502.14064) is a MONAI SwinUNETR-style
SwinTransformer (V2) 3D encoder pretrained (MAE / SimMIM) on ~131k 3D MRI
volumes. Architecturally it is the FOMO60K SwinViT twin EXCEPT its depths are
(2,2,2,2) — a uniform 2-block stack — versus FOMO60K's (2,2,6,2). Same
feature_size 48, heads (3,6,12,24), window 7, patch 2, 1 channel. We wrap the
encoder and GAP its deepest hidden state into a fresh linear head, identical
readout to the FOMO60K bridge so the two Swin arms stay apples-to-apples.

Checkpoint layout (Triad-SwinB-{MAE,SimMIM}.pth) is a flat state_dict, every
tensor keyed `backbone.swinViT.*` — the bare MONAI SwinTransformer with the
use_v2 residual convs (`layers1c..4c`) and `downsample.*`. Unlike FOMO60K it
ships NO `mask_token_injector`, NO `patch_embed.patch_embed.` nesting, and NO
affine-norm tensors in `layersNc` (its use_v2 convs match MONAI's instance-norm
blocks directly), so the converter just re-prefixes `backbone.swinViT.` ->
`model.encoder.` with no dropping. self.head is the only missing tensor.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# MONAI SwinTransformer hyperparameters matching the Triad Swin-B pretrain
# (QuickStart.py::Swin). Differs from FOMO60K only in DEPTHS.
FEATURE_SIZE = 48
DEPTHS = (2, 2, 2, 2)
NUM_HEADS = (3, 6, 12, 24)
# Deepest hidden state (x4) is emitted after the 4th downsample:
# feature_size * 2**num_stages channels.
BOTTLENECK_DIM = FEATURE_SIZE * 2 ** len(DEPTHS)  # 768

_SRC_PREFIX = "backbone.swinViT."


def _build_triad_encoder(in_channels: int) -> nn.Module:
    from monai.networks.nets.swin_unetr import SwinTransformer

    return SwinTransformer(
        in_chans=in_channels,
        embed_dim=FEATURE_SIZE,
        window_size=(7, 7, 7),
        patch_size=(2, 2, 2),
        depths=DEPTHS,
        num_heads=NUM_HEADS,
        spatial_dims=3,
        use_v2=True,
        downsample="merging",
        patch_norm=False,
        mlp_ratio=4.0,
        qkv_bias=True,
    )


class SmriTriadClsRegBackbone(nn.Module):
    """Triad SwinViT-V2 encoder + linear head for asparagus cls + reg tasks."""

    def __init__(self, input_channels, output_channels, dimensions="3D", **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        self.encoder = _build_triad_encoder(in_channels=input_channels)
        self.head = nn.Linear(BOTTLENECK_DIM, output_channels)

    def _features(self, x: Tensor) -> Tensor:
        hidden_states = self.encoder(x, normalize=True)
        deepest = hidden_states[-1] if isinstance(hidden_states, (list, tuple)) else hidden_states
        return deepest.mean(dim=(2, 3, 4))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self._features(x))

    def _encode(self, x: Tensor) -> Tensor:
        feat = self._features(x)
        return feat[:, :, None, None, None]


def convert_triad_checkpoint(src_path, dst_path) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(raw, dict):
        raise ValueError(f"unrecognized Triad checkpoint structure at {src_path}")

    state_dict = {
        f"model.encoder.{k[len(_SRC_PREFIX):]}": v
        for k, v in raw.items()
        if k.startswith(_SRC_PREFIX)
    }
    if not state_dict:
        raise ValueError(
            f"no `{_SRC_PREFIX}*` weights found in {src_path}; got keys like {list(raw)[:5]}"
        )
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "epoch": epoch}, dst_path)


from .seg_inference import SlidingWindowSegMixin  # noqa: E402


class SmriTriadSegBackbone(SlidingWindowSegMixin, nn.Module):
    """Triad SwinV2 encoder + MONAI SwinUNETR decoder (re-parented; see fomo60k seg)."""

    stem_weight_name = "encoder.patch_embed.proj.weight"

    def __init__(self, input_channels, output_channels, dimensions="3D",
                 deep_supervision=False, **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        from monai.networks.nets import SwinUNETR
        sw = SwinUNETR(
            in_channels=input_channels, out_channels=output_channels,
            feature_size=FEATURE_SIZE, depths=DEPTHS, num_heads=NUM_HEADS,
            spatial_dims=3, use_v2=True, downsample="merging",
        )
        self.encoder = sw.swinViT
        self.decoder = nn.ModuleDict({
            "encoder1": sw.encoder1, "encoder2": sw.encoder2, "encoder3": sw.encoder3,
            "encoder4": sw.encoder4, "encoder10": sw.encoder10,
            "decoder5": sw.decoder5, "decoder4": sw.decoder4, "decoder3": sw.decoder3,
            "decoder2": sw.decoder2, "decoder1": sw.decoder1, "out": sw.out,
        })

    def forward(self, x):
        hs = self.encoder(x, normalize=True)
        d = self.decoder
        enc0 = d["encoder1"](x)
        enc1 = d["encoder2"](hs[0]); enc2 = d["encoder3"](hs[1]); enc3 = d["encoder4"](hs[2])
        dec4 = d["encoder10"](hs[4])
        dec3 = d["decoder5"](dec4, hs[3]); dec2 = d["decoder4"](dec3, enc3)
        dec1 = d["decoder3"](dec2, enc2); dec0 = d["decoder2"](dec1, enc1)
        out = d["decoder1"](dec0, enc0)
        return d["out"](out)


from .seg_decoders import UniformSegBackbone  # noqa: E402


class SmriTriadUniformSegBackbone(UniformSegBackbone):
    """Triad Swin encoder + shared uniform decoder."""
    stem_weight_name = "encoder.patch_embed.proj.weight"
    pyramid_channels = [FEATURE_SIZE, FEATURE_SIZE * 2, FEATURE_SIZE * 4, FEATURE_SIZE * 8, FEATURE_SIZE * 16]

    def __init__(self, input_channels, output_channels, dimensions="3D", deep_supervision=False,
                 decoder_kind="resnet_unet", **_ignored):
        assert dimensions == "3D"
        super().__init__(output_channels, decoder_kind=decoder_kind, input_channels=input_channels)
        self.encoder = _build_triad_encoder(in_channels=input_channels)

    def _pyramid(self, x):
        return list(self.encoder(x, normalize=True))[:5]
