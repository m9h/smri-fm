"""Asparagus-compatible wrapper around the FOMO60K SwinViT-V2 backbone.

FOMO60K (pkoutsouvelis) is a MONAI SwinUNETR-style SwinTransformer (V2) 3D
encoder pretrained with a masked-autoencoder objective on ~60k brain volumes
(feature_size 48, depths 2/2/6/2, heads 3/6/12/24, window 7, patch 2, 1 channel,
96^3). We wrap the encoder and GAP its deepest hidden state into a fresh linear
head so asparagus can use it as a +model= overlay alongside smri_mae / smri_siam.

The pretrained checkpoint nests the MONAI SwinTransformer one level deeper than
MONAI itself (pkoutsouvelis' MAE wrapper adds `patch_embed.` around MONAI's own
`patch_embed`, plus a `mask_token_injector`). convert_fomo60k_checkpoint remaps
those keys back onto a bare MONAI SwinTransformer:
  - keep only `model.encoder.*`
  - drop `mask_token_injector` (MAE-only)
  - collapse `patch_embed.patch_embed.` -> `patch_embed.`
  - pass through `layers1..4.*`, the use_v2 residual convs (`layers1c..4c`),
    and `downsample.*` unchanged
and re-emit under `model.encoder.*` so asparagus' BaseModule (which strips the
leading `model.` then load_state_dict(strict=False)) lands them in self.encoder.
self.head is the only missing tensor.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# MONAI's SwinTransformer hardcodes norm_name="instance" (affine=False) for the
# use_v2 residual conv blocks, so those blocks register no learnable norm
# params. The FOMO60K checkpoint was trained with affine norms, so it ships
# layersNc.*.layer.norm{1,2}.{weight,bias} that have nowhere to land. Drop them
# explicitly (the residual conv weights — the substance — still load). Same
# spirit as the SIAM bridge dropping its stem/seg tensors.
_DROP_RE = re.compile(r"layers\dc\.\d+\.layer\.norm[12]\.")

# MONAI SwinTransformer hyperparameters matching the FOMO60K pretrain config.
FEATURE_SIZE = 48
DEPTHS = (2, 2, 6, 2)
NUM_HEADS = (3, 6, 12, 24)
# MONAI's SwinTransformer returns hidden states [x0..x4]; the deepest (x4) is
# emitted after the 4th downsample, so feature_size * 2**num_stages channels.
BOTTLENECK_DIM = FEATURE_SIZE * 2 ** len(DEPTHS)  # 768


def _build_fomo60k_encoder(in_channels: int) -> nn.Module:
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


class SmriFomo60kClsRegBackbone(nn.Module):
    """FOMO60K SwinViT-V2 encoder + linear head for asparagus cls + reg tasks."""

    def __init__(self, input_channels, output_channels, dimensions="3D", **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        self.encoder = _build_fomo60k_encoder(in_channels=input_channels)
        self.head = nn.Linear(BOTTLENECK_DIM, output_channels)

    def _features(self, x: Tensor) -> Tensor:
        """Deepest SwinTransformer hidden state -> GAP -> flat feature vector."""
        hidden_states = self.encoder(x, normalize=True)
        deepest = hidden_states[-1] if isinstance(hidden_states, (list, tuple)) else hidden_states
        return deepest.mean(dim=(2, 3, 4))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self._features(x))

    def _encode(self, x: Tensor) -> Tensor:
        feat = self._features(x)
        return feat[:, :, None, None, None]


def convert_fomo60k_checkpoint(src_path, dst_path) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(raw, dict):
        raise ValueError(f"unrecognized FOMO60K checkpoint structure at {src_path}")

    enc_prefix = "model.encoder."
    state_dict = {}
    for k, v in raw.items():
        if not k.startswith(enc_prefix):
            continue
        name = k[len(enc_prefix):]
        if "mask_token_injector" in name:
            continue
        if _DROP_RE.search(name):
            continue
        if name.startswith("patch_embed.patch_embed."):
            name = "patch_embed." + name[len("patch_embed.patch_embed."):]
        state_dict[f"model.encoder.{name}"] = v

    if not state_dict:
        raise ValueError(
            f"no `model.encoder.*` weights found in {src_path}; got keys like {list(raw)[:5]}"
        )
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "epoch": epoch}, dst_path)
