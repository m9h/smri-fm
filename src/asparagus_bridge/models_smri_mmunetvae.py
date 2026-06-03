"""Asparagus-compatible wrapper around the FOMO25-winning Multi-Modal U-Net VAE.

FOMO25 mmunetvae (Banuelos/Gordaliza et al., SSL3D + FOMO25 winner) is a
multi-modal 3D U-Net with a VAE bottleneck that splits each modality's pooled
multi-scale encoder features into a shared (subject) latent and a
modality-specific latent. It was self-supervised on FOMO60K raw T1/other MRI
(masked reconstruction, "mae" mode). For asparagus regression we run the model
in "regression" mode (num_modalities = input_channels), which concatenates the
mean shared latent with the modality latents and feeds a fresh ClsRegHead
(GAP -> Linear -> SiLU -> Dropout -> Linear). DOMAIN MATCH: pretrained on raw
FOMO60K MRI -> same domain as FOMO26, the most task-aligned arm.

We vendor the FOMO25 `src` tree under third_party/fomo25_mmunetvae (extracted
from the published docker image jbanusco/sslmmunetave:1.0.0, which is amd64-only
and cannot run on the GB10). A tiny `yucca_stub` satisfies the framework imports
that only the unused MedNeXt/UNet classes and sliding-window predict() need.

Checkpoint (fomo25_mmunetvae_pretrained.ckpt) is a Lightning checkpoint whose
`state_dict` carries 74 `model.*` tensors: `model.encoder.*` (the shared U-Net
encoder), `model.decoder.*` (the SSL reconstruction decoder, kept but unused at
regression time), and the four `model.conv_{mu,logvar}_{shared,modality}.*` VAE
projection convs. The regression task head (decoder_task / ClsRegHead) is fresh.
convert_mmunetvae_checkpoint re-emits each `model.<x>` as `model.net.<x>` so that
after asparagus strips the leading `model.` the keys land in self.net.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# Vendored FOMO25 source + yucca stub on sys.path (self-contained import).
_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "fomo25_mmunetvae"
for _p in (_THIRD_PARTY / "src", _THIRD_PARTY / "yucca_stub"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from models.networks.mmunetvae import mmunetvae  # noqa: E402

from asparagus_bridge.seg_inference import SlidingWindowSegMixin  # noqa: E402


class SmriMmunetvaeClsRegBackbone(nn.Module):
    """FOMO25 Multi-Modal U-Net VAE (regression mode) for asparagus cls + reg.

    asparagus uses one backbone class for both cls and reg; structurebench's only
    task is REGR002 brain-age, so we instantiate the network in "regression" mode.
    """

    def __init__(self, input_channels, output_channels, dimensions="3D", **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        self.net = mmunetvae(
            input_channels=input_channels,
            output_channels=output_channels,
            mode="regression",
            use_vae=True,
            use_skip_connections=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x_list=x)["task_output"]  # [B, num_classes]


class SmriMmunetvaeSegBackbone(SlidingWindowSegMixin, nn.Module):
    """FOMO25 Multi-Modal U-Net VAE (segmentation mode) for asparagus dense tasks.

    Runs the native `segmentation` mode: the pretrained shared `encoder` + the
    four VAE projection convs (`conv_{mu,logvar}_{shared,modality}`) load from the
    converted checkpoint and supply the features; the per-modality `decoder_task`
    (a UNetDecoder built fresh at the task's output_channels) + the learnable
    late-`fusion_layer` are task-specific and trained from scratch — analogous to
    SIAM's fresh seg_layers. `use_skip_connections=True` so the fresh decoder
    consumes the pretrained encoder's multi-scale skips (a real U-Net), not just
    the VAE bottleneck latent. The unused SSL recon `decoder` (output_channels=1)
    still loads its pretrained weights but is not on the task forward path.
    """

    def __init__(self, input_channels, output_channels, dimensions="3D",
                 deep_supervision=False, **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        assert not deep_supervision, "mmunetvae seg does not emit deep supervision"
        self.num_classes = output_channels
        net = mmunetvae(
            input_channels=input_channels,
            output_channels=output_channels,
            mode="segmentation",
            use_vae=True,
            use_skip_connections=True,
        )
        # Promote the network's direct children to top-level attributes so their
        # parameter names become model.<child>.* . asparagus only finetunes with a
        # differential warmup when it can split params into encoder (pretrained,
        # slow) and decoder (fresh, fast) groups by substring matching
        # "model.encoder"/"model.decoder" (functional/lr_scheduling.py); with
        # everything buried under self.net the decoder group is empty and it
        # asserts. Aliases are registered BEFORE self.net so named_parameters()
        # (which dedups) yields only the short names, while self.state_dict()
        # (which does not dedup) still also carries the model.net.* keys that
        # convert_mmunetvae_checkpoint writes — so checkpoint loading is unchanged.
        # decoder_task is the fresh seg head (fast group); encoder + the VAE
        # projection convs are pretrained (slow / default group).
        for name, child in list(net.named_children()):
            setattr(self, name, child)
        self.net = net  # registered last; drives the forward orchestration

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x_list=x)["task_output"]  # [B, num_classes, D, H, W]


def convert_mmunetvae_checkpoint(src_path, dst_path) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(raw, dict):
        raise ValueError(f"unrecognized mmunetvae checkpoint structure at {src_path}")

    state_dict = {}
    for k, v in raw.items():
        if not torch.is_tensor(v):
            continue
        inner = k[len("model."):] if k.startswith("model.") else k
        state_dict[f"model.net.{inner}"] = v
    if not state_dict:
        raise ValueError(
            f"no tensors found in {src_path}; got keys like {list(raw)[:5]}"
        )
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "epoch": epoch}, dst_path)
