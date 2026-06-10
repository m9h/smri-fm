"""Asparagus-compatible wrapper around the AnatCL 3D-ResNet-18 backbone.

AnatCL (EIDOSLAB, Barbano et al., Pattern Recognition Letters 2026) is a 3D
ResNet-18 pretrained with a weakly-supervised anatomical contrastive (y-aware)
objective on OpenBHB. We vendor the exact ResNet-18 3D it uses (BasicBlock
[2,2,2,2], 7^3 stride-2 stem, BatchNorm3d, AdaptiveAvgPool3d -> 512-d feature)
so the container needs no `anatcl` package — only the weights file — then feed
the GAP feature to a fresh linear head for asparagus cls/reg tasks.

DOMAIN CAVEAT: AnatCL was pretrained on cat12 VBM gray-matter maps at
121x128x121, NOT raw T1w intensities. Feeding FOMO26 raw-T1 crops is
out-of-distribution; AdaptiveAvgPool3d makes any input size dimensionally fine,
and we finetune the whole net, but expect this arm to lean harder on finetuning
than the T1-pretrained FMs. It is still a legitimate "does this FM transfer"
benchmark arm.

Checkpoint layout (gitlab anatcl-pretrained `weights.pth`) is a training
checkpoint dict: `{'model': state_dict, 'optimizer', 'age_estimator',
'site_estimator', ...}`. The `model` state_dict carries 120 `encoder.*`
(the ResNet) + 4 `head.*` (the projection MLP, dropped). It was pickled
referencing a top-level `models` module from the training script and embeds
sklearn estimator objects, so convert_anatcl_checkpoint loads it through a
custom unpickler that stubs `models.*` (sklearn loads normally) and keeps only
the tensor `encoder.*` weights, re-emitted as `model.encoder.*`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

ENCODER_DIM = 512


# --- vendored 3D ResNet-18 (faithful to anatcl/models/resnet3d.py) -----------
def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _ResNet3D(nn.Module):
    def __init__(self, block, layers, in_channels=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def _resnet18_3d(in_channels: int) -> nn.Module:
    return _ResNet3D(_BasicBlock, [2, 2, 2, 2], in_channels=in_channels)


class SmriAnatclClsRegBackbone(nn.Module):
    """AnatCL 3D-ResNet-18 encoder + linear head for asparagus cls + reg tasks."""

    def __init__(self, input_channels, output_channels, dimensions="3D", **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        self.encoder = _resnet18_3d(in_channels=input_channels)
        self.head = nn.Linear(ENCODER_DIM, output_channels)

    def _features(self, x: Tensor) -> Tensor:
        return self.encoder(x)  # [B, 512]

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self._features(x))

    def _encode(self, x: Tensor) -> Tensor:
        feat = self._features(x)
        return feat[:, :, None, None, None]


class _StubAny:
    """Placeholder for classes pickled from the training script's `models`
    module; we only extract tensors so these are never instantiated."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __reduce__(self):
        return (_StubAny, ())


class _AnatclUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "models" or module.startswith("models."):
            return _StubAny
        return super().find_class(module, name)


class _AnatclPickleModule:  # torch.load pickle_module shim
    Unpickler = _AnatclUnpickler
    load = staticmethod(pickle.load)


def convert_anatcl_checkpoint(src_path, dst_path) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    ckpt = torch.load(
        src_path, map_location="cpu", weights_only=False,
        pickle_module=_AnatclPickleModule,
    )
    raw = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(raw, dict):
        raise ValueError(f"unrecognized AnatCL checkpoint structure at {src_path}")

    state_dict = {
        f"model.{k}": v
        for k, v in raw.items()
        if k.startswith("encoder.") and torch.is_tensor(v)
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


class SmriAnatclSegBackbone(SlidingWindowSegMixin, nn.Module):
    """AnatCL 3D-ResNet-18 encoder + ResNet-UNet decoder for asparagus seg.

    self.encoder is the same _ResNet3D as the ClsReg wrapper, so the existing
    convert_anatcl_checkpoint's `model.encoder.*` keys load straight in; the
    fresh U-Net decoder lives under self.decoder (asparagus encoder/decoder split).
    """

    stem_weight_name = "encoder.conv1.weight"  # 1->n channel stem repeat

    def __init__(self, input_channels, output_channels, dimensions="3D",
                 deep_supervision=False, **_ignored):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        self.encoder = _resnet18_3d(in_channels=input_channels)
        self.decoder = ResNetUNetDecoder(output_channels)

    def forward(self, x):
        e = self.encoder
        s0 = e.relu(e.bn1(e.conv1(x)))   # /2,  64
        s1 = e.layer1(e.maxpool(s0))     # /4,  64
        s2 = e.layer2(s1)                # /8,  128
        s3 = e.layer3(s2)                # /16, 256
        s4 = e.layer4(s3)                # /32, 512
        return self.decoder([s0, s1, s2, s3, s4], x.shape[2:])
