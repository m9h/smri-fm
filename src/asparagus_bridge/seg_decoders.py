"""Lightweight U-Net decoder for bolting a seg head onto the ResNet-18 FM encoders
(anatcl, simclr3d). Pure torch (no MONAI block-API coupling), so it's robust across
versions. Channel/scale plan for a 3D ResNet-18:
  s0 (64, /2)  s1 (64, /4)  s2 (128, /8)  s3 (256, /16)  s4 (512, /32, bottleneck)
The decoder upsamples /32 -> /1 with skip concatenation, then a 1x1 to n_classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True), nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True), nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:  # guard odd dims from strided convs
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNetDecoder(nn.Module):
    """skips=[s0(64,/2), s1(64,/4), s2(128,/8), s3(256,/16), s4(512,/32)] -> logits (/1)."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.up4 = _UpBlock(512, 256, 256)  # /32 -> /16  + s3
        self.up3 = _UpBlock(256, 128, 128)  # /16 -> /8   + s2
        self.up2 = _UpBlock(128, 64, 64)    # /8  -> /4   + s1
        self.up1 = _UpBlock(64, 64, 64)     # /4  -> /2   + s0
        self.up0 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  # /2 -> /1 (no skip)
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, skips, full_shape):
        s0, s1, s2, s3, s4 = skips
        x = self.up4(s4, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        x = self.up0(x)
        if x.shape[2:] != full_shape:
            x = F.interpolate(x, size=full_shape, mode="trilinear", align_corners=False)
        return self.out(x)


# ---------------------------------------------------------------------------
# Uniform decoder: hold the decoder body byte-identical across encoders so the
# seg comparison reflects the ENCODER, not the decoder. Each encoder's 5-level
# pyramid (/2,/4,/8,/16,/32 — Swin and ResNet-18 both produce this) is passed
# through thin 1x1 channel adapters to a fixed common scheme, then the SAME
# ResNetUNetDecoder body. Only the adapters (minimal 1x1 capacity) differ by arm.
# ---------------------------------------------------------------------------
from .seg_inference import SlidingWindowSegMixin  # noqa: E402

_COMMON_CH = [64, 64, 128, 256, 512]  # decoder-body input channels, identical for every arm


class UniformUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels: int):
        super().__init__()
        assert len(encoder_channels) == 5
        self.adapters = nn.ModuleList(
            nn.Conv3d(ec, cc, kernel_size=1) for ec, cc in zip(encoder_channels, _COMMON_CH)
        )
        self.body = ResNetUNetDecoder(out_channels)  # the shared decoder body

    def forward(self, pyramid, full_shape):
        adapted = [a(f) for a, f in zip(self.adapters, pyramid)]
        return self.body(adapted, full_shape)


# ---------------------------------------------------------------------------
# Higher-capacity shared decoder: a byte-identical MONAI SwinUNETR decoder body
# (the exact blocks that gave the Swin arms ~0.75 natively), fed ANY encoder's
# 5-level pyramid through 1x1 channel adapters that map to [fs,2fs,4fs,8fs,16fs].
# Tests the uniform-decoder caveat — does a decoder that *can* exploit a strong
# encoder let the Swin/pretrained arms pull ahead, where the light U-Net body
# equalized everything? The decoder blocks depend only on (feature_size,
# in_channels), both fixed across arms -> body is identical; only adapters vary.
# ---------------------------------------------------------------------------
_SUNET_FS = 48  # SwinUNETR feature_size: target pyramid is [48,96,192,384,768]


class SwinUnetrSharedDecoder(nn.Module):
    def __init__(self, encoder_channels, out_channels: int, input_channels: int):
        super().__init__()
        assert len(encoder_channels) == 5
        from monai.networks.nets import SwinUNETR

        fs = _SUNET_FS
        sw = SwinUNETR(
            in_channels=input_channels, out_channels=out_channels, feature_size=fs,
            depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), spatial_dims=3,
            use_v2=True, downsample="merging",
        )
        # keep only the decoder blocks; sw.swinViT (the encoder) is discarded — each
        # arm brings its own encoder. naming under self.decoder.* keeps asparagus's
        # encoder/decoder LR split keyed on the `model.decoder` prefix.
        self.blocks = nn.ModuleDict({
            "encoder1": sw.encoder1, "encoder2": sw.encoder2, "encoder3": sw.encoder3,
            "encoder4": sw.encoder4, "encoder10": sw.encoder10,
            "decoder5": sw.decoder5, "decoder4": sw.decoder4, "decoder3": sw.decoder3,
            "decoder2": sw.decoder2, "decoder1": sw.decoder1, "out": sw.out,
        })
        target = [fs, 2 * fs, 4 * fs, 8 * fs, 16 * fs]
        self.adapters = nn.ModuleList(
            nn.Conv3d(ec, tc, kernel_size=1) for ec, tc in zip(encoder_channels, target)
        )

    def forward(self, pyramid, x_input, full_shape):
        a = [ad(p) for ad, p in zip(self.adapters, pyramid)]  # -> [fs,2fs,4fs,8fs,16fs]
        # MONAI's UnetrUpBlock cats are rigid (no interpolate), so the pyramid must sit
        # at the canonical SwinUNETR scales /2,/4,/8,/16,/32. Some stems (e.g. MONAI
        # resnet18) shift an octave; resize each level to its canonical fraction of the
        # input. No-op for already-aligned encoders -> decoder stays identical across arms.
        for i, s in enumerate((2, 4, 8, 16, 32)):
            tgt = tuple(max(1, d // s) for d in full_shape)
            if a[i].shape[2:] != tgt:
                a[i] = F.interpolate(a[i], size=tgt, mode="trilinear", align_corners=False)
        b = self.blocks
        enc0 = b["encoder1"](x_input)            # full-res skip from the raw image
        enc1 = b["encoder2"](a[0])
        enc2 = b["encoder3"](a[1])
        enc3 = b["encoder4"](a[2])
        dec4 = b["encoder10"](a[4])
        dec3 = b["decoder5"](dec4, a[3])
        dec2 = b["decoder4"](dec3, enc3)
        dec1 = b["decoder3"](dec2, enc2)
        dec0 = b["decoder2"](dec1, enc1)
        out = b["decoder1"](dec0, enc0)
        logits = b["out"](out)
        if logits.shape[2:] != full_shape:
            logits = F.interpolate(logits, size=full_shape, mode="trilinear", align_corners=False)
        return logits


class UniformSegBackbone(SlidingWindowSegMixin, nn.Module):
    """Generic FM-encoder + a shared decoder. Subclasses build self.encoder, set
    pyramid_channels + stem_weight_name, and implement _pyramid(x)->[s0..s4].

    decoder_kind selects the shared body: "resnet_unet" (light, default) or
    "swinunetr" (high-capacity). Both are byte-identical across arms; only the 1x1
    adapters differ. The swinunetr body also consumes the raw input for its full-res
    skip, so forward passes x through."""

    pyramid_channels: list = []  # set by subclass

    def __init__(self, output_channels: int, decoder_kind: str = "resnet_unet",
                 input_channels: int = 1):
        super().__init__()
        self.num_classes = output_channels
        self.decoder_kind = decoder_kind
        if decoder_kind == "swinunetr":
            self.decoder = SwinUnetrSharedDecoder(self.pyramid_channels, output_channels, input_channels)
        elif decoder_kind == "resnet_unet":
            self.decoder = UniformUNetDecoder(self.pyramid_channels, output_channels)
        else:
            raise ValueError(f"unknown decoder_kind {decoder_kind!r}")

    def _pyramid(self, x):  # -> [s0(/2), s1(/4), s2(/8), s3(/16), s4(/32)]
        raise NotImplementedError

    def forward(self, x):
        pyr = self._pyramid(x)
        if self.decoder_kind == "swinunetr":
            return self.decoder(pyr, x, x.shape[2:])
        return self.decoder(pyr, x.shape[2:])
