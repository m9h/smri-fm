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
