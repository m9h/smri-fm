"""Sliding-window dense inference for the FM seg backbones.

asparagus's SegmentationModule.test_step / predict_step call
``model.sliding_window_predict(data, patch_size, overlap)`` — a method that
gardening_tools' native UNet provides but our FM seg backbones (SIAM,
mmunetvae) do not, since they only implement ``forward``. This mixin reproduces
the gardening_tools UNet tiling + accumulation contract exactly: the same
``get_steps_for_sliding_window`` step computation, the same zero canvas with
plain summation of patch logits (per-voxel every class channel is summed the
same number of times, so argmax is unaffected by overlap multiplicity), and the
same optional 8-way flip TTA. Using the identical tiling means SIAM and
mmunetvae tile identically both to each other and to asparagus's reference seg
nets, so Dice is comparable across arms.

Backbones must define ``self.num_classes`` and a ``forward(x) -> [B,
num_classes, *spatial]`` returning dense logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# gardening_tools is a container-only dep (installed by finetune_seg_arm.sh),
# same as the rest of the asparagus seg stack — importing at module top keeps
# the exact step function asparagus's own seg nets use.
from gardening_tools.modules.networks.utils import get_steps_for_sliding_window


class SlidingWindowSegMixin:
    """Adds gardening_tools-compatible sliding_window_predict to a seg backbone."""

    # Minimum tiling patch per spatial dim. Must be >= 2 * 2**(num_pool_stages)
    # so the encoder bottleneck keeps >1 voxel per dim, else InstanceNorm raises
    # "Expected more than 1 spatial element". SIAM's nnU-Net pools 5x (needs 64);
    # mmunetvae pools 4x (needs 32). 64 is the safe floor for both arms.
    MIN_INFERENCE_PATCH = 64

    def _sliding_window_predict3D(self, data: Tensor, patch_size, overlap: float) -> Tensor:
        canvas = torch.zeros((1, self.num_classes, *data.shape[2:]), device=data.device)
        x_steps, y_steps, z_steps = get_steps_for_sliding_window(data.shape[2:], patch_size, overlap)
        px, py, pz = patch_size
        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    out = self._forward_divisible(data[:, :, xs:xs + px, ys:ys + py, zs:zs + pz])
                    canvas[:, :, xs:xs + px, ys:ys + py, zs:zs + pz] += out
        return canvas

    # Encoder/decoder strided U-Nets require each spatial dim divisible by the
    # cumulative pooling stride, else decoder upsampling and the encoder skip
    # mismatch by a voxel ("Expected size 4 but got size 3"). fit_patch_size_to_image_size
    # can hand us a tile (e.g. 112 or a 73-derived dim) that isn't, so pad the tile
    # up to a multiple of POOL_STRIDE, run forward, then crop the logits back.
    # SIAM's 3d_fullres plan is anisotropic — cumulative stride 64x64x32 (6 of 7
    # stages stride-2 in x/y, 5 in z). 64 is divisible by all of those and by
    # mmunetvae's 16, so a uniform pad-to-64 makes every tile decoder-safe.
    POOL_STRIDE = 64

    def _forward_divisible(self, patch: Tensor) -> Tensor:
        sp = patch.shape[2:]
        pads = [(0, (-d) % self.POOL_STRIDE) for d in sp]
        if any(hi for _, hi in pads):
            flat = []
            for lo, hi in reversed(pads):  # F.pad consumes dims last-to-first
                flat += [lo, hi]
            patch = F.pad(patch, flat)
        out = self.forward(patch)
        return out[:, :, : sp[0], : sp[1], : sp[2]]

    def sliding_window_predict(self, data: Tensor, patch_size, overlap: float, mirror: bool = False) -> Tensor:
        assert len(patch_size) == 3, f"only 3D patches supported, got patch_size={patch_size}"
        # asparagus floors the fitted inference patch to a multiple of 32
        # (fit_patch_size_to_image_size); thin-slice tasks like SEG009's ~29-slice
        # FLAIR collapse a dim to 0, which makes get_steps_for_sliding_window's
        # step 0 and crashes — and would equally crash asparagus's native seg
        # nets. Enforce a >=MIN_INFERENCE_PATCH patch per dim and zero-pad the image
        # up to the patch (exactly what the training data loader does), tile, then
        # crop the logits back to the original extent. Thick tasks (SEG010) need no
        # padding so their behavior is unchanged.
        patch_size = [max(int(p), self.MIN_INFERENCE_PATCH) for p in patch_size]
        spatial = tuple(data.shape[2:])
        pads = [(max(p - d, 0) // 2, max(p - d, 0) - max(p - d, 0) // 2) for d, p in zip(spatial, patch_size)]
        if any(lo or hi for lo, hi in pads):
            flat = []
            for lo, hi in reversed(pads):  # F.pad consumes dims last-to-first
                flat += [lo, hi]
            data = F.pad(data, flat)

        pred = self._sliding_window_predict3D(data, patch_size, overlap)
        if mirror:
            for dims in [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                pred += torch.flip(
                    self._sliding_window_predict3D(torch.flip(data, dims), patch_size, overlap), dims
                )
            pred /= 8

        crop = [slice(None), slice(None)] + [slice(lo, lo + d) for (lo, _), d in zip(pads, spatial)]
        return pred[tuple(crop)]
