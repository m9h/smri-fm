"""Minimal stub of yucca's YuccaNet base class.

The vendored FOMO25 mmunetvae code only needs YuccaNet as the (unused) base of
its MedNeXt / UNet classes; our asparagus bridge instantiates
MultiModalUNetVAE (an nn.Module subclass) instead, so a bare nn.Module base is
sufficient. This keeps the vendored fomo25 `src` tree byte-for-byte faithful
without pulling in the full yucca framework.
"""

import torch.nn as nn


class YuccaNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
