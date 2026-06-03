"""Stub MedNeXt blocks: only referenced inside the (never-instantiated) MedNeXt
class that the fomo25 networks package imports at module load. Bare nn.Module
subclasses satisfy the class definitions without the full yucca framework."""

import torch.nn as nn


class MedNeXtBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class MedNeXtDownBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class MedNeXtUpBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class OutBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
