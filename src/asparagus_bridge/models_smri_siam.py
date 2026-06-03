"""Asparagus-compatible wrappers around the SIAM nnU-Net backbone.

SIAM (https://github.com/romainVala/SIAM) is a 3D nnU-Net v2 tissue
segmentation model trained on single-channel T1 at 0.75 mm isotropic. We wrap
its encoder (for cls/reg) and full encoder+decoder (for seg) so asparagus can
use it as a +model= overlay alongside smri_mae.

Required environment:
    SIAM_MODEL_DIR  path to the SIAM result folder containing plans.json,
                    dataset.json, and fold_0/checkpoint_final.pth. Populate
                    with `siam-pred`'s download helper or set manually.

Asparagus' BaseModule constructs this wrapper with no weights, then calls
load_state_dict(weights, strict=False). The companion converter in
`asparagus_bridge.checkpoint` produces a state_dict keyed under `model.` so
weights land in the right place.
"""

import json
import os
from pathlib import Path

import torch.nn as nn
from torch import Tensor

from asparagus_bridge.seg_inference import SlidingWindowSegMixin


def _resolve_siam_plans() -> tuple[dict, dict]:
    """Load plans.json and dataset.json from $SIAM_MODEL_DIR."""
    raw = os.environ.get("SIAM_MODEL_DIR")
    if not raw:
        raise RuntimeError(
            "SIAM_MODEL_DIR is not set. Point it at the SIAM result folder "
            "containing plans.json and fold_0/checkpoint_final.pth."
        )
    model_dir = Path(raw)
    plans = json.loads((model_dir / "plans.json").read_text())
    dataset_json = json.loads((model_dir / "dataset.json").read_text())
    return plans, dataset_json


def _build_siam_network(
    input_channels: int,
    output_channels: int,
    deep_supervision: bool,
):
    """Instantiate the SIAM nnU-Net architecture (encoder+decoder) from plans.

    Builds with the requested input/output channel counts so the same wrapper
    serves downstream tasks with different stem widths. Weights are loaded
    later by asparagus' BaseModule.load_state_dict; channel mismatches in the
    stem fall through silently under strict=False (or are handled by
    `repeat_stem_weights` in BaseModule).
    """
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans, _dataset_json = _resolve_siam_plans()
    plans_manager = PlansManager(plans)
    config_name = os.environ.get("SIAM_CONFIG", "3d_fullres")
    config = plans_manager.get_configuration(config_name)
    return get_network_from_plans(
        arch_class_name=config.network_arch_class_name,
        arch_kwargs=config.network_arch_init_kwargs,
        arch_kwargs_req_import=config.network_arch_init_kwargs_req_import,
        input_channels=input_channels,
        output_channels=output_channels,
        allow_init=True,
        deep_supervision=deep_supervision,
    )


class SmriSiamClsRegBackbone(nn.Module):
    """SIAM encoder + linear head for asparagus cls + reg downstream tasks.

    The encoder is the SIAM nnU-Net encoder pulled out of the full segmentation
    network. We GAP over the bottleneck feature map and feed a fresh linear
    head — same single-class-for-both pattern as SmriMaeClsRegBackbone.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        **_ignored,
    ):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels

        # Build the full SIAM net just to grab its encoder. The throwaway
        # output_channels is set to a small value to avoid allocating a wide
        # final seg head we never use.
        full_net = _build_siam_network(
            input_channels=input_channels,
            output_channels=1,
            deep_supervision=False,
        )
        self.encoder = full_net.encoder

        # nnU-Net v2 encoder exposes `output_channels` as a per-stage list;
        # the bottleneck is the last entry.
        bottleneck_dim = self.encoder.output_channels[-1]
        self.head = nn.Linear(bottleneck_dim, output_channels)

    def _features(self, x: Tensor) -> Tensor:
        """Encoder bottleneck → GAP → flat feature vector."""
        skips = self.encoder(x)
        # PlainConvEncoder / ResEncoder return a list of stage outputs.
        bottleneck = skips[-1] if isinstance(skips, (list, tuple)) else skips
        return bottleneck.mean(dim=(2, 3, 4))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self._features(x))

    def _encode(self, x: Tensor) -> Tensor:
        """Encoder output in the 5D format asparagus' linear probe expects."""
        feat = self._features(x)
        return feat[:, :, None, None, None]


class SmriSiamSegBackbone(SlidingWindowSegMixin, nn.Module):
    """Full SIAM nnU-Net for asparagus segmentation downstream tasks.

    Exposes `self.encoder` + `self.decoder` so the converter's
    `model.encoder.*` + `model.decoder.*` state-dict keys load cleanly. The
    decoder's final seg_layers are built fresh at the task's output_channels
    (the converter skips loading SIAM's own seg_layers).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str = "3D",
        deep_supervision: bool = False,
        **_ignored,
    ):
        super().__init__()
        assert dimensions == "3D", f"only 3D supported, got dimensions={dimensions}"
        self.num_classes = output_channels
        full_net = _build_siam_network(
            input_channels=input_channels,
            output_channels=output_channels,
            deep_supervision=deep_supervision,
        )
        self.encoder = full_net.encoder
        self.decoder = full_net.decoder

    def forward(self, x: Tensor) -> Tensor:
        skips = self.encoder(x)
        return self.decoder(skips)
