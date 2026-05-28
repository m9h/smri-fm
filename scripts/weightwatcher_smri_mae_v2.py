"""weightwatcher analysis of MedARC's smri_mae MaskedViT encoder.

Mirrors scripts/weightwatcher_siam_v2.py so the two FMs land in the same
results dir under comparable column conventions. MaskedViT's layers are pure
nn.Linear (Patchify3D is a reshape, not a learnable Conv3D), so ww's stock
Linear path covers everything — the Conv3D patch we added for SIAM isn't
exercised here. The patched ww fork is still preferred (no behavior change for
Linear) so the column set lines up exactly with the SIAM run.

If SMRI_MAE_CKPT is set, pretrained weights are loaded before analysis;
otherwise the run reports random-init metrics, which are useful as a
calibration floor (alpha should sit well *above* the trained α, log_norm well
below the trained value).

Usage:
    PYTHONPATH=/home/mhough/dev/weightwatcher:/workspace/smri-fm/src \
      [SMRI_MAE_CKPT=/path/to/asparagus_compatible_weights.pth] \
      python scripts/weightwatcher_smri_mae_v2.py [out_dir]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from asparagus_bridge.models_smri_mae import SmriMaeClsRegBackbone


# Match the asparagus smoke-test target so layer dims line up with the actual
# fine-tuning configuration.
IMG_SIZE = (208, 240, 208)
PATCH_SIZE = 16


def _maybe_load_ckpt(model: torch.nn.Module) -> str:
    ckpt_spec = os.environ.get("SMRI_MAE_CKPT")
    if not ckpt_spec:
        return "random init (SMRI_MAE_CKPT unset) — results are an init-floor reference"
    path = Path(ckpt_spec)
    if not path.exists():
        return f"WARNING: {ckpt_spec} not found, falling back to random init"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    raw = blob.get("state_dict") or blob.get("model") or blob
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in raw.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    return f"loaded {len(sd)} tensors  missing={len(list(miss))}  unexpected={len(list(unexp))}"


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/fomo26_fm_benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> building SmriMaeClsRegBackbone(in=1, out=1, img_size={IMG_SIZE}, patch_size={PATCH_SIZE})")
    model = SmriMaeClsRegBackbone(
        input_channels=1,
        output_channels=1,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
    )
    print(f"    ckpt: {_maybe_load_ckpt(model)}")

    import weightwatcher as ww

    print(">>> ww.analyze(encoder)  [stock Linear path, patched fork for column parity with SIAM]")
    watcher = ww.WeightWatcher(model=model.encoder)
    details = watcher.analyze()
    summary = watcher.get_summary(details)

    details_path = out_dir / "ww_v2_smri_mae_details.csv"
    summary_path = out_dir / "ww_v2_smri_mae_summary.json"
    details.to_csv(details_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    print()
    print(f"layers analyzed: {len(details)}")
    if "alpha" in details.columns and len(details) > 0:
        print(f"alpha:           mean={details['alpha'].mean():.3f}  median={details['alpha'].median():.3f}")
        if "alpha_weighted" in details.columns:
            print(f"alpha_weighted:  mean={details['alpha_weighted'].mean():.3f}")
        if "stable_rank" in details.columns:
            print(f"stable_rank:     mean={details['stable_rank'].mean():.1f}")
        if "log_norm" in details.columns:
            print(f"log_norm:        mean={details['log_norm'].mean():.3f}")
        well_trained = (details["alpha"] < 2.0).mean()
        in_zone = (details["alpha"] < 3.0).mean()
        print(f"layers with alpha<2 (heavy-tail, well-trained): {well_trained:.0%}")
        print(f"layers with alpha<3 (reasonable):               {in_zone:.0%}")
    print()
    print(f"wrote {details_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
