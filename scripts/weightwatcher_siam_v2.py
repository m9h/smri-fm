"""Re-run the SIAM weightwatcher analysis using the Conv3D-patched ww fork.

Companion to scripts/weightwatcher_siam.py (which fails on Conv3D in upstream
ww). Identical interface; we keep both during verification so we can compare
the patched-ww result against scripts/spectral_analysis_siam.py.

Usage:
    PYTHONPATH=/home/mhough/dev/weightwatcher:/workspace/smri-fm/src \
      SIAM_MODEL_DIR=/siam_params/v0.3/pred_DS108_LcsfP_Ano \
      python scripts/weightwatcher_siam_v2.py [out_dir]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from asparagus_bridge.checkpoint import convert_checkpoint
from asparagus_bridge.models_smri_siam import SmriSiamClsRegBackbone


def main() -> int:
    if not os.environ.get("SIAM_MODEL_DIR"):
        print("ERROR: SIAM_MODEL_DIR is not set", file=sys.stderr)
        return 2

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/fomo26_fm_benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    siam_dir = Path(os.environ["SIAM_MODEL_DIR"])
    src_ckpt = siam_dir / "fold_0" / "checkpoint_final.pth"

    with tempfile.TemporaryDirectory() as tmp:
        dst_ckpt = Path(tmp) / "siam_asparagus.ckpt"
        print(f">>> converting {src_ckpt}")
        convert_checkpoint("smri_siam", src_ckpt, dst_ckpt)
        converted = torch.load(dst_ckpt, map_location="cpu", weights_only=False)
        wrapper_sd = {
            k[len("model."):]: v for k, v in converted["state_dict"].items()
            if k.startswith("model.")
        }
        print(">>> building SmriSiamClsRegBackbone(in=1,out=1)")
        model = SmriSiamClsRegBackbone(input_channels=1, output_channels=1)
        miss, unexp = model.load_state_dict(wrapper_sd, strict=False)
        print(f"    loaded {len([k for k in wrapper_sd if k.startswith('encoder.')])} encoder tensors")

    import weightwatcher as ww

    print(">>> ww.analyze(encoder)  [Conv3D-patched fork]")
    watcher = ww.WeightWatcher(model=model.encoder)
    details = watcher.analyze()
    summary = watcher.get_summary(details)

    details_path = out_dir / "ww_v2_siam_details.csv"
    summary_path = out_dir / "ww_v2_siam_summary.json"
    details.to_csv(details_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    print()
    print(f"layers analyzed: {len(details)}")
    if "alpha" in details.columns and len(details) > 0:
        print(f"alpha:           mean={details['alpha'].mean():.3f}  median={details['alpha'].median():.3f}")
        print(f"alpha_weighted:  mean={details['alpha_weighted'].mean():.3f}")
        if "stable_rank" in details.columns:
            print(f"stable_rank:     mean={details['stable_rank'].mean():.1f}")
        if "log_norm" in details.columns:
            print(f"log_norm:        mean={details['log_norm'].mean():.3f}")
        well_trained = (details["alpha"] < 2.0).mean()
        over_trained = (details["alpha"] < 1.0).mean()
        print(f"layers with alpha<2 (well-trained): {well_trained:.0%}")
        print(f"layers with alpha<1 (over-trained): {over_trained:.0%}")
    print()
    print(f"wrote {details_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
