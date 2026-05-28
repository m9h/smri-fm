"""Conv3D-aware spectral / power-law analysis of SIAM's encoder.

weightwatcher.WeightWatcher only recognizes Conv2D + Linear in its default
supported-layer set, so it returns an empty DataFrame on SIAM (pure Conv3D).
This script reimplements the metrics that matter for FM-quality comparison:
power-law tail exponent (α), spectral norm, Frobenius norm, stable rank.

For each Conv3D layer: reshape W from (out, in, kD, kH, kW) to (out, in*kD*kH*kW),
take singular values via torch.linalg.svdvals, derive squared values as the
empirical spectral distribution (ESD) eigenvalues, and fit a power law on the
tail using the `powerlaw` package (the same fitter weightwatcher uses).

Outputs:
    {out_dir}/spectral_siam_details.csv
    {out_dir}/spectral_siam_summary.json

Usage:
    SIAM_MODEL_DIR=~/siam_params/v0.3/pred_DS108_LcsfP_Ano \
      python scripts/spectral_analysis_siam.py [out_dir]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from asparagus_bridge.checkpoint import convert_checkpoint
from asparagus_bridge.models_smri_siam import SmriSiamClsRegBackbone


def _layer_metrics(name: str, W: torch.Tensor) -> dict:
    """Compute spectral metrics for a 2D weight matrix W (M x N)."""
    import powerlaw  # imported lazily so a missing pkg yields a clear error

    M, N = W.shape
    sigmas = torch.linalg.svdvals(W.float()).cpu().numpy()
    sigmas = sigmas[sigmas > 0]
    if sigmas.size < 5:
        return {
            "name": name,
            "M": M, "N": N,
            "min_dim": min(M, N),
            "spectral_norm": float(sigmas.max()) if sigmas.size else float("nan"),
            "frobenius_norm": float(np.linalg.norm(sigmas)) if sigmas.size else float("nan"),
            "log_norm": float(np.log10(np.linalg.norm(sigmas))) if sigmas.size else float("nan"),
            "stable_rank": float((sigmas**2).sum() / sigmas.max()**2) if sigmas.size else float("nan"),
            "alpha": float("nan"),
            "xmin": float("nan"),
            "n_tail": 0,
        }

    eigvals = sigmas**2  # ESD eigenvalues = σ²
    eigvals_desc = np.sort(eigvals)[::-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(eigvals_desc, verbose=False, xmin_distance="D")
        alpha = float(fit.alpha)
        xmin = float(fit.xmin)
        n_tail = int((eigvals_desc >= xmin).sum())

    return {
        "name": name,
        "M": M, "N": N,
        "min_dim": min(M, N),
        "spectral_norm": float(sigmas.max()),
        "frobenius_norm": float(np.linalg.norm(sigmas)),
        "log_norm": float(np.log10(np.linalg.norm(sigmas))),
        "stable_rank": float((sigmas**2).sum() / sigmas.max()**2),
        "alpha": alpha,
        "xmin": xmin,
        "n_tail": n_tail,
    }


def _conv3d_weight_as_2d(W: torch.Tensor) -> torch.Tensor:
    """Reshape Conv3D weight (out, in, kD, kH, kW) to (out, in*kD*kH*kW)."""
    out_ch = W.shape[0]
    return W.reshape(out_ch, -1)


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
        print(">>> building SmriSiamClsRegBackbone(in=1,out=1) and loading SIAM weights")
        model = SmriSiamClsRegBackbone(input_channels=1, output_channels=1)
        miss, unexp = model.load_state_dict(wrapper_sd, strict=False)
        print(f"    loaded {len([k for k in wrapper_sd if k.startswith('encoder.')])} encoder tensors  missing={len(list(miss))}")

    rows = []
    print(">>> analyzing encoder Conv3D layers")
    for name, module in model.encoder.named_modules():
        if isinstance(module, nn.Conv3d) and module.weight is not None:
            W2d = _conv3d_weight_as_2d(module.weight.data)
            if W2d.shape[1] < 2:
                continue  # 1x1x1 conv with single input channel — too small to spectrally analyze
            rows.append(_layer_metrics(f"encoder.{name}", W2d))

    df = pd.DataFrame(rows)

    details_path = out_dir / "spectral_siam_details.csv"
    summary_path = out_dir / "spectral_siam_summary.json"
    df.to_csv(details_path, index=False)

    valid_alpha = df["alpha"].dropna()
    summary = {
        "n_conv3d_layers": int(len(df)),
        "n_with_alpha": int(valid_alpha.size),
        "alpha_mean": float(valid_alpha.mean()) if valid_alpha.size else None,
        "alpha_median": float(valid_alpha.median()) if valid_alpha.size else None,
        "alpha_min": float(valid_alpha.min()) if valid_alpha.size else None,
        "alpha_max": float(valid_alpha.max()) if valid_alpha.size else None,
        "frac_alpha_lt_2": float((valid_alpha < 2.0).mean()) if valid_alpha.size else None,
        "frac_alpha_lt_3": float((valid_alpha < 3.0).mean()) if valid_alpha.size else None,
        "frac_alpha_lt_6": float((valid_alpha < 6.0).mean()) if valid_alpha.size else None,
        "log_norm_mean": float(df["log_norm"].mean()),
        "stable_rank_mean": float(df["stable_rank"].mean()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    print()
    print(f"layers analyzed: {summary['n_conv3d_layers']}  (alpha fit on {summary['n_with_alpha']})")
    print(f"alpha    mean={summary['alpha_mean']:.2f}  median={summary['alpha_median']:.2f}  "
          f"min={summary['alpha_min']:.2f}  max={summary['alpha_max']:.2f}")
    print(f"frac alpha<2 (well-trained):    {summary['frac_alpha_lt_2']:.0%}")
    print(f"frac alpha<3 (reasonable):      {summary['frac_alpha_lt_3']:.0%}")
    print(f"frac alpha<6 (any tail signal): {summary['frac_alpha_lt_6']:.0%}")
    print(f"log_norm mean: {summary['log_norm_mean']:.3f}")
    print(f"stable_rank mean: {summary['stable_rank_mean']:.1f}")
    print()
    print(f"wrote {details_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
