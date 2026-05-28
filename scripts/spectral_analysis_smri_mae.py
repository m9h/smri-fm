"""Spectral / power-law analysis of MedARC's smri_mae MaskedViT encoder.

Companion cross-check to scripts/weightwatcher_smri_mae_v2.py — does the same
per-layer SVD + power-law fit that scripts/spectral_analysis_siam.py performs,
so smri_mae and SIAM can be compared on the same custom-analyzer axis (in
addition to the patched-ww axis).

MaskedViT is pure nn.Linear (Patchify3D is a reshape), so every encoder weight
tensor is already 2D and no Conv3D unfolding is needed. We still mirror the
SIAM script's metric set (alpha, log_norm, spectral_norm, stable_rank,
n_tail) for column parity.

Outputs:
    {out_dir}/spectral_smri_mae_details.csv
    {out_dir}/spectral_smri_mae_summary.json

Usage:
    [SMRI_MAE_CKPT=/path/to/asparagus_compatible_weights.pth] \
      python scripts/spectral_analysis_smri_mae.py [out_dir]
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from asparagus_bridge.models_smri_mae import SmriMaeClsRegBackbone


IMG_SIZE = (208, 240, 208)
PATCH_SIZE = 16
MIN_DIM_FOR_FIT = 5


def _layer_metrics(name: str, W: torch.Tensor) -> dict:
    import powerlaw

    if W.ndim != 2:
        W = W.reshape(W.shape[0], -1)
    M, N = W.shape
    sigmas = torch.linalg.svdvals(W.float()).cpu().numpy()
    sigmas = sigmas[sigmas > 0]
    if sigmas.size < MIN_DIM_FOR_FIT:
        return {
            "name": name, "M": M, "N": N, "min_dim": min(M, N),
            "spectral_norm": float(sigmas.max()) if sigmas.size else float("nan"),
            "frobenius_norm": float(np.linalg.norm(sigmas)) if sigmas.size else float("nan"),
            "log_norm": float(np.log10(np.linalg.norm(sigmas))) if sigmas.size else float("nan"),
            "stable_rank": float((sigmas**2).sum() / sigmas.max()**2) if sigmas.size else float("nan"),
            "alpha": float("nan"), "xmin": float("nan"), "n_tail": 0,
        }

    eigvals = sigmas**2
    eigvals_desc = np.sort(eigvals)[::-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(eigvals_desc, verbose=False, xmin_distance="D")
        alpha = float(fit.alpha)
        xmin = float(fit.xmin)
        n_tail = int((eigvals_desc >= xmin).sum())

    return {
        "name": name, "M": M, "N": N, "min_dim": min(M, N),
        "spectral_norm": float(sigmas.max()),
        "frobenius_norm": float(np.linalg.norm(sigmas)),
        "log_norm": float(np.log10(np.linalg.norm(sigmas))),
        "stable_rank": float((sigmas**2).sum() / sigmas.max()**2),
        "alpha": alpha, "xmin": xmin, "n_tail": n_tail,
    }


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
        input_channels=1, output_channels=1,
        img_size=IMG_SIZE, patch_size=PATCH_SIZE,
    )
    print(f"    ckpt: {_maybe_load_ckpt(model)}")

    rows = []
    print(">>> analyzing encoder Linear layers")
    for name, module in model.encoder.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            W = module.weight.data
            if min(W.shape) < 2:
                continue
            rows.append(_layer_metrics(f"encoder.{name}", W))

    df = pd.DataFrame(rows)

    details_path = out_dir / "spectral_smri_mae_details.csv"
    summary_path = out_dir / "spectral_smri_mae_summary.json"
    df.to_csv(details_path, index=False)

    valid_alpha = df["alpha"].dropna()
    summary = {
        "n_linear_layers": int(len(df)),
        "n_with_alpha": int(valid_alpha.size),
        "alpha_mean": float(valid_alpha.mean()) if valid_alpha.size else None,
        "alpha_median": float(valid_alpha.median()) if valid_alpha.size else None,
        "alpha_min": float(valid_alpha.min()) if valid_alpha.size else None,
        "alpha_max": float(valid_alpha.max()) if valid_alpha.size else None,
        "frac_alpha_lt_2": float((valid_alpha < 2.0).mean()) if valid_alpha.size else None,
        "frac_alpha_lt_3": float((valid_alpha < 3.0).mean()) if valid_alpha.size else None,
        "frac_alpha_lt_6": float((valid_alpha < 6.0).mean()) if valid_alpha.size else None,
        "log_norm_mean": float(df["log_norm"].mean()) if len(df) else None,
        "stable_rank_mean": float(df["stable_rank"].mean()) if len(df) else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    print()
    print(f"layers analyzed: {summary['n_linear_layers']}  (alpha fit on {summary['n_with_alpha']})")
    if summary["n_with_alpha"]:
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
