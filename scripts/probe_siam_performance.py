"""Compute-side performance probe for the SIAM asparagus bridge.

Reports per-backbone:
  - parameter counts (total / encoder / decoder)
  - peak host & GPU memory during one forward pass
  - forward-pass wall time, averaged over a few warm runs

Runs on CPU; tries CUDA if torch.cuda.is_available(). For the GB10 (Blackwell
sm_100/sm_103) the CUDA path needs an nvcr.io/nvidia/pytorch image whose torch
was built with sm_100 support.

Usage (inside the NGC container or any env with torch + nnunetv2):
    SIAM_MODEL_DIR=~/siam_params/v0.3/pred_DS108_LcsfP_Ano \
      python scripts/probe_siam_performance.py
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
from pathlib import Path

import torch

from asparagus_bridge.models_smri_siam import (
    SmriSiamClsRegBackbone,
    SmriSiamSegBackbone,
)


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _time_forward(model: torch.nn.Module, x: torch.Tensor, repeats: int) -> tuple[float, float]:
    """Return (mean_s, std_s) of `repeats` forward passes after one warm-up."""
    model.eval()
    with torch.no_grad():
        _ = model(x)  # warmup
        if x.is_cuda:
            torch.cuda.synchronize()
        timings = []
        for _ in range(repeats):
            if x.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
    t = torch.tensor(timings)
    return float(t.mean()), float(t.std())


def _probe_one(name: str, model: torch.nn.Module, patch_size: tuple[int, int, int],
               device: torch.device, repeats: int) -> None:
    print(f"\n--- {name} on {device} @ patch_size={patch_size} ---")
    n_total = _param_count(model)
    n_enc = _param_count(model.encoder) if hasattr(model, "encoder") else None
    n_dec = _param_count(model.decoder) if hasattr(model, "decoder") else None
    print(f"    params: total={n_total/1e6:.1f}M"
          + (f"  encoder={n_enc/1e6:.1f}M" if n_enc is not None else "")
          + (f"  decoder={n_dec/1e6:.1f}M" if n_dec is not None else ""))

    try:
        model = model.to(device).eval()
        x = torch.randn(1, 1, *patch_size, device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        mean_s, std_s = _time_forward(model, x, repeats=repeats)
        rss_gb = _peak_rss_gb()
        std_str = f" ± {std_s*1000:.1f}" if repeats > 1 else ""
        print(f"    forward: {mean_s*1000:.1f}{std_str} ms  ({repeats} runs after warmup)")
        print(f"    peak host RSS: {rss_gb:.2f} GiB")
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"    peak GPU mem:  {peak_gb:.2f} GiB")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"    SKIPPED: {type(e).__name__}: {str(e)[:120]}")
    finally:
        # Free for the next probe.
        try:
            del model, x
        except NameError:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _siam_native_patch() -> tuple[int, int, int]:
    """Read SIAM's training patch size from plans.json (the only seg-safe shape)."""
    import json
    siam_dir = Path(os.environ["SIAM_MODEL_DIR"])
    plans = json.loads((siam_dir / "plans.json").read_text())
    config_name = os.environ.get("SIAM_CONFIG", "3d_fullres")
    ps = plans["configurations"][config_name]["patch_size"]
    return tuple(int(s) for s in ps)


def main() -> int:
    if not os.environ.get("SIAM_MODEL_DIR"):
        print("ERROR: SIAM_MODEL_DIR is not set", file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        print(f"    cuda device: {torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)}")
        devices.append(torch.device("cuda"))

    # Cls/reg only uses the encoder, so any 3D patch shape works; small_patch
    # gives a quick CPU baseline. Seg's decoder needs the SIAM-native patch
    # (or a downsample-aligned subset) for the skip-connection cats to line
    # up — use plans.json's patch_size.
    small_patch = (96, 96, 96)
    native_patch = _siam_native_patch()
    print(f"    SIAM native patch_size: {native_patch}")

    for device in devices:
        repeats = 5 if device.type == "cuda" else 2
        _probe_one(
            "SmriSiamClsRegBackbone(in=1,out=2) small",
            SmriSiamClsRegBackbone(input_channels=1, output_channels=2),
            small_patch, device, repeats,
        )
        # Native patch is heavy: 256x256x192 activations can OOM the GB10's
        # unified 128 GiB pool on CPU. Skip CPU native; let GPU try (and
        # _probe_one's try/except will report cleanly if it still OOMs).
        if device.type == "cuda":
            _probe_one(
                "SmriSiamClsRegBackbone(in=1,out=2) native",
                SmriSiamClsRegBackbone(input_channels=1, output_channels=2),
                native_patch, device, max(repeats // 2, 1),
            )
            _probe_one(
                "SmriSiamSegBackbone(in=1,out=5) native",
                SmriSiamSegBackbone(input_channels=1, output_channels=5),
                native_patch, device, max(repeats // 2, 1),
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
