"""Compute-side performance probe for the smri_mae asparagus bridge.

Companion to scripts/probe_siam_performance.py — same metrics (param counts,
forward-pass wall time, peak host/GPU memory) so the two FMs are directly
comparable on the same hardware.

Patch sizes:
  - small  = (160, 160, 160)              — the bridge's default img_size
  - native = (208, 240, 208)              — the asparagus smoke-test target_size
Both must be divisible by patch_size (default 16); 160/208/240 all are.

Optional pretrained weights: set SMRI_MAE_CKPT to a path or HF repo-id; the
script will load_state_dict(strict=False) for the perf timing (alphas /
spectral metrics don't move much with init, but loading lets us also smoke-test
that the checkpoint plumbing works).

Usage (inside the NGC container or any env with torch + smri_mae deps):
    python scripts/probe_smri_mae_performance.py
    SMRI_MAE_CKPT=/path/to/asparagus_compatible_weights.pth \
      python scripts/probe_smri_mae_performance.py
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
from pathlib import Path

import torch

from asparagus_bridge.models_smri_mae import SmriMaeClsRegBackbone


SMALL_PATCH = (160, 160, 160)
NATIVE_PATCH = (208, 240, 208)
PATCH_SIZE = 16


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _time_forward(model: torch.nn.Module, x: torch.Tensor, repeats: int) -> tuple[float, float]:
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


def _maybe_load_ckpt(model: torch.nn.Module) -> str:
    """If SMRI_MAE_CKPT is set, load weights into model. Returns a status string."""
    ckpt_spec = os.environ.get("SMRI_MAE_CKPT")
    if not ckpt_spec:
        return "random init (SMRI_MAE_CKPT unset)"
    path = Path(ckpt_spec)
    if not path.exists():
        return f"SKIPPED: {ckpt_spec} not found on disk"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    raw = blob.get("state_dict") or blob.get("model") or blob
    # accept both `model.encoder.*` (asparagus-converted) and bare `encoder.*`
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in raw.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    return f"loaded {len(sd)} tensors  missing={len(list(miss))}  unexpected={len(list(unexp))}"


def _probe_one(name: str, model: torch.nn.Module, patch_size: tuple[int, int, int],
               device: torch.device, repeats: int) -> None:
    print(f"\n--- {name} on {device} @ patch_size={patch_size} ---")
    n_total = _param_count(model)
    n_enc = _param_count(model.encoder) if hasattr(model, "encoder") else None
    n_head = _param_count(model.head) if hasattr(model, "head") else None
    print(f"    params: total={n_total/1e6:.1f}M"
          + (f"  encoder={n_enc/1e6:.1f}M" if n_enc is not None else "")
          + (f"  head={n_head/1e3:.1f}K" if n_head is not None else ""))

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
        print(f"    SKIPPED: {type(e).__name__}: {str(e)[:160]}")
    finally:
        try:
            del model, x
        except NameError:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _build(patch_size: tuple[int, int, int]) -> SmriMaeClsRegBackbone:
    model = SmriMaeClsRegBackbone(
        input_channels=1,
        output_channels=2,
        img_size=patch_size,
        patch_size=PATCH_SIZE,
    )
    status = _maybe_load_ckpt(model)
    print(f"    ckpt: {status}")
    return model


def main() -> int:
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        print(f"    cuda device: {torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)}")
        devices.append(torch.device("cuda"))

    for device in devices:
        repeats = 5 if device.type == "cuda" else 2
        _probe_one(
            f"SmriMaeClsRegBackbone(in=1,out=2) small={SMALL_PATCH}",
            _build(SMALL_PATCH),
            SMALL_PATCH, device, repeats,
        )
        if device.type == "cuda":
            _probe_one(
                f"SmriMaeClsRegBackbone(in=1,out=2) native={NATIVE_PATCH}",
                _build(NATIVE_PATCH),
                NATIVE_PATCH, device, max(repeats // 2, 1),
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
