"""Empirical verification of the three Item 4 hypotheses.

Runs inside ghcr.io/m9h/fomo25-arm:latest with asparagus + gardening_tools
+ NGC torch + the bundled AMAES_resenc_b checkpoint.

Hypotheses (from notes/fomo25_mae_recon_collapse_item4.md):
  H1. With `rec_loss_masked_only=False`, the U-Net's skip connections
      pass un-masked voxels through near-losslessly. So
      MSE(pred, y) on the visible 40% << MSE on the masked 60%.
  H2. The loss collapses to ~0 quickly during training because the
      visible-region term dominates and is trivial to minimize.
  H3. Even when `rec_loss_masked_only=True`, MSELoss(reduction='mean')
      divides by total voxels rather than mask.sum(), so the reported
      loss is scaled by mask_ratio (~0.6×).

Outputs:
  /out/item4_results.json      — measurements
  /out/item4_recon_figure.png  — visualization (if matplotlib available)
  /out/item4_training_curve.png — T3 loss trajectory
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_checkpoint() -> Path:
    """Resolve AMAES_resenc_b ckpt path (baked into fomo25-arm via HF cache)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id="FOMO-MRI/AMAES_resenc_b",
        filename="resenc_unet_b.ckpt",
        cache_dir="/opt/checkpoints",
    ))


def load_amaes(device: str) -> nn.Module:
    from asparagus.modules.networks.resenc_unet import resenc_unet_b
    ckpt_path = find_checkpoint()
    print(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k.split("model.", 1)[1]] = v
        elif k.startswith("network."):
            cleaned[k.split("network.", 1)[1]] = v
        else:
            cleaned[k] = v
    model = resenc_unet_b(dimensions="3D", input_channels=1, output_channels=1)
    model.load_state_dict(cleaned, strict=False)
    return model.eval().to(device)


def random_init_amaes(device: str) -> nn.Module:
    from asparagus.modules.networks.resenc_unet import resenc_unet_b
    model = resenc_unet_b(dimensions="3D", input_channels=1, output_channels=1)
    return model.to(device)


def preprocess(t1_path: Path, target: int = 96) -> np.ndarray:
    """RAS reorient → z-norm in brain → bbox crop → pad to target^3."""
    img = nib.as_closest_canonical(nib.load(t1_path))
    arr = np.asanyarray(img.dataobj, dtype=np.float32)
    mask = arr > 0
    if mask.any():
        mu, sd = arr[mask].mean(), arr[mask].std()
        arr = (arr - mu) / (sd + 1e-6)
        arr[~mask] = 0.0
        ijk = np.argwhere(mask)
        lo, hi = ijk.min(0), ijk.max(0) + 1
        arr = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    shape = np.array(arr.shape)
    pad_before = np.maximum((target - shape) // 2, 0)
    pad_after = np.maximum(target - shape - pad_before, 0)
    arr = np.pad(arr, list(zip(pad_before, pad_after)), constant_values=0.0)
    if any(d > target for d in arr.shape):
        s = [(d - target) // 2 if d > target else 0 for d in arr.shape]
        arr = arr[s[0]:s[0]+target, s[1]:s[1]+target, s[2]:s[2]+target]
    return arr  # (Z, Y, X) — caller adds batch + channel


def make_token_mask(shape, ratio: float = 0.6, token: int = 4, seed: int = 0) -> torch.Tensor:
    """Match gardening_tools Torch_Mask: token-cube random masking."""
    g = torch.Generator().manual_seed(seed)
    Z, Y, X = shape
    z_tok, y_tok, x_tok = Z // token, Y // token, X // token
    n_tokens = z_tok * y_tok * x_tok
    n_mask = int(n_tokens * ratio)
    flat = torch.zeros(n_tokens, dtype=torch.bool)
    perm = torch.randperm(n_tokens, generator=g)
    flat[perm[:n_mask]] = True
    coarse = flat.view(z_tok, y_tok, x_tok)
    mask = (
        coarse.repeat_interleave(token, 0)
              .repeat_interleave(token, 1)
              .repeat_interleave(token, 2)
    )
    # Pad if dims don't divide evenly
    if mask.shape != tuple(shape):
        m = torch.zeros(shape, dtype=torch.bool)
        s0, s1, s2 = mask.shape
        m[:s0, :s1, :s2] = mask
        mask = m
    return mask


def buggy_rec_loss(pred, y, mask):
    """Replicates _rec_loss in asparagus self_supervised.py masked-only branch."""
    y_masked = y.clone()
    pred_masked = pred.clone()
    y_masked[~mask] = 0
    pred_masked[~mask] = 0
    return nn.MSELoss(reduction="mean")(pred_masked, y_masked)


def correct_rec_loss(pred, y, mask):
    return ((pred - y)[mask] ** 2).mean()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_2_scaling_bug() -> dict:
    """T2: pure-pytorch demonstration that buggy = correct × mask_ratio."""
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 96, 96, 96)
    y = torch.randn(2, 1, 96, 96, 96)
    rng = torch.Generator().manual_seed(0)
    mask = torch.rand(2, 1, 96, 96, 96, generator=rng) < 0.6

    buggy = buggy_rec_loss(pred, y, mask).item()
    correct = correct_rec_loss(pred, y, mask).item()
    mask_ratio = mask.float().mean().item()

    return {
        "buggy_loss": buggy,
        "correct_masked_only_loss": correct,
        "mask_ratio": mask_ratio,
        "buggy_div_correct": buggy / correct,
        "predicted_buggy_eq_correct_times_mask_ratio": correct * mask_ratio,
        "delta_buggy_vs_predicted": abs(buggy - correct * mask_ratio),
        "hypothesis_confirmed": abs(buggy / correct - mask_ratio) < 0.005,
    }


def test_1_skip_copy(model, device, scans, n_scans=5) -> dict:
    """T1: visible-region MSE ≪ masked-region MSE if skip-copy hypothesis holds."""
    rows = []
    for i, p in enumerate(scans[:n_scans]):
        try:
            arr = preprocess(p)
        except Exception as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
            continue
        x_full = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,Z,Y,X)
        mask = make_token_mask(x_full.shape[2:], ratio=0.6, seed=i).to(device)
        x_masked = x_full.clone()
        x_masked[0, 0][mask] = 0.0  # zero out the masked tokens (matches Torch_Mask)
        with torch.no_grad():
            if hasattr(model, "forward_with_features"):
                pred, _ = model.forward_with_features(x_masked)
            else:
                pred = model(x_masked)
        # Squeeze to spatial-only for masking
        pred_3d = pred[0, 0]
        y_3d = x_full[0, 0]
        mse_visible = ((pred_3d - y_3d)[~mask] ** 2).mean().item()
        mse_masked = ((pred_3d - y_3d)[mask] ** 2).mean().item()
        mse_total = ((pred_3d - y_3d) ** 2).mean().item()
        rows.append({
            "scan": p.name,
            "mse_visible": mse_visible,
            "mse_masked": mse_masked,
            "mse_total_buggy": mse_total,
            "ratio_visible_over_masked": mse_visible / max(mse_masked, 1e-12),
        })
        print(f"  {p.name}  visible={mse_visible:.4f}  masked={mse_masked:.4f}  "
              f"ratio={mse_visible/max(mse_masked,1e-12):.3f}")
    return {
        "per_scan": rows,
        "mean_mse_visible": float(np.mean([r["mse_visible"] for r in rows])),
        "mean_mse_masked": float(np.mean([r["mse_masked"] for r in rows])),
        "mean_ratio_visible_over_masked": float(np.mean([r["ratio_visible_over_masked"] for r in rows])),
        "hypothesis_confirmed": float(np.mean([r["ratio_visible_over_masked"] for r in rows])) < 0.5,
    }


def test_3_tiny_training(device, scans, n_scans=5, n_steps=200) -> dict:
    """T3: run the buggy loss for ~200 SSL steps; instrument loss separately."""
    print(f"  T3: {n_steps} steps × {n_scans} scans × 1 mask resample/step")
    # Use a fresh random-init model so we see the collapse trajectory.
    model = random_init_amaes(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-5)

    # Pre-load + preprocess a small batch
    vols = []
    for p in scans[:n_scans]:
        try:
            arr = preprocess(p)
            vols.append(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0))
        except Exception as e:
            print(f"  skip {p.name}: {e}", file=sys.stderr)
    if not vols:
        return {"error": "no scans loaded"}
    batch = torch.cat(vols, dim=0).to(device)  # (B, 1, Z, Y, X)
    print(f"  batch shape {tuple(batch.shape)}")

    history = []
    for step in range(n_steps):
        seed = step
        masks = []
        for b in range(batch.shape[0]):
            m = make_token_mask(batch.shape[2:], ratio=0.6, seed=seed * 1000 + b)
            masks.append(m)
        mask_b = torch.stack(masks).unsqueeze(1).to(device)  # (B, 1, Z, Y, X)

        x_masked = batch.clone()
        x_masked[mask_b] = 0.0

        if hasattr(model, "forward_with_features"):
            pred, _ = model.forward_with_features(x_masked)
        else:
            pred = model(x_masked)
        loss_buggy = buggy_rec_loss(pred, batch, mask_b)

        opt.zero_grad()
        loss_buggy.backward()
        opt.step()

        with torch.no_grad():
            mse_v = ((pred - batch)[~mask_b] ** 2).mean().item()
            mse_m = ((pred - batch)[mask_b] ** 2).mean().item()
            mse_t = ((pred - batch) ** 2).mean().item()
        history.append({
            "step": step,
            "loss_buggy": float(loss_buggy.detach().item()),
            "mse_visible": mse_v,
            "mse_masked": mse_m,
            "mse_total": mse_t,
        })
        if step in (0, 9, 19, 49, 99, n_steps - 1):
            print(f"    step {step:>4}  buggy={history[-1]['loss_buggy']:.4f}  "
                  f"visible={mse_v:.4f}  masked={mse_m:.4f}  total={mse_t:.4f}")
    return {
        "history": history,
        "step_50_visible_mse": history[50]["mse_visible"] if len(history) > 50 else None,
        "step_50_masked_mse": history[50]["mse_masked"] if len(history) > 50 else None,
        "final_visible_mse": history[-1]["mse_visible"],
        "final_masked_mse": history[-1]["mse_masked"],
        "hypothesis_confirmed": (
            history[-1]["mse_visible"] < 0.5 * history[-1]["mse_masked"]
        ),
    }


def make_figure(model, device, scans, out_dir):
    """T4: visualize one example: target, masked input, prediction, error."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping T4 figure")
        return None

    if not scans:
        return None
    arr = preprocess(scans[0])
    x_full = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    mask = make_token_mask(x_full.shape[2:], ratio=0.6, seed=0).to(device)
    x_masked = x_full.clone()
    x_masked[0, 0][mask] = 0.0
    with torch.no_grad():
        if hasattr(model, "forward_with_features"):
            pred, _ = model.forward_with_features(x_masked)
        else:
            pred = model(x_masked)
    target = x_full[0, 0].cpu().numpy()
    inp = x_masked[0, 0].cpu().numpy()
    pre = pred[0, 0].cpu().numpy()
    err = (pre - target)
    z = target.shape[0] // 2

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, im, title in zip(axes,
                              [target, inp, pre, err],
                              ["target", "masked input", "prediction", "error (pred−target)"]):
        ax.imshow(im[z], cmap="gray" if "error" not in title else "RdBu_r",
                  vmin=-2, vmax=2)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Item 4 / T4 — AMAES_resenc_b reconstruction (axial slice z=mid)")
    fig.tight_layout()
    p = out_dir / "item4_recon_figure.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return str(p)


def make_curve_figure(t3, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    h = t3.get("history", [])
    if not h:
        return None
    steps = [r["step"] for r in h]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [r["mse_visible"] for r in h], label="MSE visible (~mask)", color="tab:blue")
    ax.plot(steps, [r["mse_masked"] for r in h], label="MSE masked", color="tab:red")
    ax.plot(steps, [r["loss_buggy"] for r in h], label="reported buggy loss", color="black", ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE")
    ax.set_title("Item 4 / T3 — buggy loss collapses, masked loss stays meaningful")
    ax.legend()
    ax.grid(alpha=0.3)
    p = out_dir / "item4_training_curve.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return str(p)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}; torch: {torch.__version__}")
    out_dir = Path("/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover input scans (mounted at /scans, BrainIAC-preprocessed)
    scans = sorted(Path("/scans").glob("*.nii.gz"))[:10]
    print(f"found {len(scans)} input scans")

    print("\n=== T2: MSELoss scaling bug ===")
    t2 = test_2_scaling_bug()
    for k, v in t2.items():
        print(f"  {k}: {v}")

    print("\n=== T1: skip-copy with checkpoint ===")
    model = load_amaes(device)
    t1 = test_1_skip_copy(model, device, scans, n_scans=5)

    print("\n=== T4: reconstruction figure ===")
    fig_recon = make_figure(model, device, scans, out_dir)

    print("\n=== T3: tiny SSL training loop ===")
    # Free the eval model before loading a fresh one for training
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    t3 = test_3_tiny_training(device, scans, n_scans=4, n_steps=200)
    fig_curve = make_curve_figure(t3, out_dir)

    results = {
        "T1_skip_copy": t1,
        "T2_scaling_bug": t2,
        "T3_tiny_training": t3,
        "figures": {"recon": fig_recon, "training_curve": fig_curve},
        "verdict": {
            "H1_skip_copy_visible_mse_lt_masked": t1.get("hypothesis_confirmed"),
            "H2_buggy_loss_collapses_during_training": t3.get("hypothesis_confirmed"),
            "H3_mse_loss_scaled_by_mask_ratio": t2.get("hypothesis_confirmed"),
        },
    }
    out = out_dir / "item4_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps(results["verdict"], indent=2))


if __name__ == "__main__":
    main()
