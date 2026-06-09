"""Frozen-feature extraction for the cross-domain FM roster via the asparagus_bridge
wrappers (fomo60k / anatcl / triad). One loop, uniform `_features()` interface.

For each arm: convert the raw pretrained checkpoint to asparagus format, instantiate
the bridge wrapper (input/output_channels=1), load the converted encoder weights
(strip leading `model.`, strict=False — the fresh head is the only missing tensor),
freeze, then pool `_features(x)` per scan into the same long-format parquet the ridge
harness consumes (tool=<arm>_embed, region=embed_NNN, session parsed from pat_id).

Usage (inside NGC container, PYTHONPATH=third_party/asparagus:src, monai installed):
  python extract_bridge_embeddings.py --arm fomo60k \
    --checkpoint /data/.../combined_regular-step=200000.ckpt \
    --input_csv data/input_csv.csv --root_dir data/t3_flat \
    --out_dir results/fomo60k_combined_regular_embed --target 96
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BIDS_RE = re.compile(
    r"(?P<sub>sub-[A-Za-z0-9]+)(?:_(?P<ses>ses-[A-Za-z0-9]+))?"
    r"(?:_(?P<acq>acq-[A-Za-z0-9]+))?(?:_(?P<run>run-\d+))?"
)


def parse_entities(pat_id: str) -> dict:
    m = BIDS_RE.search(pat_id)
    return {k: (v or "") for k, v in m.groupdict().items()} if m else {
        "sub": pat_id, "ses": "", "acq": "", "run": ""}


def preprocess(t1_path: Path, target: int) -> np.ndarray:
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
    pb = np.maximum((target - shape) // 2, 0)
    pa = np.maximum(target - shape - pb, 0)
    arr = np.pad(arr, list(zip(pb, pa)), constant_values=0.0)
    if any(d > target for d in arr.shape):
        s = [(d - target) // 2 if d > target else 0 for d in arr.shape]
        arr = arr[s[0]:s[0]+target, s[1]:s[1]+target, s[2]:s[2]+target]
    return arr[np.newaxis, np.newaxis]  # (1,1,Z,Y,X)


def build_model(arm: str, checkpoint: Path, device: str):
    """Instantiate the bridge wrapper and load the converted pretrained encoder."""
    if arm == "fomo60k":
        from asparagus_bridge.models_smri_fomo60k import (
            SmriFomo60kClsRegBackbone as W, convert_fomo60k_checkpoint as conv)
    elif arm == "anatcl":
        from asparagus_bridge.models_smri_anatcl import (
            SmriAnatclClsRegBackbone as W, convert_anatcl_checkpoint as conv)
    elif arm == "triad":
        from asparagus_bridge.models_smri_triad import (
            SmriTriadClsRegBackbone as W, convert_triad_checkpoint as conv)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tf:
        dst = Path(tf.name)
    conv(checkpoint, dst)
    state = torch.load(dst, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    cleaned = {(k[len("model."):] if k.startswith("model.") else k): v
               for k, v in state.items()}
    model = W(input_channels=1, output_channels=1, dimensions="3D")
    res = model.load_state_dict(cleaned, strict=False)
    try:
        print(f"[{arm}] load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
              file=sys.stderr)
    except AttributeError:
        pass
    return model.eval().to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["fomo60k", "anatcl", "triad"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input_csv", type=Path, required=True)
    p.add_argument("--root_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--target", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.arm, args.checkpoint, device)
    df = pd.read_csv(args.input_csv, dtype={"pat_id": str})
    print(f"[{args.arm}] {len(df)} rows, device={device}, target={args.target}")

    ids, embs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = str(row["pat_id"])
        path = args.root_dir / f"{pid}.nii.gz"
        if not path.exists():
            print(f"miss: {path}", file=sys.stderr); continue
        try:
            vol = preprocess(path, args.target)
            with torch.no_grad():
                feat = model._features(torch.from_numpy(vol).to(device))
            embs.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
            ids.append(pid)
        except Exception as e:
            print(f"FAIL {pid}: {e}", file=sys.stderr)
    if not embs:
        sys.exit("no embeddings produced")
    E = np.stack(embs); print(f"[{args.arm}] embeddings: {E.shape}")

    tool = f"{args.arm}_embed"
    np.savez(args.out_dir / f"{args.arm}_embeddings.npz", **{i: e for i, e in zip(ids, embs)})
    rows = []
    for pid, e in zip(ids, embs):
        ent = parse_entities(pid)
        for j, v in enumerate(e):
            rows.append({"subject": ent["sub"], "session": ent["ses"], "acq": ent["acq"],
                         "run": ent["run"], "tool": tool, "region": f"embed_{j:03d}",
                         "value": float(v)})
    pd.DataFrame(rows).to_parquet(args.out_dir / f"{args.arm}_embeddings.parquet", index=False)
    print(f"wrote {args.out_dir}/{args.arm}_embeddings.parquet")


if __name__ == "__main__":
    main()
