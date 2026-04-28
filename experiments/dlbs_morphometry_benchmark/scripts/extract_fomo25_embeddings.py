"""Extract FOMO25 baseline-SSL (mmunetvae) embeddings for DLBS T1s.

Mirrors `extract_brainiac_embeddings.py` so the downstream ridge call
is a one-line tool-name swap. CSV-driven over a flat directory of
already-MNI-1mm-iso skull-stripped T1s (the BrainIAC preprocessing
output works directly).

Runs **inside** `jbanusco/sslmmunetave:1.0.0` on Legion (image is
amd64 only). The driver script that invokes docker is alongside this
file at `run_fomo25_extraction_legion.sh`.

Inputs (mounts inside container):
  /csv/brainiac_dlbs_csv.csv          — pat_id, label, dataset
  /scans/<pat_id>.nii.gz              — flat dir of preprocessed T1s
  /opt/ssl3d/checkpoints/...          — mmunetvae checkpoint (image)

Outputs (written to mounted /out):
  fomo25_embeddings.npz               — wide, key=pat_id, value=(D,) f32
  fomo25_embeddings.parquet           — long, schema matches brainiac

Long-form schema (joinable with `fit_ridge_baseline.py --tool fomo25_embed`):
  subject | session | acq | run | tool=fomo25_embed | region=embed_NNN | value
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


BIDS_RE = re.compile(
    r"(?P<sub>sub-[A-Za-z0-9]+)"
    r"(?:_(?P<ses>ses-[A-Za-z0-9]+))?"
    r"(?:_(?P<acq>acq-[A-Za-z0-9]+))?"
    r"(?:_(?P<run>run-\d+))?"
)


def parse_entities(pat_id: str) -> dict[str, str]:
    m = BIDS_RE.search(pat_id)
    if not m:
        return {"sub": pat_id, "ses": "", "acq": "", "run": ""}
    return {k: (v or "") for k, v in m.groupdict().items()}


def preprocess(t1_path: Path, target: int = 96) -> np.ndarray:
    """RAS reorient → z-norm in brain → bbox crop → pad/center-crop to target^3."""
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
    return arr[np.newaxis, np.newaxis]  # (1, 1, Z, Y, X)


def load_mmunetvae(checkpoint: Path, device: str):
    """Locate the FOMO25 mmunetvae module + load its checkpoint."""
    candidates = [
        "models.networks.mmunetvae",
        "src.models.networks.mmunetvae",
        "fomo25.models.networks.mmunetvae",
    ]
    module = None
    for name in candidates:
        try:
            module = importlib.import_module(name)
            break
        except ImportError:
            continue
    if module is None:
        raise RuntimeError(
            "mmunetvae module not on PYTHONPATH. Tried:\n  "
            + "\n  ".join(candidates)
        )
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.split("model.", 1)[-1]: v for k, v in state.items()}
    model = module.MultiModalUNetVAE()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )
    return model.eval().to(device)


def embed(model, vol: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(vol).to(device)
        if hasattr(model, "encode"):
            z = model.encode(x)
        elif hasattr(model, "forward_encoder"):
            z = model.forward_encoder(x)
        else:
            z = model(x)
        if isinstance(z, (list, tuple)):
            z = z[0]
        emb = z.mean(dim=tuple(range(2, z.ndim))).squeeze(0).cpu().numpy()
    return emb.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=Path, required=True,
                   help="CSV with pat_id column (one row per scan)")
    p.add_argument("--root_dir", type=Path, required=True,
                   help="Flat dir of {pat_id}.nii.gz (e.g. BrainIAC-preproc)")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("/opt/ssl3d/checkpoints/fomo25_mmunetvae_pretrained.ckpt"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--target", type=int, default=96, help="cube side for pad/crop")
    p.add_argument("--dry-run", action="store_true",
                   help="preprocess only, write zero-vector embeddings")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, dtype={"pat_id": str})
    print(f"input rows: {len(df)}, device: {device}")

    model = None if args.dry_run else load_mmunetvae(args.checkpoint, device)

    ids: list[str] = []
    embs: list[np.ndarray] = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = str(row["pat_id"])
        path = args.root_dir / f"{pid}.nii.gz"
        if not path.exists():
            print(f"miss: {path}", file=sys.stderr)
            continue
        try:
            vol = preprocess(path, target=args.target)
        except Exception as e:
            print(f"FAIL preprocess {pid}: {e}", file=sys.stderr)
            continue
        if args.dry_run:
            emb = np.zeros(1, dtype=np.float32)
        else:
            emb = embed(model, vol, device)
        ids.append(pid)
        embs.append(emb)

    if not embs:
        sys.exit("no embeddings produced")
    E = np.stack(embs)
    print(f"embeddings: {E.shape}")

    np.savez(args.out_dir / "fomo25_embeddings.npz",
             **{pid: e for pid, e in zip(ids, embs)})

    rows = []
    for pid, e in zip(ids, embs):
        ents = parse_entities(pid)
        for j, v in enumerate(e):
            rows.append({
                "subject": ents["sub"],
                "session": ents["ses"],
                "acq": ents["acq"],
                "run": ents["run"],
                "tool": "fomo25_embed",
                "region": f"embed_{j:03d}",
                "value": float(v),
            })
    pd.DataFrame(rows).to_parquet(args.out_dir / "fomo25_embeddings.parquet",
                                  index=False)
    print(f"wrote {args.out_dir}/{{fomo25_embeddings.npz, fomo25_embeddings.parquet}}")


if __name__ == "__main__":
    main()
