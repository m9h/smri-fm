"""Extract FOMO25 mmunetvae (MultiModalUNetVAE) embeddings.

Counterpart to extract_fomo25_embeddings.py (which targets AMAES_resenc_b).
Uses the publicly-released checkpoint at github.com/jbanusco/fomo25 master
@ weights/fomo25_mmunetvae_pretrained.ckpt (217 MB).

Difference from AMAES variant:
  - Architecture: MultiModalUNetVAE (single-modality pretraining), not ResEnc-UNet
  - Patch size: 64³ (per the README), not 96³
  - Embedding: pool the encoder bottleneck (use_vae=True, get the mu).

Runs inside fomo25-arm container with the jbanusco/fomo25 src/ mounted at
/opt/fomo25_src/src/.
"""
from __future__ import annotations

import argparse
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


def preprocess(t1_path: Path, target: int = 64) -> np.ndarray:
    """RAS → z-norm in brain → bbox crop → pad/center-crop to target^3.

    mmunetvae was pretrained at 64³ patch size (per the official README).
    """
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
    """Load MultiModalUNetVAE from the public fomo25 v1.0.0 checkpoint."""
    sys.path.insert(0, "/opt/fomo25_src/src")
    from models.networks.mmunetvae import MultiModalUNetVAE

    print(f"Loading mmunetvae checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k.split("model.", 1)[1]] = v
        elif k.startswith("network."):
            cleaned[k.split("network.", 1)[1]] = v
        else:
            cleaned[k] = v

    # Per the README + ckpt size-mismatch errors: pretrained with
    # starting_filters=32 (default is 64). Bottleneck = 32 × 16 = 512;
    # conv_mu_shared input channels = sum(32 × 2^i for i in 0..4) = 992
    # which matches the checkpoint exactly.
    model = MultiModalUNetVAE(
        mode="mae",
        input_channels=1,
        output_channels=1,
        starting_filters=32,
        use_vae=True,
        use_skip_connections=False,
    )
    result = model.load_state_dict(cleaned, strict=False)
    if result is not None:
        try:
            print(f"load_state_dict: missing={len(result.missing_keys)} "
                  f"unexpected={len(result.unexpected_keys)}", file=sys.stderr)
        except AttributeError:
            pass
    return model.eval().to(device)


def embed(model, vol: np.ndarray, device: str) -> np.ndarray:
    """Forward + pool the encoder bottleneck.

    The encoder produces multi-scale feature maps; pool the deepest one
    (the bottleneck) to a single embedding vector via global average pool.
    """
    with torch.no_grad():
        x = torch.from_numpy(vol).to(device)
        # encoder.forward returns a list of feature maps (shallow → deep)
        feats = model.encoder(x)
        bottleneck = feats[-1] if isinstance(feats, (list, tuple)) else feats
        # Spatial pool dimensions 2..end
        emb = bottleneck.mean(dim=tuple(range(2, bottleneck.ndim))).squeeze(0)
        if emb.ndim == 0:
            emb = emb.unsqueeze(0)
    return emb.cpu().numpy().astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=Path, required=True,
                   help="CSV with pat_id column (one row per scan)")
    p.add_argument("--root_dir", type=Path, required=True,
                   help="Flat dir of {pat_id}.nii.gz")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("/opt/fomo25_src/weights/fomo25_mmunetvae_pretrained.ckpt"),
                   help='mmunetvae .ckpt (default: official v1.0.0 release in fomo25_src)')
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--target", type=int, default=64, help="cube side for pad/crop (mmunetvae=64)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv, dtype={"pat_id": str})
    print(f"input rows: {len(df)}, device: {device}")

    model = load_mmunetvae(args.checkpoint, device)

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
        emb = embed(model, vol, device)
        ids.append(pid)
        embs.append(emb)

    if not embs:
        sys.exit("no embeddings produced")
    E = np.stack(embs)
    print(f"embeddings: {E.shape}")

    np.savez(args.out_dir / "fomo25_mmunetvae_embeddings.npz",
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
                "tool": "fomo25_mmunetvae_embed",
                "region": f"embed_{j:03d}",
                "value": float(v),
            })
    pd.DataFrame(rows).to_parquet(
        args.out_dir / "fomo25_mmunetvae_embeddings.parquet", index=False
    )
    print(f"wrote {args.out_dir}/{{fomo25_mmunetvae_embeddings.npz, fomo25_mmunetvae_embeddings.parquet}}")


if __name__ == "__main__":
    main()
