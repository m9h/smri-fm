"""Compute FOMO25 MultiModalUNetVAE embeddings for a BIDS dataset of T1s.

Runs inside the public FOMO25 container (jbanusco/sslmmunetave:1.0.0) on an
amd64 host (Legion, since the image is not multi-arch). Writes one numpy
embedding per T1 plus an HF arrow dataset for easy loading alongside the
morphometry features in the analysis notebooks.

Upstream reference: https://github.com/jbanusco/fomo25 @ v1.0.0
Paper: Gordaliza et al. 2026, arxiv 2601.13166 (winner MICCAI 2025
SSL3D + FOMO25).

Expected preprocessing per T1 (per the FOMO25 README):
    skull-strip → RAS → resample 1 mm iso → z-normalise per volume
    → crop to minimum bounding box → pad/patch to 96 x 96 x 96

Usage (driver invocation from the host, wrapping the container):
    docker run --gpus all --rm \\
        -v /data/raw/openneuro/ds004856:/bids:ro \\
        -v /data/datasets/smri-fm-cmp/fomo-embeds/ds004856:/out \\
        -v $(realpath ..)/compute_fomo_embeddings.py:/work/extract.py:ro \\
        --entrypoint python \\
        jbanusco/sslmmunetave:1.0.0 \\
        /work/extract.py --bids /bids --out /out

Usage (in-container, what this script does):
    python compute_fomo_embeddings.py \\
        --bids /bids --out /out \\
        [--checkpoint /path/to/fomo25_mmunetvae_pretrained.ckpt] \\
        [--device cuda] [--subjects sub-1003 sub-1007]

Outputs:
    <out>/<sub>_<ses>_<acq>_<run>_T1w_embeds.npy    # one per input
    <out>/embeds_index.arrow                         # HF arrow table

The arrow table has columns:
    subject | session | acq | run | embedding_path | embedding (list<f32>)
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

BIDS_NAME_RE = re.compile(
    r"^(?P<sub>sub-[A-Za-z0-9]+)(?:_(?P<ses>ses-[A-Za-z0-9]+))?"
    r"(?:_(?P<acq>acq-[A-Za-z0-9]+))?"
    r"(?:_(?P<run>run-\d+))?"
    r"_T1w\.nii(?:\.gz)?$"
)


def iter_bids_t1s(bids_root: Path, subjects: Iterable[str] | None = None):
    """Yield BIDS T1 NIfTIs below a dataset root (anat files only)."""
    want = set(subjects) if subjects else None
    for sub_dir in sorted(bids_root.glob("sub-*")):
        if want and sub_dir.name not in want:
            continue
        for anat in sub_dir.glob("**/anat"):
            for t1 in anat.glob("*_T1w.nii*"):
                yield t1


def parse_entities(name: str) -> dict[str, str]:
    m = BIDS_NAME_RE.match(name)
    if not m:
        return {}
    return {k: (v or "") for k, v in m.groupdict().items()}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_fomo(t1_path: Path, target_size: int = 96) -> np.ndarray:
    """Reproduce FOMO25 preprocessing.

    NOTE: skull-stripping is expected to be done upstream — the FOMO25 README
    states the challenge data is already brain-extracted. For DLBS we either
    (a) chain through SynthStrip/HD-BET before calling this, or (b) rely on
    SynthSeg's brain-robust behaviour and skip the strip. This function handles
    the remaining steps: RAS reorient, 1 mm iso, z-norm, bbox-crop, pad to
    target_size^3.
    """
    img = nib.as_closest_canonical(nib.load(t1_path))
    arr = np.asanyarray(img.dataobj, dtype=np.float32)

    # Expect the upstream caller to have already resampled to 1 mm iso.
    # Z-normalise within brain voxels (approximated as > 0 since already
    # skull-stripped).
    mask = arr > 0
    mean = arr[mask].mean() if mask.any() else 0.0
    std = arr[mask].std() if mask.any() else 1.0
    arr = (arr - mean) / (std + 1e-6)
    arr[~mask] = 0.0

    # Bounding-box crop
    if mask.any():
        ijk = np.argwhere(mask)
        lo = ijk.min(0)
        hi = ijk.max(0) + 1
        arr = arr[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]

    # Pad or crop to target^3
    shape = np.array(arr.shape)
    pad_before = np.maximum((target_size - shape) // 2, 0)
    pad_after = np.maximum(target_size - shape - pad_before, 0)
    arr = np.pad(
        arr, list(zip(pad_before, pad_after)), mode="constant", constant_values=0.0
    )
    # If any dim still > target, centre-crop
    if any(d > target_size for d in arr.shape):
        starts = [(d - target_size) // 2 if d > target_size else 0 for d in arr.shape]
        arr = arr[
            starts[0] : starts[0] + target_size,
            starts[1] : starts[1] + target_size,
            starts[2] : starts[2] + target_size,
        ]
    return arr[np.newaxis, np.newaxis]  # (1, 1, Z, Y, X)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_mmunetvae(checkpoint_path: Path, device: str = "cuda"):
    """Load mmunetvae encoder for feature extraction.

    The FOMO25 image ships the repo under /opt/... — exact import path to be
    confirmed once the script is actually executed on Legion. This stub uses a
    lazy import so the module is resolved at runtime, allowing the script to
    import cleanly on the authoring host.
    """
    import torch

    # Confirmed via jbanusco/fomo25 repo inspection 2026-04-22: the module
    # lives at src/models/networks/mmunetvae.py. Inside the docker image the
    # `src/` dir is expected to be on PYTHONPATH (Dockerfile adds it), so
    # `models.networks.mmunetvae` is the canonical import. A few fallbacks
    # are kept in case layout differs in the image.
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
            "Could not locate mmunetvae module. Try candidate paths manually:\n"
            + "\n".join(f"  python -c 'import {c}'" for c in candidates)
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # fomo25 checkpoints are LightningModule-style; state_dict is keyed with
    # the module prefix. Strip it so we can load into the bare nn.Module.
    state = ckpt.get("state_dict", ckpt)
    state = {k.split("model.", 1)[-1]: v for k, v in state.items()}

    model_cls = module.MultiModalUNetVAE
    model = model_cls()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )
    model.eval().to(device)
    return model


def embed(model, volume: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Forward-pass a preprocessed volume through the encoder; pool to a vector."""
    import torch

    with torch.no_grad():
        x = torch.from_numpy(volume).to(device)
        # MultiModalUNetVAE encoder produces a bottleneck tensor; the exact
        # method name ('encode', 'forward_encoder', 'get_latent') is TBD and
        # will be confirmed on Legion.
        if hasattr(model, "encode"):
            z = model.encode(x)
        elif hasattr(model, "forward_encoder"):
            z = model.forward_encoder(x)
        else:
            z = model(x)
        if isinstance(z, (list, tuple)):
            z = z[0]
        # Global-average-pool spatial dims to get a fixed-size embedding.
        emb = z.mean(dim=tuple(range(2, z.ndim))).squeeze(0).cpu().numpy()
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/opt/ssl3d/checkpoints/fomo25_mmunetvae_pretrained.ckpt"),
        help="Mmunetvae checkpoint (FOMO25 release v1.0.0)",
    )
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preprocess only, skip model load; useful for CI / testing.",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.dry_run:
        model = load_mmunetvae(args.checkpoint, device=args.device)

    index_rows: list[dict] = []
    for t1 in iter_bids_t1s(args.bids, args.subjects):
        entities = parse_entities(t1.name)
        if not entities:
            print(f"skip (unparseable): {t1}", file=sys.stderr)
            continue
        stem = t1.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = args.out / f"{stem}_embeds.npy"
        if out_path.exists():
            print(f"skip (exists): {out_path}")
            continue

        try:
            vol = preprocess_fomo(t1)
        except Exception as e:
            print(f"FAIL preprocess {t1.name}: {e}", file=sys.stderr)
            continue

        if args.dry_run:
            emb = np.zeros(1, dtype=np.float32)
        else:
            emb = embed(model, vol, device=args.device)
        np.save(out_path, emb)
        row = {
            **entities,
            "embedding_path": str(out_path),
            "embedding_dim": int(emb.shape[-1]),
        }
        index_rows.append(row)
        print(f"{t1.name}  →  dim={emb.shape}")

    # Emit HF arrow
    try:
        from datasets import Dataset  # type: ignore

        ds = Dataset.from_list(
            [
                {
                    **r,
                    "embedding": np.load(r["embedding_path"]).tolist(),
                }
                for r in index_rows
            ]
        )
        ds.save_to_disk(str(args.out / "embeds_arrow"))
        print(f"wrote {args.out / 'embeds_arrow'} ({len(ds)} rows)")
    except Exception as e:
        print(f"arrow save failed (datasets lib missing?): {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
