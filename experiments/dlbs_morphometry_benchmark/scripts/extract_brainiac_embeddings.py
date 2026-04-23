"""Extract BrainIAC 768-d CLS-token embeddings for DLBS T1s.

Reuses Nima's DLBS → BrainIAC preprocessing CSV + root directory from
`experiments/brainiac_dlbs_eval/` and the BrainIAC modules
(`BrainAgeDataset`, `get_validation_transform`, `ViTBackboneNet`).
Swaps the fine-tuned brain-age classifier for the raw backbone output
(the 768-d CLS token per scan), so downstream ridge can exploit
features the brain-age head may have discarded.

Writes:
- `<out_dir>/brainiac_embeddings.parquet` — long format
    subject, session, run, acq, embedding_idx, value
  (compatible with fit_ridge_baseline.py if we reshape; see --wide
  flag for a wide version too).
- `<out_dir>/brainiac_embeddings.npz` — dict of `<key>:(768,) array`
  where `<key>` encodes subject/session/acq/run.

Usage:
    python extract_brainiac_embeddings.py \\
        --brainiac_src /home/mhough/dev/BrainIAC/src \\
        --input_csv /home/mhough/dev/smri-fm/experiments/brainiac_dlbs_eval/DLBS/brainage_100.csv \\
        --root_dir /home/mhough/dev/smri-fm/experiments/brainiac_dlbs_eval/DLBS/processed_brainiac_100 \\
        --simclr_checkpoint /home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt \\
        --out_dir /home/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/results
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


BIDS_RE = re.compile(
    r"(?P<sub>sub-[A-Za-z0-9]+)"
    r"(?:_(?P<ses>ses-[A-Za-z0-9]+))?"
    r"(?:_(?P<acq>acq-[A-Za-z0-9]+))?"
    r"(?:_(?P<run>run-\d+))?"
)


def parse_entities(pat_id: str) -> dict[str, str]:
    """Extract BIDS entities from the input_csv's pat_id field."""
    m = BIDS_RE.search(pat_id)
    if not m:
        return {"sub": pat_id, "ses": "", "acq": "", "run": ""}
    return {k: (v or "") for k, v in m.groupdict().items()}


def load_brainiac(brainiac_src: Path):
    if not (brainiac_src / "model.py").exists():
        raise SystemExit(f"Expected BrainIAC src at {brainiac_src}")
    sys.path.insert(0, str(brainiac_src))
    from dataset import BrainAgeDataset, get_validation_transform  # type: ignore
    from model import ViTBackboneNet  # type: ignore
    return BrainAgeDataset, get_validation_transform, ViTBackboneNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--brainiac_src", type=Path, required=True)
    p.add_argument("--input_csv", type=Path, required=True,
                   help="Nima's DLBS inference CSV (pat_id, label)")
    p.add_argument("--root_dir", type=Path, required=True,
                   help="Dir of BrainIAC-preprocessed .nii.gz files")
    p.add_argument("--simclr_checkpoint", type=Path, required=True,
                   help="BrainIAC SimCLR backbone checkpoint")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    BrainAgeDataset, get_validation_transform, ViTBackboneNet = (
        load_brainiac(args.brainiac_src)
    )

    # Load the pretrained backbone (NO fine-tuned classifier head)
    print(f"loading backbone from {args.simclr_checkpoint}")
    backbone = ViTBackboneNet(simclr_ckpt_path=str(args.simclr_checkpoint))
    backbone.eval().to(device)

    # Reuse Nima's dataset + transform exactly so preprocessing matches her
    # brain-age inference runs.
    df = pd.read_csv(args.input_csv)
    transform = get_validation_transform()
    dataset = BrainAgeDataset(
        df=df,
        root_dir=str(args.root_dir),
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    print(f"extracting {len(dataset)} embeddings...")
    for batch in tqdm(loader):
        x = batch["image"].to(device)
        pat_ids = batch.get("pat_id", [f"unknown_{i}" for i in range(x.shape[0])])
        with torch.no_grad():
            cls = backbone(x)   # [B, 768]
        cls = cls.cpu().numpy()
        for i, pid in enumerate(pat_ids):
            ids.append(str(pid))
            embeddings.append(cls[i])

    embs = np.stack(embeddings)     # (N, 768)
    print(f"stacked embeddings: {embs.shape}")

    # Save wide npz — key by pat_id
    np.savez(
        args.out_dir / "brainiac_embeddings.npz",
        **{pid: emb for pid, emb in zip(ids, embs)},
    )

    # Save long-format parquet compatible with fit_ridge_baseline.py
    rows = []
    for pid, emb in zip(ids, embs):
        ents = parse_entities(pid)
        for idx, val in enumerate(emb):
            rows.append({
                "subject": ents["sub"],
                "session": ents["ses"],
                "acq": ents["acq"],
                "run": ents["run"],
                "tool": "brainiac_embed",
                "region": f"embed_{idx:03d}",
                "value": float(val),
            })
    long_df = pd.DataFrame(rows)
    long_df.to_parquet(args.out_dir / "brainiac_embeddings.parquet", index=False)
    print(
        f"wrote {args.out_dir / 'brainiac_embeddings.npz'} and "
        f"{args.out_dir / 'brainiac_embeddings.parquet'} ({len(long_df)} rows)"
    )


if __name__ == "__main__":
    main()
