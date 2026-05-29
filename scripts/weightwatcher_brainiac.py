"""weightwatcher analysis of BrainIAC's ViT backbone.

BrainIAC (AIM-KannLab, Nat. Neurosci. 2026) is a MONAI ViT-B (img 96^3, patch
16, hidden 768, 12 layers, 12 heads) pretrained with SimCLR. Most layers are
nn.Linear (attention qkv/proj + MLP), so ww's stock Linear path covers them and
the column set lines up with the SIAM / smri_mae runs. The one exception is the
patch-embedding, which in MONAI's ViT is a real **Conv3d** (kernel 16^3, in=1,
out=768) — so this run actually exercises the Conv3D patch we added to the
weightwatcher fork on a genuine FM, alongside the Linear path.

Loads the pretrained checkpoint (these weights are the whole point — unlike the
random-init smri_mae floor, BrainIAC ships trained), so the reported alpha /
log_norm are real FM-quality numbers comparable to the SIAM baseline.

Usage:
    PYTHONPATH=/workspace/weightwatcher:/workspace/brainiac/src \
      [BRAINIAC_CKPT=/workspace/brainiac/src/checkpoints/BrainIAC.ckpt] \
      python scripts/weightwatcher_brainiac.py [out_dir]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


DEFAULT_CKPT = "/workspace/brainiac/src/checkpoints/BrainIAC.ckpt"


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/fomo26_fm_benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = os.environ.get("BRAINIAC_CKPT", DEFAULT_CKPT)
    if not Path(ckpt).exists():
        print(f"ERROR: BrainIAC checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    from model import ViTBackboneNet  # BrainIAC/src/model.py (on PYTHONPATH)

    print(f">>> building BrainIAC ViTBackboneNet + loading {ckpt}")
    model = ViTBackboneNet(ckpt)
    model.eval()
    backbone = model.backbone  # MONAI ViT

    # Identify the conv3d patch-embed vs linear layers up front, for the report.
    convs = [n for n, m in backbone.named_modules() if isinstance(m, torch.nn.Conv3d)]
    lins = [n for n, m in backbone.named_modules() if isinstance(m, torch.nn.Linear)]
    print(f"    Conv3d layers (exercise the patch): {convs}")
    print(f"    Linear layers: {len(lins)}")

    import weightwatcher as ww

    print(">>> ww.analyze(backbone)  [patched fork: Conv3D patch-embed + Linear path]")
    watcher = ww.WeightWatcher(model=backbone)
    details = watcher.analyze()
    summary = watcher.get_summary(details)

    details_path = out_dir / "ww_brainiac_details.csv"
    summary_path = out_dir / "ww_brainiac_summary.json"
    details.to_csv(details_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    print()
    print(f"layers analyzed: {len(details)}")
    if "alpha" in details.columns and len(details) > 0:
        print(f"alpha:          mean={details['alpha'].mean():.3f}  median={details['alpha'].median():.3f}")
        if "alpha_weighted" in details.columns:
            print(f"alpha_weighted: mean={details['alpha_weighted'].mean():.3f}")
        if "stable_rank" in details.columns:
            print(f"stable_rank:    mean={details['stable_rank'].mean():.1f}")
        if "log_norm" in details.columns:
            print(f"log_norm:       mean={details['log_norm'].mean():.3f}")
        in_zone = ((details["alpha"] >= 2.0) & (details["alpha"] <= 6.0)).mean()
        under2 = (details["alpha"] < 2.0).mean()
        print(f"layers with alpha in [2,6] (well-trained zone): {in_zone:.0%}")
        print(f"layers with alpha<2 (over-trained / corr-trap):  {under2:.0%}")
        # Break out the conv patch-embed row(s) if present
        if "longname" in details.columns:
            conv_rows = details[details["longname"].astype(str).str.contains("patch_emb|conv", case=False, na=False)]
            for _, r in conv_rows.iterrows():
                print(f"    [conv] {r.get('longname')}: alpha={r.get('alpha'):.3f} log_norm={r.get('log_norm'):.3f}")
    print()
    print(f"wrote {details_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
