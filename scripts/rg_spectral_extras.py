"""RG/SETOL weights-only layer diagnostics on top of the Conv3D-patched ww fork.

Implements the extra per-layer diagnostics from C. H. Martin, "Renormalization
Group Theory of Learning" (2026) that plain alpha under-determines:

  - phi_k      dominant-tail fraction = sum(top-k eigs) / sum(all eigs)   (Eq 18)
  - M_tr       participation count    = (sum eig)^2 / sum(eig^2)          (effective
               number of contributing eigencomponents; large = self-averaging)
  - num_traps  Correlation-Trap count = eigenvalues beyond the randomized-MP edge
               (taken straight from ww's randomize=True pass; Figs 5-6 of the paper)

phi_k / M_tr are computed from each layer's eigenvalue spectrum of X = W^T W / N,
fetched via WeightWatcher.get_ESD(layer=id) so they use the SAME per-slice Conv3D
decomposition ww uses for alpha (apples-to-apples with the canonical FM-quality
default). alpha + num_traps come from ww.analyze(randomize=True, mp_fit=True).

Reading: a self-averaging ("good") layer has small phi_1, large M_tr, num_traps=0,
and alpha ~ 2. A non-self-averaging (overfit-risk) layer concentrates gain in a
few dominant modes: large phi_1, small M_tr, num_traps > 0, often alpha < 2.

Use as a library (compute_rg_extras / summarize_rg_extras) from the per-arm ww
scripts, or as a CLI that loads any structurebench arm via its asparagus bridge:

    PYTHONPATH=/home/mhough/dev/weightwatcher:src \
      SIAM_MODEL_DIR=/home/mhough/siam_params/v0.3/pred_DS108_LcsfP_Ano \
      python scripts/rg_spectral_extras.py smri_siam
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# core: eigenvalue-derived RG diagnostics
# ----------------------------------------------------------------------------
def _eig_metrics(evals, ks=(1, 5, 10)) -> dict:
    e = np.asarray(evals, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    out: dict = {}
    if e.size == 0:
        for k in ks:
            out[f"phi_{k}"] = np.nan
        out.update(M_tr=np.nan, M_tr_frac=np.nan, num_evals_pos=0)
        return out
    s, s2 = float(e.sum()), float((e * e).sum())
    es = np.sort(e)[::-1]
    for k in ks:
        out[f"phi_{k}"] = float(es[:k].sum() / s)
    out["M_tr"] = float(s * s / s2)
    out["M_tr_frac"] = float((s * s / s2) / e.size)  # effective fraction of modes
    out["num_evals_pos"] = int(e.size)
    return out


def isolated_trap_counts(watcher, model=None) -> dict:
    """Map layer_id -> number of *isolated* Correlation Traps (the Fig 5-6
    randomized/permuted workflow: rank-1 modes beyond the permuted-MP edge).
    This is the meaningful trap count; the plain `num_traps`/`rand_num_spikes`
    details column instead counts the whole correlated tail above the bulk."""
    try:
        # analyze_traps needs a pre-randomized (permuted) model + its trap_state;
        # it does not randomize internally. NB the fork's randomize_model skips
        # layers whose Wmats != 1 (i.e. multi-slice per-slice Conv3D), so on pure
        # Conv3D encoders the isolated-trap workflow only covers single-matrix
        # (Linear / rf=1) layers.
        rand_model, trap_state = watcher.randomize_model(model=model, return_state=True)
        permuted = sorted(trap_state.get("permuted_ids", {}).keys())
        if not permuted:
            print("    [warn] no single-matrix layers were permuted "
                  "(pure multi-slice Conv3D) — isolated-trap count unavailable",
                  file=sys.stderr)
            return {}
        traps = watcher.analyze_traps(randomized_model=rand_model,
                                      trap_state=trap_state, layers=permuted)
    except Exception as exc:
        print(f"    [warn] analyze_traps failed: {exc}", file=sys.stderr)
        return {}
    if isinstance(traps, tuple):  # some ww paths return (details, summary/state)
        traps = next((t for t in traps if isinstance(t, pd.DataFrame)), None)
    if traps is None or len(traps) == 0:
        return {}
    id_col = next((c for c in ("layer_id", "id", "ww_layer_id") if c in traps.columns), None)
    if id_col is None:
        return {}
    return {int(k): int(v) for k, v in traps.groupby(id_col).size().items()}


def compute_rg_extras(watcher, details: pd.DataFrame, ks=(1, 5, 10),
                      trap_counts: dict | None = None) -> pd.DataFrame:
    """One row per analyzed layer: phi_k / M_tr from get_ESD, plus alpha and the
    isolated-trap count merged in by layer_id."""
    carry = [c for c in ("alpha", "alpha_weighted", "num_pl_spikes",
                          "rand_num_spikes", "lambda_max", "Q")
             if c in details.columns]
    trap_counts = trap_counts or {}
    rows = []
    for _, r in details.iterrows():
        lid = int(r["layer_id"])
        rec = {"layer_id": lid, "name": r.get("name")}
        try:
            evals = watcher.get_ESD(layer=lid)
        except Exception as exc:  # a layer ww can describe but not ESD
            print(f"    [warn] get_ESD failed for layer {lid}: {exc}", file=sys.stderr)
            evals = []
        rec.update(_eig_metrics(evals, ks))
        for c in carry:
            rec[c] = r.get(c)
        rec["n_traps_isolated"] = int(trap_counts.get(lid, 0))
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_rg_extras(df: pd.DataFrame, ks=(1, 5, 10)) -> dict:
    summ: dict = {"n_layers": int(len(df))}
    for c in [f"phi_{k}" for k in ks] + ["M_tr", "M_tr_frac", "alpha"]:
        if c in df.columns:
            summ[f"{c}_mean"] = float(df[c].mean())
            summ[f"{c}_median"] = float(df[c].median())
    for c in ("n_traps_isolated", "rand_num_spikes", "num_pl_spikes"):
        if c in df.columns:
            summ[f"{c}_total"] = float(df[c].fillna(0).sum())
            summ[f"layers_with_{c}"] = int((df[c].fillna(0) > 0).sum())
    if "alpha" in df.columns:
        summ["frac_alpha_lt2"] = float((df["alpha"] < 2.0).mean())
    return summ


# ----------------------------------------------------------------------------
# arm loaders (mirror scripts/modal_cls002_sweep.py CONVERT + the v2 ww scripts)
# ----------------------------------------------------------------------------
# arm -> (backbone module, backbone class, converter fn or None for SIAM)
ARMS = {
    "smri_siam":     ("asparagus_bridge.models_smri_siam",     "SmriSiamClsRegBackbone",     None),
    "smri_fomo60k":  ("asparagus_bridge.models_smri_fomo60k",  "SmriFomo60kClsRegBackbone",  "convert_fomo60k_checkpoint"),
    "smri_brainiac": ("asparagus_bridge.models_smri_brainiac", "SmriBrainiacClsRegBackbone", "convert_brainiac_checkpoint"),
    "smri_triad":    ("asparagus_bridge.models_smri_triad",    "SmriTriadClsRegBackbone",    "convert_triad_checkpoint"),
    "smri_anatcl":   ("asparagus_bridge.models_smri_anatcl",   "SmriAnatclClsRegBackbone",   "convert_anatcl_checkpoint"),
    "smri_mmunetvae":("asparagus_bridge.models_smri_mmunetvae","SmriMmunetvaeClsRegBackbone","convert_mmunetvae_checkpoint"),
    "smri_simclr3d": ("asparagus_bridge.models_smri_simclr3d", "SmriSimclr3dClsRegBackbone", "convert_simclr3d_checkpoint"),
}
# local source checkpoints (match scripts/modal_upload.sh sources)
_W = "/data/datasets/fomo26/weights"
DEFAULT_CKPT = {
    "smri_fomo60k":  f"{_W}/fomo60k_pkoutsouvelis/combined_regular-step=200000.ckpt",
    "smri_brainiac": "/home/mhough/dev/BrainIAC/src/checkpoints/BrainIAC.ckpt",
    "smri_triad":    f"{_W}/triad/Triad-SwinB-MAE.pth",
    "smri_anatcl":   f"{_W}/anatcl/anatcl_global_fold0.pth",
    "smri_mmunetvae":f"{_W}/mmunetvae/fomo25_mmunetvae_pretrained.ckpt",
    "smri_simclr3d": f"{_W}/simclr3d/simclr_3d_brain_foundation.tar",
}


def _build_model(arm: str, ckpt: str | None):
    import importlib
    import torch
    from asparagus_bridge.checkpoint import convert_checkpoint

    mod_name, cls_name, conv_fn = ARMS[arm]
    Backbone = getattr(importlib.import_module(mod_name), cls_name)

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / f"{arm}.ckpt"
        if arm == "smri_siam":
            siam_dir = Path(os.environ["SIAM_MODEL_DIR"])
            convert_checkpoint("smri_siam", siam_dir / "fold_0" / "checkpoint_final.pth", dst)
        else:
            getattr(importlib.import_module(mod_name), conv_fn)(ckpt or DEFAULT_CKPT[arm], dst)
        converted = torch.load(dst, map_location="cpu", weights_only=False)

    sd = {k[len("model."):]: v for k, v in converted["state_dict"].items()
          if k.startswith("model.")}
    model = Backbone(input_channels=1, output_channels=1)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"    loaded {len(sd) - len(unexp)} tensors  (missing={len(miss)} unexpected={len(unexp)})")

    # analyze the pretrained submodule (exclude the fresh task head) for an
    # apples-to-apples per-slice spectrum; fall back to the whole backbone.
    for attr in ("encoder", "net", "backbone"):
        sub = getattr(model, attr, None)
        if sub is not None and any(p.requires_grad for p in sub.parameters()):
            print(f"    analyzing submodule .{attr}")
            return sub
    print("    analyzing whole backbone (no encoder/net/backbone submodule found)")
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--ckpt", default=None, help="override source checkpoint")
    ap.add_argument("--out-dir", default="experiments/fomo26_fm_benchmark/results")
    ap.add_argument("--ks", default="1,5,10")
    args = ap.parse_args()
    ks = tuple(int(x) for x in args.ks.split(","))

    if args.arm == "smri_siam" and not os.environ.get("SIAM_MODEL_DIR"):
        print("ERROR: SIAM_MODEL_DIR is not set", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> building {args.arm}")
    net = _build_model(args.arm, args.ckpt)

    import weightwatcher as ww
    print(">>> ww.analyze(randomize=True, mp_fit=True)  [Conv3D-patched fork]")
    watcher = ww.WeightWatcher(model=net)
    details = watcher.analyze(randomize=True, mp_fit=True)

    print(">>> ww.analyze_traps()  [isolated Correlation Traps]")
    trap_counts = isolated_trap_counts(watcher, model=net)

    extras = compute_rg_extras(watcher, details, ks=ks, trap_counts=trap_counts)
    summary = summarize_rg_extras(extras, ks=ks)

    tag = args.arm[len("smri_"):]
    details_path = out_dir / f"rg_extras_{tag}_details.csv"
    summary_path = out_dir / f"rg_extras_{tag}_summary.json"
    extras.to_csv(details_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))

    print()
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {details_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
