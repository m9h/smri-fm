"""Concatenate morphometric + foundation-model features → LOSO ridge.

Reads one or more feature parquets (produced by
`extract_{fastsurfer,t1prep}_features.py` or `extract_brainiac_embeddings.py`),
pivots each to wide form, joins on (subject, session), and fits a single
RidgeCV regressor with leave-one-subject-out CV.

Emits the same JSON schema as `fit_ridge_baseline.py` with an added
`feature_sources` array (one entry per input parquet).

The key claim this script lets us test: does adding T1Prep thickness
*and* BrainIAC-768 on top of FastSurfer volumes do meaningfully better
than any single source? If MAE drops substantially, morphometry + FM
are complementary — that's the headline MedARC message.

Usage:
    fit_ridge_concat.py \\
        --source fastsurfer_features.parquet:aseg+DKT.VINN:volume_mm3 \\
        --source t1prep_features.parquet:t1prep_thickness:thickness_mm \\
        --source brainiac_embeddings.parquet:brainiac_embed:value \\
        --participants /data/raw/openneuro/ds004856/participants.tsv \\
        --out results/ridge_concat.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse bias-correction + metric helpers from the single-source script.
sys.path.insert(0, str(Path(__file__).parent))
from fit_ridge_baseline import (  # noqa: E402
    beheshti_correction,
    cole_correction,
    load_participants_age,
    long_to_wide,
    metrics,
    zhang_correction,
)


def parse_source(spec: str) -> tuple[Path, str, str]:
    """`path:tool:value_col` → (Path, tool, value_col)."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--source must be path:tool:value_col, got {spec!r}")
    return Path(parts[0]), parts[1], parts[2]


def load_source(path: Path, tool: str, value_col: str) -> tuple[pd.DataFrame, str]:
    """Returns a wide DF on (subject, session) and a tag used to prefix its feature columns."""
    df = pd.read_parquet(path)
    wide = long_to_wide(df, tool, value_col)
    # Tag feature columns so the concat step cannot collide across sources
    # (e.g. FastSurfer "Left-Hippocampus" volume vs T1Prep "Left-Hippocampus"
    # thickness would otherwise share a column name).
    tag = tool.replace(" ", "_").replace("+", "_").replace(".", "_")
    rename = {
        c: f"{tag}__{c}"
        for c in wide.columns
        if c not in ("subject", "session")
    }
    return wide.rename(columns=rename), tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help='`parquet_path:tool:value_col`, may be given multiple times',
    )
    ap.add_argument(
        "--participants",
        type=Path,
        default=Path("/data/raw/openneuro/ds004856/participants.tsv"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--wandb-project",
        default=None,
        help='W&B project name. If set, logs metrics per bias-correction scheme.',
    )
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    if len(args.source) < 2:
        print(
            "WARNING: concat with < 2 sources is equivalent to fit_ridge_baseline.py",
            file=sys.stderr,
        )

    feature_sources: list[dict] = []
    merged: pd.DataFrame | None = None
    for spec in args.source:
        path, tool, value_col = parse_source(spec)
        wide, tag = load_source(path, tool, value_col)
        feature_sources.append(
            {
                "parquet": str(path),
                "tool": tool,
                "value_col": value_col,
                "tag": tag,
                "n_features": int(wide.shape[1] - 2),  # minus subject + session
                "n_scans_in_source": int(wide.shape[0]),
            }
        )
        if merged is None:
            merged = wide
        else:
            merged = merged.merge(wide, on=["subject", "session"], how="inner")
    assert merged is not None
    print(f"merged shape after inner-join on (subject, session): {merged.shape}")

    ages = load_participants_age(args.participants)
    df = merged.merge(ages, on=["subject", "session"], how="inner")
    print(f"after age join: {df.shape}; subjects: {df['subject'].nunique()}")

    if df.empty or df["subject"].nunique() < 2:
        raise SystemExit("need >= 2 subjects with at least one scan across all sources")

    feature_cols = [c for c in df.columns if c not in ("subject", "session", "age")]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["age"].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()

    # NaN imputation (column median) — some FM embeddings may have
    # NaNs from dead neurons or missing scans.
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    # LOSO ridge with per-fold standardisation — critical when mixing
    # units across feature sources (mm³, mm, unit-less embeddings).
    preds = np.zeros_like(y)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ])
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])

    results = {
        "feature_sources": feature_sources,
        "n_features": int(X.shape[1]),
        "n_scans": int(len(y)),
        "n_subjects": int(df["subject"].nunique()),
        "raw": metrics(y, preds),
        "cole": metrics(y, cole_correction(y, preds)),
        "beheshti": metrics(y, beheshti_correction(y, preds)),
        "zhang": metrics(y, zhang_correction(y, preds)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    if args.wandb_project:
        try:
            import wandb
            wb = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or "concat-" + "+".join(
                    s["tag"] for s in feature_sources
                ),
                config={
                    "feature_sources": feature_sources,
                    "n_features": results["n_features"],
                },
                tags=["ridge", "concat", "dlbs"],
                reinit=True,
            )
            flat = {
                f"{scheme}/{k}": v
                for scheme in ("raw", "cole", "beheshti", "zhang")
                for k, v in results[scheme].items()
            }
            flat.update(
                n_scans=results["n_scans"],
                n_subjects=results["n_subjects"],
                n_features=results["n_features"],
            )
            wb.log(flat)
            wb.summary.update(flat)
            wb.save(str(args.out))
            wb.finish()
        except Exception as e:
            print(f"wandb logging failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
