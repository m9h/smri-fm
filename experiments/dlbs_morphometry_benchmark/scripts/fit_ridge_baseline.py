"""Fit a ridge brain-age regressor on a parquet of morphometric features.

Reads a long-format parquet produced by `extract_{fastsurfer,t1prep}_features.py`,
pivots to wide (one row per scan, features as columns), joins with age from the
DLBS `participants.tsv`, and fits ridge regression with leave-one-subject-out
cross-validation. Reports MAE / RMSE / R² / Pearson r / bias under four
bias-correction regimes: uncorrected, Cole-Smith (Smith 2019), Beheshti
(Liang 2019), and Zhang age-level (Zhang 2023). Emits a JSON metrics file
and a scatter PNG.

Intended as **rung 1–4** of the four-rung ladder; which rung depends on the
parquet's `tool` filter.

Usage:
    fit_ridge_baseline.py \\
        --features results/fastsurfer_features.parquet \\
        --tool aseg+DKT.VINN \\
        --participants /data/raw/openneuro/ds004856/participants.tsv \\
        --normalise-by-icv \\
        --out results/ridge_fastsurfer_aseg+DKT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def wave_to_age_col() -> dict[str, str]:
    # Map BIDS session name → participants.tsv column holding the MRI age
    return {"ses-wave1": "AgeMRI_W1", "ses-wave2": "AgeMRI_W2", "ses-wave3": "AgeMRI_W3"}


def long_to_wide(df: pd.DataFrame, tool: str, value_col: str) -> pd.DataFrame:
    sub = df[df["tool"] == tool].copy()
    if sub.empty:
        raise ValueError(f"no rows for tool={tool!r}")
    wide = sub.pivot_table(
        index=["subject", "session"], columns="region", values=value_col
    ).reset_index()
    wide.columns.name = None
    return wide


def load_participants_age(tsv: Path) -> pd.DataFrame:
    p = pd.read_csv(tsv, sep="\t")
    rows = []
    for col, ses in {"AgeMRI_W1": "ses-wave1", "AgeMRI_W2": "ses-wave2", "AgeMRI_W3": "ses-wave3"}.items():
        for _, r in p.iterrows():
            if col in p.columns and pd.notna(r[col]) and str(r[col]) != "n/a":
                try:
                    rows.append(
                        {"subject": r["participant_id"], "session": ses, "age": float(r[col])}
                    )
                except (ValueError, TypeError):
                    continue
    return pd.DataFrame(rows)


def cole_correction(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Smith 2019 / Cole linear correction: regress pred on true, invert."""
    if len(y_true) < 2:
        return y_pred.copy()
    alpha, beta = np.polyfit(y_true, y_pred, 1)
    return (y_pred - beta) / max(alpha, 1e-6)


def beheshti_correction(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Beheshti (Liang 2019): regress residual on age, subtract fit."""
    if len(y_true) < 2:
        return y_pred.copy()
    resid = y_pred - y_true
    alpha, beta = np.polyfit(y_true, resid, 1)
    return y_pred - (alpha * y_true + beta)


def zhang_correction(y_true: np.ndarray, y_pred: np.ndarray, bin_width: float = 5.0) -> np.ndarray:
    """Zhang 2023 age-level: z-score residuals within each age bin."""
    resid = y_pred - y_true
    out = y_pred.copy()
    if len(y_true) < 5:
        return out
    bins = np.floor(y_true / bin_width).astype(int)
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() < 2:
            continue
        mu, sd = resid[sel].mean(), resid[sel].std()
        if sd < 1e-6:
            continue
        out[sel] = y_true[sel] + (resid[sel] - mu) / sd * sd
    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if len(y_true) >= 2:
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r2, pearson = float("nan"), float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": pearson, "bias": bias, "n": int(len(y_true))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--tool", type=str, required=True, help='tool filter (e.g. "aseg+DKT.VINN")')
    ap.add_argument(
        "--value-col",
        default="volume_mm3",
        help='column in the features parquet to use (volume_mm3, thickness_mm, area_mm2)',
    )
    ap.add_argument(
        "--participants",
        type=Path,
        default=Path("/data/raw/openneuro/ds004856/participants.tsv"),
    )
    ap.add_argument(
        "--normalise-by-icv",
        action="store_true",
        help='divide all features by total intracranial volume (requires a TIV-equivalent column)',
    )
    ap.add_argument(
        "--icv-region",
        default="total intracranial",
        help='Region whose volume_mm3 is treated as ICV (FS: total intracranial; T1Prep: TIV)',
    )
    ap.add_argument(
        "--cv-mode",
        choices=["loso", "groupkfold5"],
        default="groupkfold5",
        help='LOSO = LeaveOneSubjectOut (deep longitudinal). '
             'groupkfold5 = GroupKFold(n_splits=5) by subject — matches '
             'MedARC smri-fm/experiments/synthseg_ridge_baseline default.',
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--wandb-project",
        default=None,
        help='W&B project name. If set, logs metrics per bias-correction scheme. '
             'Run "wandb login" first. Example: dlbs-morphometry-benchmark',
    )
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    wb = None
    if args.wandb_project:
        try:
            import wandb
            wb = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name
                or f"{args.tool}-{'icv' if args.normalise_by_icv else 'raw'}",
                config={
                    "tool": args.tool,
                    "value_col": args.value_col,
                    "features_parquet": str(args.features),
                    "normalise_by_icv": args.normalise_by_icv,
                },
                tags=["ridge", "morphometry", "dlbs", args.tool],
                reinit=True,
            )
        except Exception as e:
            print(f"wandb init failed: {e}; continuing without tracking", file=sys.stderr)

    features = pd.read_parquet(args.features)
    wide = long_to_wide(features, args.tool, args.value_col)
    print(f"wide shape: {wide.shape}")

    # Attach age
    ages = load_participants_age(args.participants)
    df = wide.merge(ages, on=["subject", "session"], how="inner")
    print(f"after age join: {df.shape}; subjects: {df['subject'].nunique()}")

    if df.empty or df["subject"].nunique() < 2:
        print("WARNING: need >= 2 subjects for LOSO — running on-sample for dry-run")

    feature_cols = [c for c in df.columns if c not in ("subject", "session", "age")]
    # Optional ICV normalisation
    if args.normalise_by_icv and args.icv_region in feature_cols:
        icv = df[args.icv_region].replace(0, np.nan)
        for c in feature_cols:
            if c == args.icv_region:
                continue
            df[c] = df[c] / icv
    feature_cols = [c for c in feature_cols if c != args.icv_region]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["age"].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()

    # Replace any NaN with column median
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    # LOSO ridge — skip if only one group. StandardScaler+Ridge pipeline:
    # with mixed feature scales inside a single rung (volumes in mm³, etc.)
    # an unscaled ridge silently down-weights small-magnitude features.
    def _mk_ridge():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ])

    preds = np.zeros_like(y)
    n_groups = df["subject"].nunique()
    if n_groups >= 2:
        if args.cv_mode == "loso":
            splitter = LeaveOneGroupOut()
        else:  # groupkfold5 — MedARC's smri-fm default
            # n_splits capped at n_groups to avoid a ValueError on tiny cohorts
            splitter = GroupKFold(n_splits=min(5, n_groups))
        for tr, te in splitter.split(X, y, groups):
            model = _mk_ridge()
            model.fit(X[tr], y[tr])
            preds[te] = model.predict(X[te])
    else:
        model = _mk_ridge()
        model.fit(X, y)
        preds = model.predict(X)

    results = {
        "tool": args.tool,
        "value_col": args.value_col,
        "n_features": int(X.shape[1]),
        "n_scans": int(len(y)),
        "n_subjects": int(df["subject"].nunique()),
        "cv_mode": args.cv_mode,
        "icv_normalised": bool(args.normalise_by_icv),
        "raw": metrics(y, preds),
        "cole": metrics(y, cole_correction(y, preds)),
        "beheshti": metrics(y, beheshti_correction(y, preds)),
        "zhang": metrics(y, zhang_correction(y, preds)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    if wb is not None:
        # Flatten correction-scheme metrics for wandb's key-value logger
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
        try:
            import wandb  # noqa: F401

            wb.save(str(args.out))
        except Exception:
            pass
        wb.finish()


if __name__ == "__main__":
    main()
