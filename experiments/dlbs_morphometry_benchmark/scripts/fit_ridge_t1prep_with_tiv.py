"""Fit ridge on T1Prep thickness or area features, properly normalised by TIV.

The default fit_ridge_baseline.py only normalises by an ICV region IF that
region appears in the same tool's pivot. T1Prep's TIV lives in the
`t1prep_tissue` tool, not in `t1prep_thickness` or `t1prep_area`, so the
existing _icv variants for thickness/area silently no-op'd.

This runner pulls TIV from `t1prep_tissue`, joins it into the thickness/
area wide form, divides each feature by TIV, and runs the same Nima
GroupKFold(5) + StandardScaler + RidgeCV pipeline + 4 bias-correction
schemes.

Usage:
  python fit_ridge_t1prep_with_tiv.py --tool t1prep_thickness \\
      --features results/t1prep_features.parquet \\
      --participants /data/raw/openneuro/ds004856/participants.tsv \\
      --out results/ridge_t1prep_thickness_TIV_real.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from fit_ridge_baseline import (  # noqa: E402
    beheshti_correction,
    cole_correction,
    load_participants_age,
    long_to_wide,
    metrics,
    zhang_correction,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--tool", required=True,
                    help="t1prep_thickness | t1prep_area")
    ap.add_argument("--value-col", default=None,
                    help="thickness_mm or area_mm2 — defaults to match --tool")
    ap.add_argument("--participants", type=Path,
                    default=Path("/data/raw/openneuro/ds004856/participants.tsv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.value_col is None:
        args.value_col = {
            "t1prep_thickness": "thickness_mm",
            "t1prep_area": "area_mm2",
        }.get(args.tool, "value")
    print(f"tool={args.tool}  value_col={args.value_col}")

    raw = pd.read_parquet(args.features)

    # 1. Wide form for the chosen tool
    wide = long_to_wide(raw, args.tool, args.value_col)
    print(f"  wide shape: {wide.shape}")

    # 2. Extract TIV from t1prep_tissue
    tissue = raw[raw["tool"] == "t1prep_tissue"].copy()
    if tissue.empty:
        sys.exit("no t1prep_tissue rows in features parquet — can't normalize")
    # tissue has GM/WM/CSF/TIV regions; extract just TIV
    tiv = tissue[tissue["region"] == "TIV"][["subject", "session", "volume_mm3"]].rename(
        columns={"volume_mm3": "TIV"}
    )
    if tiv.empty:
        # Try alternate value column names
        for vc in ("volume_mm3", "value", "thickness_mm", "tissue_volume_mm3"):
            t = tissue[tissue["region"] == "TIV"]
            if vc in t.columns and t[vc].notna().any():
                tiv = t[["subject", "session", vc]].rename(columns={vc: "TIV"})
                break
    if tiv.empty:
        # Inspect to debug
        print(f"  tissue rows: {len(tissue)}, regions: {tissue['region'].unique().tolist()}")
        print(f"  tissue cols: {tissue.columns.tolist()}")
        sys.exit("can't locate TIV value column in t1prep_tissue rows")
    print(f"  TIV rows: {len(tiv)}, mean={tiv['TIV'].mean():.0f}, std={tiv['TIV'].std():.0f}")

    # 3. Join + divide each thickness/area column by TIV
    df_with_tiv = wide.merge(tiv, on=["subject", "session"], how="inner")
    feature_cols = [c for c in df_with_tiv.columns if c not in ("subject", "session", "TIV")]
    print(f"  features before normalize: {len(feature_cols)}")
    icv = df_with_tiv["TIV"].replace(0, np.nan)
    for c in feature_cols:
        df_with_tiv[c] = df_with_tiv[c] / icv

    # 4. Attach age + ridge
    ages = load_participants_age(args.participants)
    df = df_with_tiv.drop(columns=["TIV"]).merge(ages, on=["subject", "session"], how="inner")
    print(f"  after age join: {df.shape} subjects: {df['subject'].nunique()}")

    feature_cols = [c for c in df.columns if c not in ("subject", "session", "age")]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["age"].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    # 5. GroupKFold(5) ridge
    splitter = GroupKFold(n_splits=min(5, df["subject"].nunique()))
    preds = np.zeros_like(y)
    for tr, te in splitter.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])

    results = {
        "tool": args.tool,
        "value_col": args.value_col,
        "n_features": int(X.shape[1]),
        "n_scans": int(len(y)),
        "n_subjects": int(df["subject"].nunique()),
        "cv_mode": "groupkfold5",
        "icv_normalised": True,
        "icv_source": "TIV from t1prep_tissue (cross-tool join)",
        "raw":      metrics(y, preds),
        "cole":     metrics(y, cole_correction(y, preds)),
        "beheshti": metrics(y, beheshti_correction(y, preds)),
        "zhang":    metrics(y, zhang_correction(y, preds)),
    }
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n--- {args.tool} (TIV-normalized via cross-tool join) ---")
    for s in ("raw", "cole", "beheshti", "zhang"):
        m = results[s]
        print(f"  {s:10s} MAE={m['mae']:6.3f}  r={m['pearson_r']:+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
