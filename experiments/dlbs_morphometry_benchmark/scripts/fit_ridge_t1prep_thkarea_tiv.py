"""T1Prep thickness+area concat with TIV-normalization (cross-tool TIV join).

Mirrors the existing ridge_t1prep_thkarea.json run but actually divides
both thickness (mm) and area (mm²) by TIV from t1prep_tissue. Uses the
same Nima protocol.
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
    beheshti_correction, cole_correction,
    load_participants_age, long_to_wide, metrics, zhang_correction,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--participants", type=Path,
                    default=Path("/data/raw/openneuro/ds004856/participants.tsv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    raw = pd.read_parquet(args.features)

    thk = long_to_wide(raw, "t1prep_thickness", "thickness_mm")
    area = long_to_wide(raw, "t1prep_area", "area_mm2")
    # Tag columns so they don't collide
    thk = thk.rename(columns={c: f"thk_{c}" for c in thk.columns if c not in ("subject","session")})
    area = area.rename(columns={c: f"area_{c}" for c in area.columns if c not in ("subject","session")})

    tissue = raw[raw["tool"] == "t1prep_tissue"]
    tiv = tissue[tissue["region"] == "TIV"][["subject", "session", "volume_mm3"]].rename(
        columns={"volume_mm3": "TIV"}
    )

    df = thk.merge(area, on=["subject", "session"], how="inner")
    df = df.merge(tiv, on=["subject", "session"], how="inner")
    print(f"  merged shape: {df.shape}, n_subj: {df['subject'].nunique()}")

    feat_cols = [c for c in df.columns if c not in ("subject", "session", "TIV")]
    icv = df["TIV"].replace(0, np.nan)
    for c in feat_cols:
        df[c] = df[c] / icv
    df = df.drop(columns=["TIV"])

    ages = load_participants_age(args.participants)
    df = df.merge(ages, on=["subject", "session"], how="inner")
    feat_cols = [c for c in df.columns if c not in ("subject", "session", "age")]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["age"].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

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
        "tool": "t1prep_thkarea_TIV",
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
    for s in ("raw", "cole", "beheshti", "zhang"):
        m = results[s]
        print(f"  {s:10s} MAE={m['mae']:6.3f}  r={m['pearson_r']:+.3f}")


if __name__ == "__main__":
    main()
