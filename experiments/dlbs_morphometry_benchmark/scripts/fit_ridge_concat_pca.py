"""Concat-then-PCA ridge for the FS+T1Prep+BrainIAC mega-feature stack.

Goal: test whether dimensionality reduction rescues the 939-feature
n ≪ p concat case, mirroring gemini-agent's PCA finding for the
SSL-only embeddings.

Builds the same concat as ridge_concat_fs_t1prep_brainiac.json:
  FastSurfer aseg+DKT.VINN (100) + T1Prep thickness (71) + BrainIAC (768)
                                                          = 939 features

Then runs ridge with PCA(n_components=k) for k ∈ {8, 16, 32, 48} —
capped at GroupKFold(5) train fold size of 48 (60 × 4/5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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
    ap.add_argument("--participants", type=Path,
                    default=Path("/data/raw/openneuro/ds004856/participants.tsv"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/home/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/results"))
    args = ap.parse_args()

    R = args.out_dir
    fs = pd.read_parquet(R / "fastsurfer_features.parquet")
    tp = pd.read_parquet(R / "t1prep_features.parquet")
    bi = pd.read_parquet(R / "brainiac_embeddings.parquet")

    fs_w = long_to_wide(fs, "aseg+DKT.VINN", "volume_mm3")
    tp_w = long_to_wide(tp, "t1prep_thickness", "thickness_mm")
    bi_w = long_to_wide(bi, "brainiac_embed", "value")

    # tag columns to avoid collisions
    for w, tag in ((fs_w, "fs"), (tp_w, "thk"), (bi_w, "bi")):
        w.columns = [
            c if c in ("subject", "session") else f"{tag}__{c}"
            for c in w.columns
        ]

    df = fs_w.merge(tp_w, on=["subject", "session"], how="inner") \
             .merge(bi_w, on=["subject", "session"], how="inner")
    print(f"merged shape: {df.shape}")

    ages = load_participants_age(args.participants)
    df = df.merge(ages, on=["subject", "session"], how="inner")
    feat_cols = [c for c in df.columns if c not in ("subject", "session", "age")]

    X = df[feat_cols].to_numpy(dtype=float).copy()
    y = df["age"].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()

    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    print(f"X shape: {X.shape}, n_subjects: {df['subject'].nunique()}")

    splitter = GroupKFold(n_splits=5)
    train_fold_size = int(np.median([len(tr) for tr, te in splitter.split(X, y, groups)]))
    print(f"median train-fold size: {train_fold_size}")

    ks = [k for k in (8, 16, 32, 48) if k <= train_fold_size]
    print(f"PCA components to test: {ks}")

    for k in ks:
        preds = np.zeros_like(y)
        for tr, te in splitter.split(X, y, groups):
            actual_k = min(k, len(tr) - 1)  # PCA caps at min(n_features, n_samples)
            model = Pipeline([
                ("scaler1", StandardScaler()),
                ("pca", PCA(n_components=actual_k)),
                ("scaler2", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ])
            model.fit(X[tr], y[tr])
            preds[te] = model.predict(X[te])

        results = {
            "tool": f"concat_fs_t1prep_brainiac_pca{k}",
            "n_features": int(k),
            "raw_n_features": int(X.shape[1]),
            "n_scans": int(len(y)),
            "n_subjects": int(df["subject"].nunique()),
            "cv_mode": "groupkfold5",
            "raw":      metrics(y, preds),
            "cole":     metrics(y, cole_correction(y, preds)),
            "beheshti": metrics(y, beheshti_correction(y, preds)),
            "zhang":    metrics(y, zhang_correction(y, preds)),
        }
        out = R / f"ridge_concat_fs_t1prep_brainiac_pca{k}.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"  PCA-{k:3d}  raw_MAE={results['raw']['mae']:6.3f}  "
              f"Zhang={results['zhang']['mae']:6.3f}  r_Z={results['zhang']['pearson_r']:+.3f}  → {out.name}")

    print()
    # Also report the no-PCA baseline for comparison
    print("baseline (no PCA, 939 feats): raw_MAE=8.31  Zhang=7.35  r_Z=+0.91  (from ridge_concat_fs_t1prep_brainiac.json)")


if __name__ == "__main__":
    main()
