"""PCA ablation on SSL embeddings for brain-age ridge.

Tests whether dimensionality reduction before ridge closes the gap between
high-dimensional SSL features and low-dimensional morphometry on small cohorts.

Usage:
    python scripts/ridge_with_pca.py \
        --features results/brainiac_embeddings.parquet \
        --tool brainiac_embed \
        --participants /data/raw/openneuro/ds004856/participants.tsv \
        --out results/ridge_brainiac_pca_ablation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse helpers from the baseline script
sys.path.insert(0, str(Path(__file__).parent))
from fit_ridge_baseline import (
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
    ap.add_argument("--tool", type=str, required=True)
    ap.add_argument("--participants", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df_feats = pd.read_parquet(args.features)
    wide = long_to_wide(df_feats, args.tool, "value")
    
    ages = load_participants_age(args.participants)
    df = wide.merge(ages, on=["subject", "session"], how="inner")
    
    feature_cols = [c for c in df.columns if c not in ("subject", "session", "age")]
    X = df[feature_cols].to_numpy(dtype=float).copy()
    y = df["age"].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()

    # Impute NaNs
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    ks = [8, 16, 32, 64, 128, 256]
    ks = [k for k in ks if k <= X.shape[1]]
    
    results_per_k = []
    
    for k in ks:
        print(f"Fitting PCA(n_components={k}) + RidgeCV...")
        preds = np.zeros_like(y)
        splitter = GroupKFold(n_splits=5)
        
        for tr, te in splitter.split(X, y, groups):
            # Cap k by the number of training samples
            actual_k = min(k, len(tr))
            model = Pipeline([
                ("scaler1", StandardScaler()),
                ("pca", PCA(n_components=actual_k)),
                ("scaler2", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ])
            model.fit(X[tr], y[tr])
            preds[te] = model.predict(X[te])
            
        res = {
            "n_components": k,
            "raw": metrics(y, preds),
            "cole": metrics(y, cole_correction(y, preds)),
            "beheshti": metrics(y, beheshti_correction(y, preds)),
            "zhang": metrics(y, zhang_correction(y, preds)),
        }
        results_per_k.append(res)
        
        # Save individual result in the schema make_meeting_summary.py expects
        # results/ridge_{tool}_pca{k}.json
        indiv_res = {
            "tool": f"{args.tool}_pca{k}",
            "value_col": "value",
            "n_features": k,
            "n_scans": int(len(y)),
            "n_subjects": int(df["subject"].nunique()),
            "cv_mode": "groupkfold5",
            **res
        }
        del indiv_res["n_components"]
        out_path = args.out.parent / f"ridge_{args.tool}_pca{k}.json"
        out_path.write_text(json.dumps(indiv_res, indent=2))

    # Save summary JSON
    args.out.write_text(json.dumps(results_per_k, indent=2))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    maes_raw = [r["raw"]["mae"] for r in results_per_k]
    maes_zhang = [r["zhang"]["mae"] for r in results_per_k]
    plt.plot(ks, maes_raw, "o-", label="Raw MAE")
    plt.plot(ks, maes_zhang, "s-", label="Zhang-corrected MAE")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("MAE (years)")
    plt.title(f"PCA Ablation: {args.tool}")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.out.with_suffix(".png"))
    print(f"Wrote summary to {args.out} and plot to {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
