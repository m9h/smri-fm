"""Concat ridge with per-source TIV-normalization for volumetric tools.

Mirrors fit_ridge_concat.py but applies the same TIV-norm fix that
fit_ridge_t1prep_with_tiv.py uses for cross-tool TIV joins:

  - Volumetric sources (FS aseg+DKT, SynthSeg volumes) get divided by
    their own intra-tool TIV / "total intracranial" column before join.
  - Non-volumetric (T1Prep thickness, BrainIAC, FOMO25) pass through.
  - All sources merge on (subject, session), then ridge with
    StandardScaler + RidgeCV under GroupKFold(5).

Three configs run:
  fs_t1prep        — FS aseg+DKT (TIV-norm) + T1Prep thickness
  synthseg_t1prep  — SynthSeg vols (TIV-norm) + T1Prep thickness
  fs_t1prep_brainiac — FS (TIV-norm) + T1Prep thickness + BrainIAC

For the 939-feat row we ALSO run a PCA-48 variant with TIV-norm
applied to FS before PCA, matching the PCA-48 finding from
fit_ridge_concat_pca.py.
"""
from __future__ import annotations

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

R = Path("/home/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/results")
PARTICIPANTS = Path("/data/raw/openneuro/ds004856/participants.tsv")


def tiv_normed_wide(parquet: Path, tool: str, value_col: str,
                    icv_region: str = "total intracranial") -> pd.DataFrame:
    """Pivot to wide and divide all features by the icv_region column."""
    df = pd.read_parquet(parquet)
    wide = long_to_wide(df, tool, value_col)
    if icv_region not in wide.columns:
        raise KeyError(f"{icv_region} not in {tool}'s pivoted columns "
                       f"(have {list(wide.columns)[:5]}...)")
    icv = wide[icv_region].replace(0, np.nan)
    feat = [c for c in wide.columns if c not in ("subject", "session", icv_region)]
    out = wide[["subject", "session"]].copy()
    for c in feat:
        out[c] = wide[c] / icv
    return out


def passthrough_wide(parquet: Path, tool: str, value_col: str) -> pd.DataFrame:
    return long_to_wide(pd.read_parquet(parquet), tool, value_col)


def tag_columns(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c if c in ("subject", "session") else f"{tag}__{c}"
        for c in df.columns
    ]
    return df


def run_ridge(X, y, groups, with_pca: int | None = None):
    splitter = GroupKFold(n_splits=5)
    preds = np.zeros_like(y)
    for tr, te in splitter.split(X, y, groups):
        steps = [("scaler1", StandardScaler())]
        if with_pca is not None:
            actual_k = min(with_pca, len(tr) - 1)
            steps.append(("pca", PCA(n_components=actual_k)))
            steps.append(("scaler2", StandardScaler()))
        steps.append(("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))))
        model = Pipeline(steps)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    return preds


def write_result(name: str, preds, y, groups, n_features: int, raw_n_features: int,
                 with_pca: int | None, tags: list[str]):
    results = {
        "tool": name,
        "n_features": int(n_features),
        "raw_n_features": int(raw_n_features),
        "n_scans": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "cv_mode": "groupkfold5",
        "tiv_normalised_sources": tags,
        "pca": with_pca,
        "raw":      metrics(y, preds),
        "cole":     metrics(y, cole_correction(y, preds)),
        "beheshti": metrics(y, beheshti_correction(y, preds)),
        "zhang":    metrics(y, zhang_correction(y, preds)),
    }
    out = R / f"ridge_{name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  {name:60s}  raw={results['raw']['mae']:6.3f}  "
          f"Zhang={results['zhang']['mae']:6.3f}  r_Z={results['zhang']['pearson_r']:+.3f}")


def main():
    ages = load_participants_age(PARTICIPANTS)

    # Source loaders (lazy until each run uses them)
    fs_tiv = tag_columns(tiv_normed_wide(R / "fastsurfer_features.parquet",
                                          "aseg+DKT.VINN", "volume_mm3"), "fs")
    synthseg_tiv = tag_columns(tiv_normed_wide(R / "synthseg_features.parquet",
                                                "synthseg_volumes", "volume_mm3"), "ss")
    t1prep_thk = tag_columns(passthrough_wide(R / "t1prep_features.parquet",
                                               "t1prep_thickness", "thickness_mm"), "thk")
    brainiac = tag_columns(passthrough_wide(R / "brainiac_embeddings.parquet",
                                             "brainiac_embed", "value"), "bi")

    print(f"fs_tiv shape: {fs_tiv.shape}")
    print(f"synthseg_tiv shape: {synthseg_tiv.shape}")
    print(f"t1prep_thk shape: {t1prep_thk.shape}")
    print(f"brainiac shape: {brainiac.shape}")

    def assemble(parts):
        df = parts[0]
        for p in parts[1:]:
            df = df.merge(p, on=["subject", "session"], how="inner")
        df = df.merge(ages, on=["subject", "session"], how="inner")
        feats = [c for c in df.columns if c not in ("subject", "session", "age")]
        X = df[feats].to_numpy(dtype=float).copy()
        y = df["age"].to_numpy(dtype=float)
        groups = df["subject"].to_numpy()
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
        return X, y, groups

    print("\n=== concat + per-source TIV-norm ===")

    # 1. FS (TIV) + T1Prep thickness
    X, y, groups = assemble([fs_tiv, t1prep_thk])
    preds = run_ridge(X, y, groups)
    write_result("concat_fs_TIV_t1prep_thk", preds, y, groups,
                 n_features=X.shape[1], raw_n_features=X.shape[1],
                 with_pca=None, tags=["fs:TIV-normed", "t1prep_thickness:passthrough"])

    # 2. SynthSeg (TIV) + T1Prep thickness
    X, y, groups = assemble([synthseg_tiv, t1prep_thk])
    preds = run_ridge(X, y, groups)
    write_result("concat_synthseg_TIV_t1prep_thk", preds, y, groups,
                 n_features=X.shape[1], raw_n_features=X.shape[1],
                 with_pca=None, tags=["synthseg:TIV-normed", "t1prep_thickness:passthrough"])

    # 3. FS (TIV) + T1Prep thickness + BrainIAC (no PCA)
    X, y, groups = assemble([fs_tiv, t1prep_thk, brainiac])
    preds = run_ridge(X, y, groups)
    write_result("concat_fs_TIV_t1prep_brainiac", preds, y, groups,
                 n_features=X.shape[1], raw_n_features=X.shape[1],
                 with_pca=None, tags=["fs:TIV-normed", "t1prep_thickness:passthrough",
                                       "brainiac:passthrough"])

    # 4. Same + PCA-48
    preds = run_ridge(X, y, groups, with_pca=48)
    write_result("concat_fs_TIV_t1prep_brainiac_pca48", preds, y, groups,
                 n_features=48, raw_n_features=X.shape[1],
                 with_pca=48, tags=["fs:TIV-normed", "t1prep_thickness:passthrough",
                                     "brainiac:passthrough"])


if __name__ == "__main__":
    main()
