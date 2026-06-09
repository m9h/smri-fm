"""Statistical rigor on the DLBS ridge matrix.

For every existing per-tool features parquet under results/, runs the same
ridge pipeline as fit_ridge_baseline.py but adds:

1. **Bootstrap MAE confidence intervals** (B=1000 subject-level resamples)
   so we can say "FOMO25 vs BrainIAC margin = 2.93 yr [95% CI: 1.4, 4.7]"
   instead of just point estimates.

2. **Age-subgroup splits**: re-fit ridge on younger-half (< median age)
   and older-half subgroups separately. Tests whether the tool ranking
   flips by age bracket.

3. **Per-fold variance**: report MAE for each of the 5 GroupKFold folds
   so we can see which tools are stable vs noisy.

Single output: ridge_statistical_rigor.json with one entry per tool +
a comparison block showing pairwise CI overlaps.

Usage:
    python ridge_statistical_rigor.py --out results/ridge_statistical_rigor.json
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
    zhang_correction,
)


# Map tool tag → (parquet_relpath, value_col, normalise_by_icv, icv_region)
# Mirrors how the existing ridge_*.json files were generated; pulled from
# the comprehensive report.
TOOL_SPECS: list[dict] = [
    {"tool": "synthseg_volumes_tiv", "parquet_tool": "synthseg_volumes",
     "parquet": "synthseg_features.parquet", "value_col": "volume_mm3",
     "icv_norm": True, "icv_region": "total intracranial"},
    {"tool": "synthseg_volumes", "parquet_tool": "synthseg_volumes",
     "parquet": "synthseg_features.parquet", "value_col": "volume_mm3",
     "icv_norm": False, "icv_region": "total intracranial"},
    {"tool": "fs_asegdkt_icv", "parquet_tool": "aseg+DKT.VINN",
     "parquet": "fastsurfer_features.parquet", "value_col": "volume_mm3",
     "icv_norm": True, "icv_region": "total intracranial"},
    {"tool": "fs_asegdkt", "parquet_tool": "aseg+DKT.VINN",
     "parquet": "fastsurfer_features.parquet", "value_col": "volume_mm3",
     "icv_norm": False, "icv_region": "total intracranial"},
    {"tool": "brainiac_embed", "parquet_tool": "brainiac_embed",
     "parquet": "brainiac_embeddings.parquet", "value_col": "value",
     "icv_norm": False, "icv_region": ""},
    {"tool": "fomo25_embed", "parquet_tool": "fomo25_embed",
     "parquet": "/data/datasets/smri-fm-cmp/fomo-embeds/ds004856/fomo25_embeddings.parquet",
     "value_col": "value",
     "icv_norm": False, "icv_region": ""},
]


def metrics_with_corrections(y, p) -> dict:
    """Compute MAE/RMSE/r for raw + 3 bias corrections."""
    out = {}
    for label, p_corr in (
        ("raw", p),
        ("cole", cole_correction(y, p)),
        ("beheshti", beheshti_correction(y, p)),
        ("zhang", zhang_correction(y, p)),
    ):
        err = p_corr - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        r = float(np.corrcoef(y, p_corr)[0, 1]) if len(y) >= 2 else float("nan")
        out[label] = {"mae": mae, "rmse": rmse, "r": r}
    return out


def fit_ridge_kfold(X, y, groups, n_splits: int = 5):
    """Ridge with GroupKFold, returns out-of-fold predictions + per-fold MAEs."""
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    preds = np.zeros_like(y)
    fold_maes = []
    for tr, te in splitter.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
        fold_maes.append(float(np.mean(np.abs(preds[te] - y[te]))))
    return preds, fold_maes


def bootstrap_subjects(y, p, groups, B: int = 1000, seed: int = 42) -> dict:
    """Resample whole subjects with replacement; recompute MAE on each draw.

    Returns 95% CI on raw + Zhang-corrected MAE.
    """
    rng = np.random.default_rng(seed)
    unique_subs = np.unique(groups)
    raw_maes = []
    zhang_maes = []
    for _ in range(B):
        boot_subs = rng.choice(unique_subs, size=len(unique_subs), replace=True)
        idx = np.concatenate([np.where(groups == s)[0] for s in boot_subs])
        if len(idx) < 2:
            continue
        y_b, p_b = y[idx], p[idx]
        raw_maes.append(np.mean(np.abs(p_b - y_b)))
        zhang_maes.append(np.mean(np.abs(zhang_correction(y_b, p_b) - y_b)))
    raw_arr = np.array(raw_maes)
    zhang_arr = np.array(zhang_maes)
    return {
        "B": int(len(raw_maes)),
        "raw_mae_ci95": [float(np.percentile(raw_arr, 2.5)),
                         float(np.percentile(raw_arr, 97.5))],
        "raw_mae_mean": float(raw_arr.mean()),
        "zhang_mae_ci95": [float(np.percentile(zhang_arr, 2.5)),
                           float(np.percentile(zhang_arr, 97.5))],
        "zhang_mae_mean": float(zhang_arr.mean()),
    }


def load_features_for_tool(spec: dict, results_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load + ICV-normalise features for one tool. Returns X, y, groups, and feature_cols."""
    p = Path(spec["parquet"])
    if not p.is_absolute():
        p = results_dir / p
    if not p.exists():
        raise FileNotFoundError(p)

    raw = pd.read_parquet(p)
    wide = long_to_wide(raw, spec["parquet_tool"], spec["value_col"])

    ages = load_participants_age(Path("/data/raw/openneuro/ds004856/participants.tsv"))
    df = wide.merge(ages, on=["subject", "session"], how="inner")

    feature_cols = [c for c in df.columns if c not in ("subject", "session", "age")]
    if spec.get("icv_norm") and spec["icv_region"] in feature_cols:
        icv = df[spec["icv_region"]].replace(0, np.nan)
        for c in feature_cols:
            if c == spec["icv_region"]:
                continue
            df[c] = df[c] / icv
        feature_cols = [c for c in feature_cols if c != spec["icv_region"]]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["age"].to_numpy(dtype=float)
    groups = df["subject"].to_numpy()

    # NaN → column median
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])
    return X, y, groups, feature_cols


def analyze_tool(spec: dict, results_dir: Path) -> dict:
    print(f"\n=== {spec['tool']} ===")
    X, y, groups, feat_cols = load_features_for_tool(spec, results_dir)
    print(f"  n_scans={len(y)} n_subjects={len(np.unique(groups))} n_features={X.shape[1]}")

    # 1) Full GroupKFold(5) ridge — the canonical run
    preds, fold_maes = fit_ridge_kfold(X, y, groups)
    full = metrics_with_corrections(y, preds)
    print(f"  full GroupKFold5: raw_MAE={full['raw']['mae']:.3f} zhang_MAE={full['zhang']['mae']:.3f}")
    print(f"  per-fold raw MAEs: {[f'{m:.2f}' for m in fold_maes]}  std={np.std(fold_maes):.3f}")

    # 2) Bootstrap CI
    boot = bootstrap_subjects(y, preds, groups, B=1000)
    print(f"  bootstrap raw MAE 95% CI: [{boot['raw_mae_ci95'][0]:.3f}, {boot['raw_mae_ci95'][1]:.3f}]")
    print(f"  bootstrap zhang MAE 95% CI: [{boot['zhang_mae_ci95'][0]:.3f}, {boot['zhang_mae_ci95'][1]:.3f}]")

    # 3) Age-subgroup re-runs (younger-half vs older-half by median age)
    # Split BY SUBJECT median age so a subject's longitudinal scans stay together.
    sub_ages = pd.DataFrame({"sub": groups, "age": y}).groupby("sub")["age"].mean()
    median_age = sub_ages.median()
    young_subs = sub_ages[sub_ages < median_age].index.values
    old_subs = sub_ages[sub_ages >= median_age].index.values
    age_subgroups = {}
    for label, subs in (("younger", young_subs), ("older", old_subs)):
        mask = np.isin(groups, subs)
        if mask.sum() < 4 or len(np.unique(groups[mask])) < 2:
            age_subgroups[label] = None
            continue
        X_s, y_s, g_s = X[mask], y[mask], groups[mask]
        preds_s, fold_maes_s = fit_ridge_kfold(X_s, y_s, g_s, n_splits=min(5, len(np.unique(g_s))))
        m_s = metrics_with_corrections(y_s, preds_s)
        age_subgroups[label] = {
            "n_scans": int(mask.sum()),
            "n_subjects": int(len(np.unique(g_s))),
            "age_range": [float(y_s.min()), float(y_s.max())],
            "raw_mae": m_s["raw"]["mae"],
            "zhang_mae": m_s["zhang"]["mae"],
            "raw_r": m_s["raw"]["r"],
            "fold_maes": fold_maes_s,
        }
        print(f"  {label} half (age<{median_age:.0f} or age≥{median_age:.0f}): "
              f"n={mask.sum()} raw_MAE={m_s['raw']['mae']:.2f} zhang_MAE={m_s['zhang']['mae']:.2f}")

    return {
        "tool": spec["tool"],
        "n_scans": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "n_features": int(X.shape[1]),
        "icv_norm": bool(spec.get("icv_norm", False)),
        "full": full,
        "fold_maes": fold_maes,
        "fold_mae_std": float(np.std(fold_maes)),
        "bootstrap": boot,
        "age_subgroups": age_subgroups,
        "median_split_age": float(median_age),
    }


def pairwise_ci_overlap(results: list[dict]) -> list[dict]:
    """For every pair of tools, report whether their bootstrap CIs overlap.

    Non-overlapping CIs ≈ statistically distinguishable.
    """
    pairs = []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            for metric in ("raw_mae_ci95", "zhang_mae_ci95"):
                ci_a, ci_b = a["bootstrap"][metric], b["bootstrap"][metric]
                # CIs overlap iff max(lo) < min(hi)
                overlap = max(ci_a[0], ci_b[0]) < min(ci_a[1], ci_b[1])
                pairs.append({
                    "tool_a": a["tool"], "tool_b": b["tool"], "metric": metric,
                    "ci_a": ci_a, "ci_b": ci_b, "ci_overlap": bool(overlap),
                    "mean_a": a["bootstrap"][metric.replace("_ci95", "_mean")],
                    "mean_b": b["bootstrap"][metric.replace("_ci95", "_mean")],
                    "delta_mean": a["bootstrap"][metric.replace("_ci95", "_mean")] -
                                  b["bootstrap"][metric.replace("_ci95", "_mean")],
                })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/home/mhough/dev/smri-fm/experiments/dlbs_morphometry_benchmark/results"))
    args = ap.parse_args()

    results = []
    for spec in TOOL_SPECS:
        try:
            results.append(analyze_tool(spec, args.results_dir))
        except FileNotFoundError as e:
            print(f"  SKIP {spec['tool']}: {e}", file=sys.stderr)
            continue

    pairs = pairwise_ci_overlap(results)

    out = {
        "n_tools": len(results),
        "tools": results,
        "pairwise_ci_overlap": pairs,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    # Headline summary
    print("\n=== headline: zhang MAE 95% CIs ===")
    for r in sorted(results, key=lambda x: x["bootstrap"]["zhang_mae_mean"]):
        ci = r["bootstrap"]["zhang_mae_ci95"]
        print(f"  {r['tool']:30s} {r['bootstrap']['zhang_mae_mean']:5.2f} "
              f"[{ci[0]:5.2f}, {ci[1]:5.2f}]  fold std {r['fold_mae_std']:5.3f}")
    print("\n=== ranking stability check: do younger and older halves agree? ===")
    for r in results:
        y = r["age_subgroups"].get("younger")
        o = r["age_subgroups"].get("older")
        if y and o:
            print(f"  {r['tool']:30s} younger zhang={y['zhang_mae']:5.2f} | older zhang={o['zhang_mae']:5.2f}")


if __name__ == "__main__":
    main()
