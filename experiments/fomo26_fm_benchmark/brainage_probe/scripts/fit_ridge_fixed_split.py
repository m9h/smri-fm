"""Fixed-split ridge brain-age eval on FOMO26 Task 3 — comparable to the
asparagus finetune baselines (Nima: official AMAES 6.15, PDF-1M 6.58 MAE on the
50-subject TEST_80_10_10 split).

Unlike fit_ridge_baseline.py (GroupKFold-5 CV over all 494), this trains a frozen
linear probe (StandardScaler -> RidgeCV) on the official train/val subjects and
evaluates on the held-out 50 TEST subjects — the same protocol the finetune
baselines use, so the numbers sit side by side.

Consumes the same long-format parquet [subject, session, tool, region, value] and
the participants.tsv produced by fomo26_prepare.py. Reuses the bias-correction +
metric helpers from the vendored fit_ridge_baseline.py.

Usage:
  python fit_ridge_fixed_split.py \
      --features results/amaes_embeddings.parquet --tool fomo25_embed \
      --participants data/participants.tsv \
      --test-subjects data/test_subjects.json \
      --out results/ridge_amaes_fixedtest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fit_ridge_baseline import (  # noqa: E402
    beheshti_correction, cole_correction, long_to_wide,
    load_participants_age, metrics, zhang_correction,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--participants", type=Path, required=True)
    ap.add_argument("--test-subjects", type=Path, required=True)
    ap.add_argument("--normalise-by-icv", action="store_true")
    ap.add_argument("--icv-region", default="total intracranial")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wide = long_to_wide(pd.read_parquet(args.features), args.tool, args.value_col)
    ages = load_participants_age(args.participants)
    df = wide.merge(ages, on=["subject", "session"], how="inner")
    print(f"joined: {df.shape}; subjects={df['subject'].nunique()}")

    feature_cols = [c for c in df.columns if c not in ("subject", "session", "age")]
    if args.normalise_by_icv and args.icv_region in df.columns:
        icv = df[args.icv_region].replace(0, np.nan)
        for c in feature_cols:
            if c != args.icv_region:
                df[c] = df[c] / icv
        feature_cols = [c for c in feature_cols if c != args.icv_region]
    df = df.dropna(subset=feature_cols, how="any")

    test_subjects = set(json.loads(args.test_subjects.read_text()))
    is_test = df["subject"].isin(test_subjects)
    train, test = df[~is_test], df[is_test]
    if test.empty:
        raise SystemExit("no TEST subjects matched the feature parquet")
    print(f"train n={len(train)} ({train['subject'].nunique()} subj) | "
          f"test n={len(test)} ({test['subject'].nunique()} subj)")

    Xtr, ytr = train[feature_cols].to_numpy(float), train["age"].to_numpy(float)
    Xte, yte = test[feature_cols].to_numpy(float), test["age"].to_numpy(float)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)

    # bias corrections are fit on the TEST set (standard in brain-age reporting);
    # raw is the uncalibrated, directly-comparable-to-finetune number.
    out = {
        "tool": args.tool,
        "eval": "fixed_test_80_10_10",
        "n_features": len(feature_cols),
        "n_train": int(len(train)), "n_train_subjects": int(train["subject"].nunique()),
        "n_test": int(len(test)), "n_test_subjects": int(test["subject"].nunique()),
        "icv_normalised": bool(args.normalise_by_icv),
        "raw": metrics(yte, yp),
        "cole": metrics(yte, cole_correction(yte, yp)),
        "beheshti": metrics(yte, beheshti_correction(yte, yp)),
        "zhang": metrics(yte, zhang_correction(yte, yp)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    print(f"  TEST raw MAE={out['raw']['mae']:.3f}  r={out['raw']['pearson_r']:+.3f}  "
          f"zhang MAE={out['zhang']['mae']:.3f}")


if __name__ == "__main__":
    main()
