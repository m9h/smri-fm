"""Render side-by-side v1 (n=60, 23 subj) vs v2 (n=117, 42 subj) ridge MAEs.

For the meeting: shows how the cohort expansion shifted every ridge result
when going from a curated narrow-age subset to a wider lifespan cohort.
"""
import json
from pathlib import Path

R = Path(__file__).parent.parent / "results"

PAIRS = [
    # (display name, v1_stem, v2_stem)
    ("SynthSeg + TIV-norm",          "ridge_synthseg_icv",          "ridge_synthseg_tiv_v2"),
    ("SynthSeg (no TIV)",             "ridge_synthseg",              "ridge_synthseg_v2"),
    ("FastSurfer + ICV",              "ridge_fs_asegdkt_icv",        "ridge_fs_asegdkt_icv_v2"),
    ("FastSurfer (no ICV)",           "ridge_fs_asegdkt",            "ridge_fs_asegdkt_v2"),
    ("T1Prep thickness",              "ridge_t1prep_thickness",      "ridge_t1prep_thickness_v2"),
    ("T1Prep thickness + TIV",        "ridge_t1prep_thickness_TIV_real", "ridge_t1prep_thickness_TIV_v2"),
    ("T1Prep area",                   "ridge_t1prep_area",           "ridge_t1prep_area_v2"),
    ("T1Prep area + TIV",             "ridge_t1prep_area_TIV_real",  "ridge_t1prep_area_TIV_v2"),
    ("T1Prep tissue ratios + TIV",    "ridge_t1prep_tissue_icv",     "ridge_t1prep_tissue_icv_v2"),
    ("FOMO25 AMAES_resenc_b",         "ridge_fomo25_embed",          "ridge_fomo25_embed_v2"),
    ("FOMO25 mmunetvae (v2 only)",    "__none__",                    "ridge_fomo25_mmunetvae_v2"),
    # Per-N PCA rows are intentionally omitted here: the individual
    # ridge_fomo25_embed_pcaN.json files now hold v2 (42-subj) results, so a
    # v1-vs-v2 PCA comparison would mislabel them. The PCA n-vs-p story lives in
    # the dedicated ablation artefacts (ridge_fomo25_pca_ablation.json = v1,
    # ridge_fomo25_embed_pca_v2.json = v2).
    ("BrainIAC SimCLR",               "ridge_brainiac_embed",        "ridge_brainiac_embed_v2"),
    ("BrainIAC (PCA-32)",             "ridge_brainiac_embed_pca32",  "ridge_brainiac_embed_pca32_v2"),
]


def load_zhang(stem):
    p = R / f"{stem}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r["zhang"]["mae"], r["zhang"]["pearson_r"], r["raw"]["mae"], r.get("n_scans", "?"), r.get("n_subjects", "?")


def main():
    print(f"{'Tool':<32}  {'v1 (n=60, 23 subj)':<22}  {'v2 (n=117, 42 subj)':<22}  {'Δ Zhang':>8}")
    print("-" * 92)
    for name, v1_stem, v2_stem in PAIRS:
        v1 = load_zhang(v1_stem)
        v2 = load_zhang(v2_stem)
        if v1 is None and v2 is None:
            continue
        v1_str = f"raw {v1[2]:5.2f}  Z {v1[0]:5.2f} r{v1[1]:+.2f}" if v1 else "—"
        v2_str = f"raw {v2[2]:5.2f}  Z {v2[0]:5.2f} r{v2[1]:+.2f}" if v2 else "—"
        if v1 and v2:
            delta = v2[0] - v1[0]
            delta_str = f"{delta:+.2f}"
        else:
            delta_str = "—"
        print(f"{name:<32}  {v1_str:<22}  {v2_str:<22}  {delta_str:>8}")


if __name__ == "__main__":
    main()
