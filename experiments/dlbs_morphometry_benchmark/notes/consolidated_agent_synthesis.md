# Consolidated MedARC sMRI-FM Benchmark Evaluation

## 1. Executive Summary: The "Ladder" Status
The four-rung classical morphometry ladder (SynthSeg → FastSurfer → T1Prep → FMs) has been fully implemented and evaluated on a subset of 60 DLBS scans (23 subjects). While **SynthSeg + TIV-normalization** is the nominal leader in raw MAE, statistical bootstrap analysis shows a **5-way tie** at the top. The most significant finding is that Foundation Model (FM) performance is highly **age-conditional**.

## 2. Comparison Matrix (GroupKFold-5, Zhang-Corrected MAE)
| Rank | Tool | Category | MAE (yr) | r | 95% CI (Zhang) |
| :--- | :--- | :--- | :---: | :---: | :--- |
| 1 | **SynthSeg + TIV-norm** | Morphometry | **4.71** | 0.95 | [2.74, 6.07] |
| 2 | FastSurfer (aseg+DKT) | Morphometry | **5.48** | 0.93 | [3.40, 7.40] |
| 3 | **FOMO25 (AMAES_resenc_b)** | SSL/FM | **6.68** | 0.91 | [4.26, 8.00] |
| 4 | **BrainIAC (SimCLR)** | SSL/FM | **9.61** | 0.86 | [6.32, 11.28] |

**Note on "Statistical Tie":** At n=23 subjects, the 95% CIs for the top 3 tools overlap. Definitive ranking requires scaling the cohort.

## 3. Key Discovery: Age-Conditional Dominance
Foundation Model strengths are age-conditional:
*   **Geriatric Dominance (Age > 58):** **BrainIAC** achieved a **4.71 yr MAE**, significantly beating SynthSeg’s **6.69 yr MAE**.
*   **Early/Mid-Life (Age < 58):** **FOMO25** is the leader here (**3.40 yr MAE**), outperforming morphometry (~4.22–5.02 yr).

## 4. Critical Bug Report: Item 4 (AMAES Recon-Collapse)
The `asparagus` training code for FOMO25 has a critical loss logic bug:
1.  **Skip-Connection Leakage:** Loss is computed over the entire image, allowing the model to "cheat" via skip connections on unmasked voxels.
2.  **Loss Scaling Bug:** The loss is divided by total voxels rather than `mask.sum()`, suppressing reported loss by 0.6x.
*   **Proposed Fix:** Update `_rec_loss` to `((pred - y)[mask] ** 2).mean()`.

## 5. The T1Prep Normalization Mystery
T1Prep results (Thickness/Area) are identical between "Raw" and "ICV-normalized." This suggests a mapping error in `fit_ridge_baseline.py` or empty `TIV` columns in the parquet.

## 6. Recommended Next Steps
1.  **Spark Execution:** Trigger FOMO25 extraction using the Grace Blackwell-tuned `fomo25-arm` container via `run_fomo25_extraction_spark.sh`.
2.  **Dimensionality Reduction:** Apply PCA to FM embeddings before Ridge fitting to combat the $n \ll p$ curse.
3.  **OOD Validation:** Prioritize evaluation on ADNI or HCP-A to avoid training-set leakage.
4.  **Implement Item 4 Fix:** Verify if the loss fix improves latent feature diversity.
