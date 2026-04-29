# DLBS Morphometry vs SSL Benchmark — Meeting Walkthrough Script (2026-04-30)

**Estimated Time**: 5–7 minutes

---

### 1. Opening (30 sec)
*   **Framing**: "Summary of the DLBS 23-subject benchmarking sprint."
*   **Goal**: "Objective was to build a 4-rung classical morphometry ladder to benchmark FOMO25 and BrainIAC."
*   **Headline**: "Both the FOMO25 ridge run and the 'Item 4' loss-collapse investigation delivered counter-intuitive results that shift how we frame the FM value proposition."

### 2. Item 3 Result — The Performance Gap & Age Reversal (90 sec)
*   **The Ranking**: Lead with the comparison matrix. "SynthSeg + TIV-normalization is the current MAE leader (4.71 yr Zhang-corrected)."
*   **The SSL Surprise (G2)**: "Dimensionality reduction is mandatory for SSL on this cohort. PCA-16 for FOMO25 delivers **5.86 yr Zhang MAE**, tying it with SynthSeg. BrainIAC improves to **7.94 yr**."
*   **Statistical Reality**: "At n=23 subjects, the top 5 tools (SynthSeg, FastSurfer, and FOMO25+PCA) are statistically tied. Their 95% bootstrap CIs overlap. Only BrainIAC is reliably worse."
*   **The Age-Bracket Reversal**:
    *   "FOMO25 is the best tool in **both** age halves when analyzed separately (Younger 3.40, Older 3.74)."
    *   "Conclusion: SSL backbones capture age signal tighter within age-brackets; the full-cohort number (6.68) is diluted by variance across the lifespan."

### 3. Item 4 Result — Hypothesis Testing (90 sec)
*   **Framing**: "We probed the 'loss → 0' pattern reported by Dojo/Rohit against the published AMAES_resenc_b checkpoint."
*   **The Disconfirmations**: 
    *   "Hypothesis 1 (Skip-Copying): **Disconfirmed**. Visible MSE is actually 1.5× *higher* than masked MSE. The model is doing genuine MAE inference, not identity copying."
    *   "Hypothesis 2 (Instant Collapse): **Disconfirmed**. Fresh random-init training shows a normal, smooth 0.49 → 0.05 descent over 200 steps."
*   **The Confirmation**:
    *   "Hypothesis 3 (MSE Scaling Bug): **Confirmed**. The loss is divided by total voxels rather than masked voxels, suppressing reported loss by 0.6× (mask_ratio)."
    *   "Takeaway: The loss looks weird because of a minor accounting bug (H3), but the underlying representation learning is healthy."

### 4. Caveats & Rigor (60 sec)
*   **Leakage**: "DLBS is in the pretraining source for both FOMO and BrainIAC. These are not held-out generalization claims."
*   **Cohort Size**: "With n=23, Ranking margins under ~1 yr are noise. The bootstrap CIs are ~3 yr wide."
*   **Path to Finality**: "We need a non-overlapping eval. ADNI is our only path to a real 'MedARC' claim."

### 5. Open Asks & Next Steps (30 sec)
*   **Ahmed**: "Confirm if the intent was always the AMAES baseline or if we specifically need the mmunetvae weights."
*   **Dojo/Rohit**: "Which config produced the instant zero loss? (Model/Mask/Normalization?)."
*   **Access**: "Any leads on HCP-Aging or OASIS for a true held-out eval?"
