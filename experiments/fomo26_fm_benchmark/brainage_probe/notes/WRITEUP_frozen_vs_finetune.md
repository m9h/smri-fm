# Foundation models for sMRI: frozen probes mis-rank what finetuning rewards

*A two-part study on FOMO26-adjacent tasks. Companion to the team's finetune
baselines; all code on `fomo26-siam-bridge`, results in
`experiments/fomo26_fm_benchmark/brainage_probe/`.*

## TL;DR

1. **Frozen linear-probe rankings do not predict finetuned performance**
   (Spearman ρ ≈ 0.31 across 6 backbones, n.s.). The frozen-probe *winner* isn't the
   seg winner; the frozen-probe *loser* (SIAM) is a seg winner; a strong-frozen arm
   (SimCLR3D) is mediocre finetuned.
2. **The apparent "+0.07–0.10 pretraining gain" on finetuned ISLES22 seg is a *decoder*
   effect, not encoder pretraining.** Hold the decoder byte-identical across arms (a
   shared body fed via 1×1 adapters) and compare each FM encoder against a *random-init
   encoder of the same architecture*: with a **light** U-Net body the gain collapses to
   ≤ +0.03 (noise/negative for 3 of 4 backbones); with a **high-capacity SwinUNETR**
   body it **vanishes entirely** — a *random-init* Swin encoder (0.767) is the **best
   arm overall**, tying/beating every pretrained FM on the same decoder.
3. **The decoder is worth ~4–5× more than the encoder ever was.** Swapping the shared
   body light→SwinUNETR adds **+0.11–0.16 Dice to every arm, random-init included**; the
   largest encoder-pretraining effect we ever measured was +0.033, and it disappears once
   the decoder is capable. fomo60k's lone positive signal (+0.033 on the light decoder →
   −0.006 on SwinUNETR) was the encoder *compensating for a weak decoder*, not a
   transferable advantage. (Sanity-checked: our shared SwinUNETR decoder reproduces the
   native SwinUNETR result, 0.761 ≈ 0.754.)
4. **Pretraining is not free insurance:** Triad's pretrained Swin finetunes *below* random
   init on both shared decoders. A foundation model is not automatically better than random.

**Practical message: don't select or reject an sMRI foundation model on a frozen linear
probe; and on this task, the decoder you bolt on and the data budget you train under
dominate — encoder pretraining buys ≈0 at 200 cases once the decoder is capable. The
open question pretraining is actually for is the low-data regime (a label-budget sweep is
the next experiment).**

## Why these two tasks

The team's brain-age (REGR002) baselines are T1-only — a regime where classical
morphometry is already strong and the FMs' multimodal/diffusion pretraining is barely
exercised. So we ran two complementary probes:

- **Brain-age frozen linear probe + a morphometry floor** (cheap, label-light) — to
  read frozen representation quality and compare against a classical baseline.
- **ISLES22 stroke-lesion segmentation, finetuned** — the *modality-matched* task
  (DWI/ADC, the plurality modality in FOMO300K pretraining), dense prediction (what
  these models are built for), with a from-scratch control.

## Part 1 — Brain-age frozen probe (DLBS REGR002, T1, ridge on frozen embeddings)

12-arm roster, GroupKFold-5 over 494 scans (the load-bearing column; the 50-subject
official TEST is too small to rank). Each FM extracted at its native input size.

- **Only 2 of the FMs beat the FastSurfer aseg-volume floor (CV 6.41 MAE) frozen:**
  fomo60k-regular (**6.05**) and simclr3d (**6.23**). The flagship FOMO models *lost*:
  AMAES 6.89, the FOMO25-winner mmunetvae 8.26; SIAM was worst (10.01, ~⅓ of its
  embedding channels are dead post-ReLU).
- Frozen-feature quality spans **4.4 years** of MAE across the roster.

*Caveats:* this is a linear probe, not the team's finetune protocol; frozen-vs-finetune
isn't a pixel-identical pipeline; and **DLBS is in FOMO pretraining** (verified in the
manifest), so brain-age carries a memorization confound. Treat Part 1 as exploratory
context, not a headline.

## Part 2 — ISLES22 stroke-lesion segmentation (finetuned, the controlled result)

ISLES22 (Zenodo 7153326, CC-BY): 250 cases, DWI/ADC/FLAIR + expert lesion masks; one of
the FOMO260K paper's own held-out validation tasks. We added it as a new asparagus seg
task (`SEG011_ISLES22_IschStroke`, DWI+ADC — co-registered with the masks; FLAIR is
off-grid). **Leakage-verified clean:** grepped both the FOMO260K (public) and FOMO300K
(gated) `mri_info.tsv` manifests — ISLES22 is absent from all 910/915 source datasets.
5-fold, 200-epoch finetune; from-scratch nnU-Net control; run on Modal (A100).

**5-fold lesion Dice (mean ± std):**

| Arm | Backbone / decoder | Dice | vs scratch |
|---|---|---|---|
| triad | Swin-MAE / SwinUNETR | **0.763 ± 0.012** | +0.10 |
| fomo60k | Swin-MAE / SwinUNETR | 0.754 ± 0.047 | +0.09 |
| siam | nnU-Net / nnU-Net | 0.740 ± 0.015 | +0.07 |
| mmunetvae | UNet-VAE / native | 0.734 ± 0.033 | +0.07 |
| simclr3d | ResNet-18 / ResNet-UNet | 0.678 ± 0.025 | +0.01 |
| **scratch** | nnU-Net (random init) | **0.667 ± 0.021** | — |
| anatcl | ResNet-18 / ResNet-UNet | 0.617 ± 0.010 | **−0.05** |

(brainiac deferred: its ViT pretrained at 96³ + UNETR's fixed input size conflict with
the 128³ sliding-window inference.)

**Findings:**
- **Pretraining helps the strong backbones** (+0.07–0.10), but **not universally** —
  AnatCL finetunes to below scratch.
- **Frozen↔finetune ranking is decoupled** (Spearman ρ ≈ 0.31, n=6, n.s.):
  - **SIAM** — *worst* frozen (dead features) — is **3rd on seg and beats scratch +0.07
    on the *same* nnU-Net decoder** (a decoder-confound-free comparison).
  - **SimCLR3D** — *2nd-best* frozen — only ties scratch.
- Transformer-MAE backbones (triad, fomo60k) are the most robust across both tasks.

## Part 3 — Uniform decoder: the +0.07 gain is mostly the decoder, not the encoder

Part 2's cross-arm ranking has a confound it flagged itself: each arm carried a
*different* decoder (SwinUNETR for the Swin arms, our ResNet-UNet for the ResNet arms,
nnU-Net for SIAM/scratch). So a "+0.07 over scratch" mixes the encoder *and* the
decoder *and* a from-scratch baseline that used yet another topology (nnU-Net). To
isolate the **encoder**, we re-ran the 4 encoder-only arms through a **byte-identical
decoder**: each encoder's 5-level pyramid (/2…/32 — Swin and ResNet-18 both produce
this) → thin 1×1 channel adapters → the *same* U-Net decoder body. Only the adapters
(minimal 1×1 capacity) differ by arm. We added two **decoder-controlled scratch
floors**: a random-init Swin and a random-init ResNet-18, each through that same
decoder — so FM-vs-scratch differs *only* in pretrained-vs-random encoder weights.

**Uniform-decoder 5-fold lesion Dice (mean ± std), all on the identical decoder body:**

| Arm | Encoder | Uniform Dice | Native-decoder Dice (Part 2) |
|---|---|---|---|
| fomo60k | Swin-MAE | **0.641 ± 0.024** | 0.754 (SwinUNETR) |
| simclr3d | ResNet-18 | 0.632 ± 0.047 | 0.678 (ResNet-UNet) |
| anatcl | ResNet-18 | 0.625 ± 0.011 | 0.617 (ResNet-UNet) |
| **scratch (ResNet-18)** | random | **0.620 ± 0.007** | — |
| **scratch (Swin)** | random | **0.608 ± 0.019** | — |
| triad | Swin-MAE | 0.590 ± 0.018 | 0.763 (SwinUNETR) |

**Decoder-controlled pretraining gain** (paired over folds, same arch + same decoder,
pretrained vs random encoder):

| FM arm | vs scratch | Δ Dice | folds + |
|---|---|---|---|
| fomo60k | Swin | **+0.033** | 4/5 |
| triad | Swin | **−0.018** | 1/5 |
| simclr3d | ResNet-18 | +0.012 | 2/5 |
| anatcl | ResNet-18 | +0.004 | 3/5 |

**Findings:**
- **The cross-arm spread collapses ~0.15 → ~0.05** and the **ranking reorders**: Triad
  goes from *best* with SwinUNETR (0.763) to *worst* on the shared decoder (0.590,
  below its own scratch); AnatCL goes from below-scratch to mid-pack. The Swin arms
  shed 0.11–0.17 when they lose SwinUNETR; the ResNet arms (whose native decoder
  already resembled the uniform one) barely move. **Most of the Part-2 cross-arm signal
  was the decoder.**
- **Once the decoder is fixed, encoder pretraining buys ≤ +0.03 Dice.** Only **fomo60k**
  shows a real (if modest) gain (+0.033, 4/5 folds). SimCLR3D/AnatCL are within noise
  (+0.012 / +0.004); **Triad is negative** (−0.018, 1/5). The Part-2 "+0.07–0.10" was
  inflated by the decoder swap *and* a weak nnU-Net-topology scratch (0.667) — the
  matched-arch random-init scratch here is **0.61–0.62**, much closer to the FMs.
- **fomo60k is the one backbone that transfers on both probes** — frozen brain-age
  champion *and* the only positive decoder-controlled seg gain. Triad is the mirror
  image: SwinUNETR winner but frozen-mediocre and decoder-controlled-negative — its
  strength was the encoder+decoder *pairing*, not the encoder.

*Caveat (resolved in Part 4):* the light uniform decoder is lower-capacity than SwinUNETR
(Swin absolute Dice drops 0.75 → 0.59–0.64), so one could argue "a weak shared decoder
can't exploit a strong encoder, masking its advantage." **Part 4 tests exactly this with a
high-capacity SwinUNETR shared body — and the gain doesn't grow, it vanishes** (a random
encoder becomes the top arm). So the small light-decoder gain wasn't a masked advantage;
it was the encoder compensating for a weak decoder.

## Part 4 — A high-capacity shared decoder erases the gain (and a random encoder wins)

Part 3's light U-Net body is lighter than SwinUNETR, leaving the reading "a weak shared
decoder can't exploit a strong encoder, masking its advantage." So we ran the *same*
controlled experiment with a **high-capacity shared decoder**: the exact MONAI SwinUNETR
decoder blocks (the body that gave the Swin arms ~0.75 natively), held byte-identical at
feature_size 48, fed each encoder's pyramid through 1×1 adapters (`SwinUnetrSharedDecoder`).
Same 4 FM encoders + 2 matched-arch random-init scratch floors, same 5-fold ISLES22.

**5-fold lesion Dice — SwinUNETR shared body (and the light body, for contrast):**

| Arm | Encoder | **SwinUNETR body** | Light body (Part 3) |
|---|---|---|---|
| **scratch (Swin, random)** | random | **0.767 ± 0.017** | 0.608 |
| fomo60k | Swin-MAE | 0.761 ± 0.026 | 0.641 |
| triad | Swin-MAE | 0.746 ± 0.022 | 0.590 |
| anatcl | ResNet-18 | 0.738 ± 0.021 | 0.625 |
| scratch (ResNet, random) | random | 0.731 ± 0.026 | 0.620 |
| simclr3d* | ResNet-18 | 0.659 ± 0.066 | 0.632 |

**Decoder-controlled pretraining gain (paired) — light vs SwinUNETR body:**

| Arm | Light gain | **SwinUNETR gain** |
|---|---|---|
| fomo60k (Swin) | +0.033 (4/5) | **−0.006 (3/5)** |
| triad (Swin) | −0.018 (1/5) | −0.021 (0/5) |
| anatcl (ResNet) | +0.004 (3/5) | +0.007 (3/5) |
| simclr3d (ResNet)* | +0.012 (2/5) | −0.071 (0/5)* |

**Findings:**
- **The decoder swap is worth +0.11–0.16 Dice for *every* arm — random-init included**
  (scratch-Swin +0.159, triad +0.156, fomo60k +0.121). That is **~4–5× the largest
  encoder-pretraining effect ever measured here** (+0.033). On this task the decoder is
  the dominant factor by a wide margin; the encoder-pretraining question is a rounding
  error next to it.
- **With a capable decoder, encoder pretraining buys ≈0.** A *random-init* Swin encoder
  (0.767) is the **top arm**, statistically tied with pretrained fomo60k (0.761) on the
  identical decoder. fomo60k's lone light-decoder gain (+0.033) **flips to −0.006**. The
  clean ResNet arm (anatcl, +0.007) is the same noise it was on the light body. So the
  small light-decoder gain was the pretrained encoder *compensating for a weak decoder* —
  give the decoder enough capacity and the compensation isn't needed.
- **Sanity check passes:** fomo60k on our *shared* SwinUNETR decoder (0.761) ≈ fomo60k on
  its *native* SwinUNETR (Part 2: 0.754). The apparatus reproduces the native pairing, so
  "random Swin ties pretrained Swin" is a real result, not a wiring artifact.

*simclr3d is caveated: its MONAI-resnet pyramid sits off the canonical /2…/32 octaves, so
the rigid SwinUNETR cats required a resize (it gained only +0.028 from the decoder and its
−0.071 partly reflects that distortion). The clean ResNet arm is anatcl. The headline rests
on the clean arms (fomo60k, anatcl, the two scratch floors).*

**What Parts 3+4 jointly establish:** the finetuned ISLES22 ranking is governed by the
decoder and the data budget, not by whether the encoder was pretrained. At 200 cases with
a capable decoder, a random-init encoder matches the best foundation model. That does *not*
mean pretraining is worthless — it means its value, if any, lives in the **low-data regime**
this 200-case full-finetune doesn't probe. The decisive next experiment is a label-budget
sweep (5/10/20/50/200 cases): does the pretrained-vs-random gap that is ≈0 at 200 reopen
few-shot? That is the question this whole controlled apparatus was built to ask cleanly.

## The synthesis

Frozen linear-probe rank and finetuned-seg rank don't track each other. Finetuning
*rescues* some backbones the frozen probe writes off (SIAM) and fails to rescue others
(AnatCL); a backbone that probes well frozen (SimCLR3D) can underwhelm finetuned. So a
frozen probe is a poor model-selection signal for any task you intend to finetune.

And the finetuned cross-arm ranking itself is a **decoder** ranking: hold the decoder
fixed and the gaps reshuffle and shrink (light body) or vanish (SwinUNETR body), where a
*random-init* encoder is the top arm. The decoder swap is worth +0.11–0.16 Dice — several
times any encoder-pretraining effect. The honest bottom line for sMRI FMs on this task:
**at 200 cases, encoder pretraining buys ≈0 once the decoder is capable — the result is
set by the decoder you bolt on and the data budget you train under, not by whether the
encoder was pretrained.** The legitimate case for pretraining is the low-data regime this
study doesn't probe; a label-budget sweep is the experiment that would actually adjudicate it.

## Related work

A concurrent cost-utility study — Pang et al., *How Much MRI Preprocessing Is Enough?
A Cost-Utility Study for Brain MRI Foundation Models* (arXiv:2606.08164, Jun 2026) —
reaches a convergent conclusion from an orthogonal axis. They pretrain 3D ViT-Base MAE
and JEPA models on 20k volumes drawn from **FOMO300K — the same corpus our backbones come
from** — across a graded **P0–P7 preprocessing spectrum**, and ask how much downstream
performance the extra preprocessing buys. Their answer: beyond a "minimum viable" P2
(orientation + resample + percentile-clip z-score), replacing P2 with the *best* feasible
level lifts aggregate utility by only **+3.4 pp (MAE) / +1.8 pp (JEPA)**, and of 24
P2-vs-best paired comparisons **only one survived multiple-testing correction**. The
benefit is task-dependent — MCI classification gains most from heavy preprocessing, while
**brain-age regression and tumor segmentation are already near-optimal at P2** — and even
the MCI gain is largely recoverable by applying the heavier preprocessing *only at
downstream time* (a P2-pretrained checkpoint on P7 data recovers ~69–81 % of the gap).

Their study and ours are the **same move on different variables**. They hold the model
fixed and vary *preprocessing*; we hold encoder/decoder/architecture fixed and vary *what
is pretrained*. Both isolate a confound that ordinarily inflates an MRI FM's apparent
value, and both find the residual benefit small and regime-dependent — their few-percent
aggregate lift is the same order of magnitude as our ≤ +0.03 decoder-controlled Dice, and
their "downstream-time preprocessing recovers most of the gap" parallels our "finetuning
rescues a backbone the frozen probe wrote off." Three concrete connections:

1. **It bounds a confound we don't control.** We don't hold *preprocessing alignment*
   fixed (each backbone was pretrained under its own preprocessing; we finetune under
   asparagus's). But because Pang et al. share our corpus and find **segmentation and age
   regression near-optimal at minimal preprocessing**, the minimum-preprocessing pipeline
   we use is unlikely to be suppressing our gains — preprocessing alignment is a *smaller*
   confound for our tasks (seg, age) than for theirs (MCI). That converts an open caveat
   into a bounded one.
2. **Their statistical discipline is the bar we should adopt.** Paired CIs +
   multiple-testing correction (1/24 surviving) is exactly the rigor our n=5-fold
   cross-arm claims need before any "FM A > FM B" assertion; our ρ≈0.31 (n.s.) and
   per-fold paired deltas are in that spirit but should be reported with corrected CIs.
3. **The axes compose.** A full cost-utility map of sMRI FMs varies *preprocessing ×
   decoder/architecture × label budget* together. Their paper covers the first axis, this
   report the second; the third (a label-budget sweep) is our proposed next experiment.
   Together they make the same argument — that headline FM "wins" on brain MRI are
   dominated by axes evaluators rarely hold fixed — from independent directions.

## Limitations (read before citing)

- **Cross-arm seg ranking mixes encoder + decoder** — *now quantified in Part 3.* The
  Part-2 decoders differ by arm (SwinUNETR / ResNet-UNet / nnU-Net); the uniform-decoder
  rerun shows this confound was *large* (spread 0.15 → 0.05, ranking reorders). The one
  Part-2 claim that survived without a uniform decoder: SIAM is worst-frozen yet +0.07
  over scratch **on the identical nnU-Net decoder** — no decoder difference explains it.
- ~~The uniform decoder is lighter than SwinUNETR~~ **— tested in Part 4.** The
  high-capacity SwinUNETR shared body does *not* recover a pretraining advantage; it
  erases it (random encoder wins). So the light-body gains weren't understated.
- **simclr3d's pyramid is off the canonical octaves** for the rigid SwinUNETR cats, so its
  Part-4 number required a resize and is the one untrustworthy sunet cell; anatcl is the
  clean ResNet arm. (The light-body Part-3 simclr3d number is unaffected.)
- **All of this is at 200-case full finetune.** Pretraining's classic payoff is low-data;
  the ≈0 gain here doesn't speak to few-shot. A label-budget sweep is required before any
  "pretraining doesn't help" claim generalizes beyond this regime.
- Our Part-2 from-scratch baseline (0.667) is weaker than the FOMO paper's few-shot
  scratch (0.729) — different topology/budget/2-vs-3 modalities. Part 3's matched-arch
  scratch (0.61–0.62 on the shared decoder) is the internally-controlled floor; the
  Part-2 "+0.07" is "vs *that* weaker nnU-Net scratch," not the paper's number.
- n=25 test/fold; 5-fold gives the error bars. ρ=0.31 at n=6 isn't significant on its
  own, but the per-arm dissociations (SIAM, SimCLR3D) are individually clear.
- **Preprocessing alignment is not held fixed.** Each backbone was pretrained under its
  own preprocessing; we finetune under one asparagus pipeline. Pang et al. (2026,
  arXiv:2606.08164) show this matters little for *seg / age* at minimal preprocessing
  (their P2-optimality) but materially for MCI-type tasks — so it's a bounded confound
  here, not a free pass (see Related work).

## Reusable for the team

- **`SEG011_ISLES22_IschStroke`** — a new working seg task; the team's seg leaderboard
  was meningioma 0.0 / trigeminal 0.18 (both failures), this one reaches ~0.74.
- **A SIAM seg-inference bug fix** — SIAM's anisotropic `3d_fullres` plan (stride
  64×64×32) crashed sliding-window inference; fixed in `seg_inference.py` (pad tiles to
  mult-of-64). Helps any SIAM seg run.
- **Bolt-on seg decoders** for the encoder-only FM arms (SwinUNETR re-parent for Swin,
  ResNet-UNet for ResNet) — `models_smri_*.py` + `seg_decoders.py`, wired so asparagus's
  encoder/decoder LR split works.
- **Two shared decoders, one switch** (`seg_decoders.py`): `UniformUNetDecoder` (light)
  and `SwinUnetrSharedDecoder` (high-capacity), selected per arm via `decoder_kind`. Any
  encoder's 5-level pyramid → 1×1 adapters → a byte-identical body, so seg comparisons
  isolate the *encoder*. `*_useg.yaml` / `*_sunet.yaml` configs + `scratch_{swin,resnet}_*`
  random-init floors. This apparatus turned a "+0.07 pretraining win" into "≤+0.03 on a
  light decoder, ≈0 on a strong one (random encoder wins)."
- **Modal seg sweep** (`scripts/modal_seg_sweep.py`) — 5-fold × roster in ~2 h vs ~30 h
  on one GPU, with a `--debug` smoke for cheap red-green validation and `--out` for
  race-free parallel runs.

## Next steps

1. **Label-budget sweep (5/10/20/50/200 cases)** — *the decisive experiment.* Parts 3+4
   show pretrained ≈ random at 200 cases; pretraining's classic payoff is low-data. Does
   the gap reopen few-shot? The shared-decoder apparatus already isolates the encoder, so
   this is a subsample loop on the existing sweep. Completes the cost-utility map
   (preprocessing [Pang et al.] × decoder [Parts 3–4] × **label budget**).
2. **Corrected statistics** — paired CIs + multiple-testing correction on all cross-arm
   deltas (matching Pang et al.'s 1/24-survive rigor) before any "A > B" claim ships.
3. ~~Higher-capacity shared decoder~~ **— done (Part 4):** the gain vanishes, not grows.
4. Add **brainiac** (UNETR at 96³ with a matched patch size) for completeness.
5. **3-modality** ISLES22 (resample FLAIR to DWI) for the exact paper protocol.
