# Foundation models for sMRI: frozen probes mis-rank what finetuning rewards

*A two-part study on FOMO26-adjacent tasks. Companion to the team's finetune
baselines; all code on `fomo26-siam-bridge`, results in
`experiments/fomo26_fm_benchmark/brainage_probe/`.*

## TL;DR

1. **Frozen linear-probe rankings do not predict finetuned performance**
   (Spearman ρ ≈ 0.31 across 6 backbones, n.s.). The frozen-probe *winner* isn't the
   seg winner; the frozen-probe *loser* (SIAM) is a seg winner; a strong-frozen arm
   (SimCLR3D) is mediocre finetuned.
2. **The apparent "+0.07–0.10 pretraining gain" on finetuned ISLES22 seg is largely a
   *decoder* artifact, not encoder pretraining.** When we hold the decoder
   byte-identical across arms (a shared U-Net decoder body fed via 1×1 adapters) and
   compare each FM encoder against a *random-init encoder of the same architecture*,
   the gain collapses to **≤ +0.03 Dice — and is within noise or negative for 3 of 4
   backbones.** The cross-arm seg spread shrinks from ~0.15 to ~0.05, and the ranking
   *reorders* (Triad, the SwinUNETR winner, drops to worst, below its own scratch).
3. **Only one backbone (fomo60k) shows a real, if modest, decoder-controlled gain
   (+0.033, 4/5 folds)** — and it's also the frozen brain-age champion. The rest are
   decoder-dependent: their finetuned lead was the decoder pairing, not the encoder.
4. **Pretraining is not free insurance:** even with matched decoders, Triad's
   pretrained Swin finetunes *below* random init. A foundation model is not
   automatically better than random.

**Practical message: don't select or reject an sMRI foundation model on a frozen
linear probe — and don't read a finetuned cross-arm seg gap as "encoder quality"
until the decoder is held fixed. Once it is, the encoder-pretraining benefit on this
task is small and backbone-specific.**

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

*Caveat:* the uniform decoder is **lighter** than SwinUNETR (Swin absolute Dice drops
0.75 → 0.59–0.64), so one reading is "a low-capacity shared decoder can't exploit a
strong encoder, masking its advantage." That's a real limitation — but it cuts to the
same practical point: the pretrained-encoder benefit on ISLES22 is *entangled* with a
co-designed high-capacity decoder, and a shared decoder mostly equalizes pretrained and
random encoders. The within-family FM-vs-scratch contrast (same arch, same decoder) is
airtight for "*with this decoder*, pretraining the encoder barely helps."

## The synthesis

Frozen linear-probe rank and finetuned-seg rank don't track each other. Finetuning
*rescues* some backbones the frozen probe writes off (SIAM) and fails to rescue others
(AnatCL); a backbone that probes well frozen (SimCLR3D) can underwhelm finetuned. So a
frozen probe is a poor model-selection signal for any task you intend to finetune.

And the finetuned cross-arm ranking itself is mostly a **decoder** ranking: hold the
decoder fixed and the gaps largely vanish, the order reshuffles, and the residual
encoder-pretraining benefit is small (≤ +0.03) and backbone-specific (only fomo60k).
The honest bottom line for sMRI FMs on this task: **encoder pretraining is a small,
inconsistent effect once architecture and decoder are controlled — the big numbers come
from the decoder you bolt on, and from comparing against a weaker baseline.**

## Limitations (read before citing)

- **Cross-arm seg ranking mixes encoder + decoder** — *now quantified in Part 3.* The
  Part-2 decoders differ by arm (SwinUNETR / ResNet-UNet / nnU-Net); the uniform-decoder
  rerun shows this confound was *large* (spread 0.15 → 0.05, ranking reorders). The one
  Part-2 claim that survived without a uniform decoder: SIAM is worst-frozen yet +0.07
  over scratch **on the identical nnU-Net decoder** — no decoder difference explains it.
- **The uniform decoder is lighter than SwinUNETR** (Part 3 caveat): it may under-exploit
  a strong encoder, so the small decoder-controlled gains could understate a co-designed
  encoder+decoder's potential. The matched-arch FM-vs-scratch contrast is still clean.
- Our Part-2 from-scratch baseline (0.667) is weaker than the FOMO paper's few-shot
  scratch (0.729) — different topology/budget/2-vs-3 modalities. Part 3's matched-arch
  scratch (0.61–0.62 on the shared decoder) is the internally-controlled floor; the
  Part-2 "+0.07" is "vs *that* weaker nnU-Net scratch," not the paper's number.
- n=25 test/fold; 5-fold gives the error bars. ρ=0.31 at n=6 isn't significant on its
  own, but the per-arm dissociations (SIAM, SimCLR3D) are individually clear.

## Reusable for the team

- **`SEG011_ISLES22_IschStroke`** — a new working seg task; the team's seg leaderboard
  was meningioma 0.0 / trigeminal 0.18 (both failures), this one reaches ~0.74.
- **A SIAM seg-inference bug fix** — SIAM's anisotropic `3d_fullres` plan (stride
  64×64×32) crashed sliding-window inference; fixed in `seg_inference.py` (pad tiles to
  mult-of-64). Helps any SIAM seg run.
- **Bolt-on seg decoders** for the encoder-only FM arms (SwinUNETR re-parent for Swin,
  ResNet-UNet for ResNet) — `models_smri_*.py` + `seg_decoders.py`, wired so asparagus's
  encoder/decoder LR split works.
- **Uniform shared decoder** (`UniformUNetDecoder` / `UniformSegBackbone` in
  `seg_decoders.py`) — any encoder's 5-level pyramid → 1×1 adapters → a byte-identical
  U-Net body, so seg comparisons isolate the *encoder*. Each `models_smri_*.py` adds a
  `*UniformSegBackbone`; `*_useg.yaml` configs + `scratch_{swin,resnet}_useg.yaml` floors.
  This is the apparatus that turned a "+0.07 pretraining win" into "≤+0.03, decoder-bound."
- **Modal seg sweep** (`scripts/modal_seg_sweep.py`) — 5-fold × roster in ~2 h vs ~30 h
  on one GPU, with a `--debug` smoke for cheap red-green validation.

## Next steps

1. ~~Uniform decoder across arms → clean *encoder* ranking~~ **— done (Part 3).** The
   decoder confound was large; decoder-controlled encoder-pretraining gain is ≤ +0.03.
2. **Higher-capacity shared decoder** (e.g. a SwinUNETR-class body fed by the same
   adapters) to test the Part-3 caveat — does fomo60k's +0.03 grow, and do the others
   stay flat, with a decoder that can exploit a strong encoder?
3. Add **brainiac** (UNETR at 96³ with a matched patch size) for completeness.
4. **3-modality** ISLES22 (resample FLAIR to DWI) for the exact paper protocol.
5. A **better-tuned scratch** (proper nnU-Net schedule) — Part 3 already shows the
   matched-arch scratch (0.61–0.62) closes most of the Part-2 "+0.07."
