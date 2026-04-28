# FOMO25 baseline ridge — checkpoint clarification needed

**Status**: pivoting to AMAES_resenc_b for now to unblock item 3, but
flagging here so we don't ship a deliverable that misses the team's
intended baseline.

## What the meeting said

> brain age ridge regression eval of fomo25 baseline ssl model (@ahmed)

## What I found when I went to grab "the FOMO25 baseline SSL model"

Two candidates exist, and they aren't the same model:

| Backbone | Hosted? | Paper | Pretraining | Notes |
|---|---|---|---|---|
| **AMAES_resenc_b** | ✅ `huggingface.co/FOMO-MRI/AMAES_resenc_b` | arxiv 2408.00640 | FOMO300K, masked autoencoding | ResEnc-UNet-B, 102M params, .ckpt published, code in `github.com/Sllambias/asparagus` |
| **mmunetvae** (SSL3D challenge winner) | ❌ not on HF FOMO-MRI org | Gordaliza et al. 2026 (arxiv 2601.13166) | FOMO50K | repo `github.com/jbanusco/fomo25` v1.0.0 has the source but the README's "💾 Model Checkpoints" section is empty, and the `jbanusco/sslmmunetave:1.0.0` image only ships source + deps, no .ckpt |

Either could plausibly be what was meant by "fomo25 baseline ssl model".

## What I'm doing in the meantime

Item 3 ridge eval is going to ship against **AMAES_resenc_b** because
that's what's actually downloadable today (HF org, documented model
card, paper, code). Same Nima protocol (GroupKFold(5) + StandardScaler
+ RidgeCV, 4 bias-correction schemes), same 60 DLBS scans / 23
subjects, drop alongside `ridge_brainiac_embed.json` for direct
comparison. Should land in 1–2 days once the arm Grace Blackwell
container builds + smoke passes.

## What I need from the team

@ahmed — did you mean AMAES_resenc_b, or do you have an mmunetvae
checkpoint somewhere we can get a copy of? If the latter, please drop
a link / NAS path / anything and I'll re-run the ridge against it
trivially.

If both are interesting, happy to ship two ridge tables.

## Caveats either way

Both AMAES (FOMO300K) and mmunetvae (FOMO50K) almost certainly have
DLBS scans in their pretraining sets — FOMO50K has DLBS explicitly
listed in source cohorts; FOMO300K is a superset. So this isn't a
held-out eval. The number is a useful comparator vs BrainIAC (which
also has overlap risk) but not an OOD generalization claim. Will
disclose this in the result write-up.
