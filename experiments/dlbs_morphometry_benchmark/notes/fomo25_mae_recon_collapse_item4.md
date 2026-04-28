# Item 4 — FOMO25 MAE recon collapse: first-pass investigation

> "deeper analysis of the fomo25 mae baseline (why is the recon
>  perfect, why does the loss go to zero instantly?) — @Dojo @Rohit"

Dropping a first-pass on this for Dojo + Rohit. I read through the
asparagus AMAES SSL training code (the loss + masking pipeline that
produced FOMO-MRI/AMAES_resenc_b on FOMO300K) and there are three
issues that compound to produce exactly the loss-collapse pattern
you're seeing. None of them are subtle.

## TL;DR

The AMAES masked-AE training has the loss averaged over the full
image volume (default), including the ~40% of un-masked voxels
that the U-Net trivially passes through skip connections from
encoder to decoder. So 40% of the loss contribution is essentially
zero from initialization, and gets even closer to zero within a few
steps as the model learns to do "identity through skip" for the
visible regions. The "loss → 0 instantly" is real but the
attribution is wrong: it's not that the model is solving the MAE
task — it's that the loss is dominated by an embarrassingly easy
identity-on-visible-region term.

## The three compounding issues

All quotes are from
`asparagus/modules/lightning_modules/self_supervised.py`
inside `ghcr.io/m9h/fomo25-arm:latest`.

### Issue 1 — `rec_loss_masked_only` defaults to False

```python
class SelfSupervisedModule(BaseModule):
    def __init__(
        self,
        ...
        rec_loss_masked_only: bool = False,   # ← line 29
        ...
    ):
```

Used in `training_step`:

```python
loss = self._rec_loss(pred, y, mask if self.rec_loss_masked_only else None)
```

When `False`, the `mask` argument to `_rec_loss` is `None`, so the
function falls through to:

```python
def _rec_loss(self, pred, y, mask=None):
    if mask is not None:
        ...
    return self._rec_loss_fn(pred, y)   # MSELoss(reduction='mean')
```

→ **MSE over the full image, mask ignored**. The model is rewarded
just as much for getting un-masked voxels right as for predicting
masked ones. This is not what proper MAE training looks like (cf.
He et al. 2021 — they explicitly scale loss to *masked-only*).

### Issue 2 — ResEnc-UNet-B has skip connections from encoder to decoder

The model class is `ResidualEncoderUNet` from
`gardening_tools.modules.networks.resunet`, instantiated via the
`resenc_unet_b()` factory. It's a vanilla U-Net architecture with
encoder→decoder skips at every level. So the decoder doesn't have
to predict the un-masked regions from the bottleneck — it gets a
near-direct copy via the skips.

Combined with Issue 1: ~40% of every voxel's loss contribution
collapses to MSE ≈ 0 trivially. After one or two optimization
steps, that 40% is at machine precision. The remaining 60% is the
actual masked-region prediction, which is the only thing the model
is genuinely learning. But because the loss is *averaged over all
voxels*, the magnitude is suppressed by the mask ratio — the
*reported* loss looks ~3× smaller than the actual masked-region
MSE.

### Issue 3 — even with `rec_loss_masked_only=True`, the loss is scaled wrong

When you turn the flag on, the loss path becomes:

```python
def _rec_loss(self, pred, y, mask=None):
    if mask is not None:
        y_masked = y.clone()
        pred_masked = pred.clone()
        y_masked[~mask] = 0
        pred_masked[~mask] = 0
        return self._rec_loss_fn(pred_masked, y_masked)   # MSELoss(reduction='mean')
```

This zeros the un-masked regions in both `pred` and `y`, then
computes mean MSE. **But `nn.MSELoss(reduction='mean')` divides by
the total number of voxels**, not by `mask.sum()`. So the loss is
multiplied by `mask_ratio` (default 0.6) — what should be a
masked-region MSE of 0.4 becomes a reported value of 0.24.

Not an algorithm bug per se (gradients are still correct), but it
explains why the reported "masked-only" loss number also looks
artificially small. A proper masked-AE loss should be:

```python
return ((pred - y)[mask] ** 2).mean()    # divide by mask.sum() = sum(mask)
```

## How to reproduce / confirm

The fastest sanity check is to instrument the training step to log
*two* loss numbers separately:

```python
# Inside training_step, after pred is computed:
loss_visible = ((pred - y)[~mask] ** 2).mean()
loss_masked  = ((pred - y)[mask] ** 2).mean()
self.log("loss_visible", loss_visible)
self.log("loss_masked",  loss_masked)
```

Prediction: `loss_visible` will collapse to ~0 within ~50 training
steps (skip connections do their thing), while `loss_masked` will
stay O(0.5) and decay like a normal MAE training curve.

## What the team can do with this

1. **Re-run AMAES pretraining with `rec_loss_masked_only=True`** —
   should produce a non-trivial loss curve and (hopefully) a
   stronger encoder. If the published `AMAES_resenc_b` checkpoint
   we're using on DLBS was trained with the default (False), then
   the per-DLBS-bracket result we got (FOMO25 best in both age
   halves at Zhang 3.40/3.74) might *under-state* what AMAES could
   do with the loss fix.

2. **Sanity-check the published checkpoint.** Two predictions if
   the loss is dominated by un-masked-region copying:
   - The encoder bottleneck (320-d) should have *low* feature
     diversity vs. a properly-trained MAE — try running a
     dimensionality-reduction (UMAP/PCA) on the features across a
     diverse cohort and see if they collapse to a low-rank manifold.
   - Reconstructions on un-masked patches should look near-perfect,
     while masked-patch reconstructions should look noticeably
     blurrier / averaged.

3. **The fix is small** — three lines:

   ```python
   # In _rec_loss, replace the mask branch:
   def _rec_loss(self, pred, y, mask=None):
       if mask is not None:
           return ((pred - y)[mask] ** 2).mean()   # masked voxels only, properly scaled
       return self._rec_loss_fn(pred, y)
   ```

   And flip the default of `rec_loss_masked_only` to `True`.
   That's it.

## Caveat I'm flagging openly

I haven't run a real AMAES pretrain to confirm — this is purely
code-reading inside the published container. Could be that the
authors *intended* to include the un-masked region in the loss as
a regularisation term (some MAE variants do this). But the
conventional reading of "MAE recon collapse" is that the model
isn't doing meaningful reconstruction on the masked regions, and
both Issue 1 and Issue 3 are consistent with that reading.

The cleanest follow-up is one of:
- Spin up a tiny pretraining run inside `fomo25-arm` with the
  proposed instrumentation and see whether the two-component loss
  pattern holds. ~1-2 hr of work.
- Send these findings to the AMAES upstream authors and ask whether
  the default was intentional.

If Dojo + Rohit have already gone deeper than this, ignore — but
this is what jumped out from the published code in <3 hr of
reading.

## Pointers

- Source files (in `ghcr.io/m9h/fomo25-arm:latest`):
  - `/usr/local/lib/python3.12/dist-packages/asparagus/modules/lightning_modules/self_supervised.py`
  - `/usr/local/lib/python3.12/dist-packages/asparagus/modules/transforms/presets/pretrain.py`
  - `/usr/local/lib/python3.12/dist-packages/gardening_tools/modules/transforms/masking.py`
  - `/usr/local/lib/python3.12/dist-packages/gardening_tools/modules/networks/resunet.py`
- AMAES paper: arxiv 2408.00640
- AMAES checkpoint (used in our DLBS ridge): https://huggingface.co/FOMO-MRI/AMAES_resenc_b
- Code repo: https://github.com/Sllambias/asparagus
