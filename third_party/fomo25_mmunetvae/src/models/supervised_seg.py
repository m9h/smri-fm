from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice

from yucca.modules.optimization.loss_functions.deep_supervision import (
    DeepSupervisionLoss,
)
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE
from yucca.modules.metrics.training_metrics import F1

from models.supervised_base import BaseSupervisedModel


class TverskyCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        alpha=0.7,
        beta=0.3,
        smooth=1e-6,
        apply_softmax=True,
        weight_ce=1.0,
        weight_tversky=1.0,
        ce_weight=None,  # tensor of class weights if needed
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.apply_softmax = apply_softmax
        self.weight_ce = weight_ce
        self.weight_tversky = weight_tversky
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)


    def forward(self, logits, targets):
        """
        logits: [B, C, D, H, W]
        targets: [B, D, H, W] with class indices (long)
        """
        targets = targets.squeeze(1).long()

        # Make sure weights are on same device as logits
        self.ce_loss.weight = self.ce_loss.weight.to(logits.device)
        ce = self.ce_loss(logits, targets)

        if self.apply_softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits

        num_classes = probs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
    
        # Compute foreground presence for each sample (i.e., not all background)
        fg_mask = targets_one_hot[:, 1:].sum(dim=(1, 2, 3, 4)) > 0  # exclude background class
    
        if fg_mask.any():
            # Only use the non-empty patches for Tversky loss
            probs_fg = probs[fg_mask]
            targets_one_hot_fg = targets_one_hot[fg_mask]
    
            dims = (0, 2, 3, 4)
            TP = torch.sum(probs_fg * targets_one_hot_fg, dim=dims)
            FP = torch.sum(probs_fg * (1 - targets_one_hot_fg), dim=dims)
            FN = torch.sum((1 - probs_fg) * targets_one_hot_fg, dim=dims)
    
            tversky = (TP + self.smooth) / (
                TP + self.alpha * FN + self.beta * FP + self.smooth
            )
            tversky_loss = 1.0 - tversky.mean()
        else:
            # No foreground in any patch — skip Tversky loss
            tversky_loss = torch.tensor(0.0, device=logits.device)
    
        return self.weight_ce * ce + self.weight_tversky * tversky_loss


class SupervisedSegModel(BaseSupervisedModel):
    """
    Supervised model for segmentation tasks.
    Inherits from BaseSupervisedModel and implements segmentation-specific functionality.
    """

    def __init__(
        self,
        config: dict = {},
        learning_rate: float = 1e-3,
        do_compile: Optional[bool] = False,
        compile_mode: Optional[str] = "default",
        weight_decay: float = 3e-5,
        amsgrad: bool = False,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
        deep_supervision: bool = False,
        use_skip_connections: bool = True,
    ):
        super().__init__(
            config=config,
            learning_rate=learning_rate,
            do_compile=do_compile,
            compile_mode=compile_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps=eps,
            betas=betas,
            deep_supervision=deep_supervision,
            use_skip_connections=use_skip_connections,
        )
        # keep Dice instance for convenience (binary)
        self.case_dice_metric = Dice(
            num_classes=2,
            ignore_index=0,
        )
        self.val_case_dices = []  # buffer for per-case values

        
    def validation_step(self, batch, batch_idx):
        inputs, target, _ = self._process_batch(batch)

        # run sliding-window prediction on the full volume
        preds = self.run_predict(inputs)["task_output"]

        # --- compute per-case dice ---
        dice_val = self.case_dice_metric(
            preds,  # discrete preds
            target  # ground truth indices
        ).item()

        # save this case’s dice
        self.val_case_dices.append(dice_val)

        # (optional) still log patch-level loss/metrics with TorchMetrics if you want
        loss = self.loss_fn_val(preds, target)        
        metrics = self.compute_metrics(self.val_metrics, preds, target)
        self.log_dict(
            {"val/loss": loss} | metrics,
            prog_bar=self.progress_bar,
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return loss


    def on_validation_epoch_end(self):
        # now average per-case dice (macro)
        if self.val_case_dices:
            mean_dice = sum(self.val_case_dices) / len(self.val_case_dices)
            self.log("val/full_case_dice", mean_dice, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True,)
            self.val_case_dices.clear()  # reset buffer


    def _configure_metrics(self, prefix: str):
        """
        Configure segmentation-specific metrics

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of segmentation metrics
        """
        return MetricCollection(
            {
                f"{prefix}/dice": Dice(
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                ),
                f"{prefix}/F1": F1(
                    num_classes=self.num_classes,
                    ignore_index=0 if self.num_classes > 1 else None,
                    average=None,
                ),
            },
        )


    def _configure_losses(self):
        """
        Configure segmentation-specific loss functions

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """
        loss_fn_train = TverskyCrossEntropyLoss(
             alpha=0.7, beta=0.3,
             apply_softmax=True,
             weight_ce=0.5, weight_tversky=1.0,
             ce_weight=torch.tensor([0.1, 0.9])
        )
        loss_fn_val = TverskyCrossEntropyLoss(
             alpha=0.7, beta=0.3,
             apply_softmax=True,
             weight_ce=0.5, weight_tversky=1.0,
             ce_weight=torch.tensor([0.1, 0.9])
        )

        if self.deep_supervision:
            loss_fn_train = DeepSupervisionLoss(loss_fn_train, weights=None)

        return loss_fn_train, loss_fn_val


    def compute_metrics(self, metrics, output, target, ignore_index: int = 0):
        """
        Compute segmentation metrics, handling per-class results

        Args:
            metrics: Metrics collection
            output: Model output
            target: Ground truth
            ignore_index: Index to ignore in metrics (usually background)

        Returns:
            dict: Dictionary of computed metrics
        """
        metrics = metrics(output, target)
        tmp = {}
        to_drop = []
        for key in metrics.keys():
            if metrics[key].numel() > 1:
                to_drop.append(key)
                for i, val in enumerate(metrics[key]):
                    if not i == ignore_index:
                        tmp[key + "_" + str(i)] = val
        for k in to_drop:
            metrics.pop(k)
        metrics.update(tmp)
        return metrics



            
