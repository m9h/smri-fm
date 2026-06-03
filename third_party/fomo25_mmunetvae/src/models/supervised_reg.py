from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score

from models.supervised_base import BaseSupervisedModel


class RegressionLoss(nn.Module):
    def __init__(self, beta=0.1):
        """
        Regression Loss
        """
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss(beta=beta)

    def forward(self, logits, targets):
        loss = self.reg_loss(targets, logits)

        return loss


class SupervisedRegModel(BaseSupervisedModel):
    """
    Supervised model for regression tasks.
    Inherits from BaseSupervisedModel and implements regression-specific functionality.
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
            deep_supervision=False,  # Regression doesn't use deep supervision
        )


    def _configure_metrics(self, prefix: str):
        """
        Configure regression-specific metrics

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of regression metrics
        """
        return MetricCollection(
            {
                f"{prefix}/mse": MeanSquaredError(),
                f"{prefix}/mae": MeanAbsoluteError(),
                # f"{prefix}/r2": R2Score(),  # R2 gives error for batch=1
            }
        )


    def _configure_losses(self):
        """
        Configure regression-specific loss functions

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """        
        loss_fn = RegressionLoss(beta=10.0)

        return loss_fn, loss_fn


    def _process_batch(self, batch):
        """
        Process regression batch data

        Args:
            batch: Input batch

        Returns:
            tuple: (inputs, target, file_path)
        """
        inputs, target, file_path, seg = batch["image"], batch["label"], batch["file_path"], batch["seg"]

        # Keep target as float for regression tasks
        target = target.float()

        return inputs, target, file_path


    def compute_metrics(self, metrics, output, target, ignore_index=None):
        """
        Compute regression metrics

        Args:
            metrics: Metrics collection
            output: Model output
            target: Ground truth
            ignore_index: Index to ignore in metrics (not used in regression)

        Returns:
            dict: Dictionary of computed metrics
        """
        # print(output, target)
        return metrics(output, target)


    def evaluate_full_case(self, batch, patch_size=(64,64,64), overlap=0.5):
        image = batch["image"].float().to(self.device)
        target = batch["label"].float().to(self.device)

        with torch.no_grad():
            pred_val = self.model.predict(
                mode="3D",
                data=image,
                patch_size=patch_size,
                overlap=overlap,
                sliding_window_prediction=True,
                device=self.device,
            )
            # Median over batches
            pred_val = torch.median(pred_val.view(-1))

        # Absoulte error
        abs_err = torch.abs(pred_val - target.view(-1))

        return abs_err.mean()