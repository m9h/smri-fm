from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC

from models.supervised_base import BaseSupervisedModel


class SmoothedWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.0):
        """
        weight: Tensor of shape [C] for class weights (optional)
        smoothing: float in [0, 1], label smoothing factor
        """
        super().__init__()
        self.weight = weight
        self.smoothing = smoothing


    def forward(self, logits, targets):        
        num_classes = logits.size(1)
        targets = F.one_hot(targets, num_classes).float()

        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + self.smoothing / num_classes

        log_probs = F.log_softmax(logits, dim=1)

        if self.weight is not None:
            weight = self.weight.unsqueeze(0)  # [1, C]
            loss = - (targets * log_probs * weight).sum(dim=1)
        else:
            loss = - (targets * log_probs).sum(dim=1)
        loss = loss.mean()

        return loss


class SupervisedClsModel(BaseSupervisedModel):
    """
    Supervised model for classification tasks.
    Inherits from BaseSupervisedModel and implements classification-specific functionality.
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
            deep_supervision=False,  # Classification doesn't use deep supervision
        )


    def _configure_metrics(self, prefix: str):
        """
        Configure classification-specific metrics

        Args:
            prefix: Prefix for metric names (train or val)

        Returns:
            MetricCollection: Collection of classification metrics
        """
        return MetricCollection(
            {
                f"{prefix}/accuracy": Accuracy(
                    task="multiclass", num_classes=self.num_classes
                ),
                f"{prefix}/precision": Precision(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                f"{prefix}/recall": Recall(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                f"{prefix}/f1": F1Score(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
            }
        )


    def _configure_losses(self):
        """
        Configure classification-specific loss functions

        Returns:
            tuple: (train_loss_fn, val_loss_fn)
        """        
        class_weights = torch.tensor([0.4, 0.6], device=self.device)
        label_smoothing = 0.
        loss_fn = SmoothedWeightedCrossEntropyLoss(weight=class_weights, smoothing=label_smoothing)

        return loss_fn, loss_fn


    def _process_batch(self, batch):
        """
        Process classification batch data

        Args:
            batch: Input batch

        Returns:
            tuple: (inputs, target, file_path)
        """
        inputs, target, file_path, seg = batch["image"], batch["label"], batch["file_path"], batch["seg"]
        # Convert target to long for classification tasks
        target = target.long()

        if seg is not None:
            # Tumor present, keep original label
            target = target * (seg > 0).long()
        
        if target.dim() > 1:
            target = target.squeeze(-1)        

        return inputs, target, file_path


    def compute_metrics(self, metrics, output, target, ignore_index=None):
        """
        Compute classification metrics

        Args:
            metrics: Metrics collection
            output: Model output
            target: Ground truth
            ignore_index: Index to ignore in metrics (not used in classification)

        Returns:
            dict: Dictionary of computed metrics
        """
        # Apply softmax to get probabilities
        # print(output, target)
        probabilities = F.softmax(output, dim=1)
        return metrics(probabilities, target)


    # full-case evaluation
    def evaluate_full_case(self, batch, patch_size=(64,64,64), overlap=0.5):
        image = batch["image"].float().to(self.device)
        target = batch["label"].long().to(self.device)

        with torch.no_grad():
            probs = self.model.predict(
                mode="3D",
                data=image,
                patch_size=patch_size,
                overlap=overlap,
                sliding_window_prediction=True,
                device=self.device,
            )
            # Max over batches [class [0/1]]
            pred = torch.argmax(probs, dim=-1)  # shape [B]

        # Accuracy
        acc = (pred.cpu() == target.cpu()).float().mean().item()
        
        return acc