import torch
import torch.nn as nn
import torch.nn.functional as F

from yucca.functional.utils.kwargs import filter_kwargs

from augmentations.mask import random_mask
from models import networks


from models.self_supervised import SelfSupervisedModel


def kl_divergence(mu, logvar):
    # KL divergence between N(mu, sigma^2) and N(0, 1)
    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1. - logvar)
    # return kl.sum(dim=1).mean()  # average over batch
    return kl.mean()  # average over batch


def cosine_contrastive_loss(z1, z2):
    # Pisitive cosine similiarity to minimize distance
    z1 = F.normalize(z1.view(z1.size(0), -1), dim=1)
    z2 = F.normalize(z2.view(z2.size(0), -1), dim=1)
    return 1 - F.cosine_similarity(z1, z2, dim=1).mean()


def modality_contrastive_loss(z_m_a, z_m_b):
    # Negative cosine similarity to maximize distance
    z_m_a = F.normalize(z_m_a.view(z_m_a.size(0), -1), dim=1)
    z_m_b = F.normalize(z_m_b.view(z_m_b.size(0), -1), dim=1)
    return F.cosine_similarity(z_m_a, z_m_b, dim=1).mean()


class SelfSupervisedModelCrossPatch(SelfSupervisedModel):
    def __init__(
        self,
        model_name: str,
        steps_per_epoch: int,
        epochs: int,
        learning_rate: float,
        config: dict,
        optimizer: str = "AdamW",
        warmup_epochs: int = 10,
        cosine_period_ratio: float = 1,
        input_channels: int = 1,
        num_classes: int = 1,
        patch_size: list | tuple = None,
        mask_patch_size: int = 4,
        mask_ratio: float = 0.6,
        should_compile: bool = False,
        compile_mode: str = None,
        debug_losses: bool = False,
        rec_loss_masked_only: bool = False,
        disable_image_logging: bool = False,
    ):
        super().__init__(model_name, steps_per_epoch, epochs, learning_rate, config, optimizer, warmup_epochs,
                         cosine_period_ratio, input_channels, num_classes, patch_size, mask_patch_size, mask_ratio,
                         should_compile, compile_mode, debug_losses, rec_loss_masked_only, disable_image_logging)
                
        # Losses for self-supervised
        mse_reduction = "none" if self.debug_losses else "mean"
        self._rec_loss_fn = nn.SmoothL1Loss(beta=0.1, reduction=mse_reduction)
        self.lambda_rec = 3.0  # Self-reconstruction
        self.lambda_cross = 1.0  # Cross-reconstruction
        self.lambda_kl = 0.001  # KL
        self.lambda_contrast = 0.5  # Contrastive learning
        self.total_anneal_epochs = 10  # For the KL

        # Save params and start training
        self.save_hyperparameters()
        self.load_model()


    def load_model(self):
        print(f"Loading Model: 3D {self.model_name}")
        model_func = getattr(networks, self.model_name)

        print("Found model: ", model_func)

        conv_op = torch.nn.Conv3d
        norm_op = torch.nn.InstanceNorm3d

        model_kwargs = {
            # Applies to all models
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            # Applies to most CNN-based architectures (exceptions: UXNet)
            "norm_op": norm_op,
            # Pretrainnig
            "mode": "mae",
            "patch_size": self.patch_size,
            "use_vae": True, # Use VAE for pretraining
        }
        model_kwargs = filter_kwargs(model_func, model_kwargs)
        model = model_func(**model_kwargs)

        self.model = (
            torch.compile(model, mode=self.compile_mode)
            if self.should_compile
            else model
        )


    def training_step(self, batch, batch_idx):
        x_a = batch["x_a"]["image"].float()
        x_b = batch["x_b"]["image"] if batch["x_b"]["image"] is not None else None

        assert (
            x_a.shape[1] == self.input_channels
        ), f"Expected {self.input_channels} input channels but got {x_a.shape[1]}"
        if not (0 <= x_a.min() and x_a.max() <= 1):
            print(
                f"Intensities of batch are not in (0, 1) but instead {(x_a.min(), x_a.max())}"
            ) 

        if x_b is not None:
            x_b = x_b.float()
            assert (
                x_b.shape[1] == self.input_channels
            ), f"Expected {self.input_channels} input channels but got {x_b.shape[1]}"
            if not (0 <= x_b.min() and x_b.max() <= 1):
                print(
                    f"Intensities of batch are not in (0, 1) but instead {(x_b.min(), x_b.max())}"
                ) 

        results_fwd, mask_a, mask_b, mask_x_a, mask_x_b = self._augment_and_forward(x_a, x_b)
        loss, loss_dict = self.multimodal_loss(results_fwd, x_a, x_b=x_b, mask_a=mask_a, mask_b=mask_b, rec_loss_masked_only=self.rec_loss_masked_only, return_dict=True, stage="train")
        
        if batch_idx == 0 and not self.disable_image_logging and not self.trainer.sanity_checking:
            y_hat = results_fwd['x_recon']
            print(f"Max. y_hat: {y_hat.max()}")
            file_path = batch["x_a"]["file_path"]
            self._log_debug_images(mask_x_a, x_a, y_hat, stage="train", file_paths=file_path, idx=0)

        assert loss is not None, "Loss is None"
        assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"
        
        self.log_dict(loss_dict)
        
        return loss


    def validation_step(self, batch, batch_idx):
        x_a = batch["x_a"]["image"].float()
        x_b = batch["x_b"]["image"] if batch["x_b"]["image"] is not None else None

        assert (
            x_a.shape[1] == self.input_channels
        ), f"Expected {self.input_channels} input channels but got {x_a.shape[1]}"
        if not (0 <= x_a.min() and x_a.max() <= 1):
            print(
                f"Intensities of batch are not in (0, 1) but instead {(x_a.min(), x_a.max())}"
            ) 

        if x_b is not None:
            x_b = x_b.float()
            assert (
                x_b.shape[1] == self.input_channels
            ), f"Expected {self.input_channels} input channels but got {x_b.shape[1]}"
            if not (0 <= x_b.min() and x_b.max() <= 1):
                print(
                    f"Intensities of batch are not in (0, 1) but instead {(x_b.min(), x_b.max())}"
                ) 
        
        results_fwd, mask_a, mask_b, mask_x_a, mask_x_b = self._augment_and_forward(x_a, x_b)
        loss, loss_dict = self.multimodal_loss(results_fwd, x_a, x_b=x_b, mask_a=mask_a, mask_b=mask_b, rec_loss_masked_only=self.rec_loss_masked_only, return_dict=True, stage="val")

        if batch_idx == 0 and not self.disable_image_logging and not self.trainer.sanity_checking:            
            y_hat = results_fwd['x_recon']
            file_path = batch["x_a"]["file_path"]
            self._log_debug_images(mask_x_a, x_a, y_hat, stage="val", file_paths=file_path, idx=0)

        assert loss is not None, "Loss is None"
        assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"
        
        self.log_dict(loss_dict)


    def rec_loss(self, y, y_hat, mask=None):
        """
        Reconstruction MSE loss. If a mask tensor is provided, the loss will only be calculated on masked tokens.
        """
        # y_hat = torch.sigmoid(y_hat)
        if mask is not None:
            diff = (y - y_hat) ** 2
            masked_diff = diff[mask]
            return masked_diff.mean()  # average over masked voxels only
        else:
            return self._rec_loss_fn(y, y_hat)


    def multimodal_loss(self, results_fwd, x_a, x_b=None, mask_a=None, mask_b=None, rec_loss_masked_only=False, 
                        return_dict=False, stage="train"):

        if not rec_loss_masked_only:
            # Be sure we are not using them
            mask_a = None
            mask_b = None

        # Rec loss.
        L_rec_a = self.rec_loss(x_a, results_fwd['x_recon'], mask_a)
        try:            
            L_rec_b = self.rec_loss(x_b, results_fwd['x_recon_b'], mask_b) if 'x_recon_b' in results_fwd else 0
        except AttributeError:
            print("X_b")
            print(x_b)
            print("Recon")
            print(results_fwd['x_recon_b'])
            raise AttributeError("Error in loss rec computation for B.")
        
        # KL divergence
        L_kl_s_a = kl_divergence(results_fwd['mu_s'], results_fwd['logvar_s'])
        L_kl_m_a = kl_divergence(results_fwd['mu_m'], results_fwd['logvar_m'])

        L_kl_s_b = kl_divergence(results_fwd['mu_s_b'], results_fwd['logvar_s_b']) if 'mu_s_b' in results_fwd else 0
        L_kl_m_b = kl_divergence(results_fwd['mu_m_b'], results_fwd['logvar_m_b']) if 'mu_m_b' in results_fwd else 0

        # Cross-rec        
        L_cross_ab = self.rec_loss(x_a, results_fwd['x_cross_ab'], mask_a) if 'x_cross_ab' in results_fwd else 0
        L_cross_ba = self.rec_loss(x_b, results_fwd['x_cross_ba'], mask_b) if 'x_cross_ba' in results_fwd else 0
        
        # Contrastive term + (for the subjects)
        L_contrast = cosine_contrastive_loss(results_fwd['z_s'], results_fwd['z_s_b']) if 'z_s_b' in results_fwd else 0        
        
        # Contrastive term - (for the modalities)
        L_contrast_mod = modality_contrastive_loss(results_fwd['z_m'], results_fwd['z_m_b']) if 'z_m_b' in results_fwd else 0

        kl_weight = min(self.lambda_kl, self.current_epoch / self.total_anneal_epochs * self.lambda_kl)
        loss = (
            self.lambda_rec * (L_rec_a + L_rec_b) +
            self.lambda_cross * (L_cross_ab + L_cross_ba) +
            kl_weight * (L_kl_s_a + L_kl_m_a + L_kl_s_b + L_kl_m_b) +
            self.lambda_contrast * L_contrast +
            self.lambda_contrast * L_contrast_mod
        )

        loss_dict = {
            f"{stage}/loss": loss,
            f"{stage}/rec_a": L_rec_a,
            f"{stage}/rec_b": L_rec_b,
            f"{stage}/cross_ab": L_cross_ab,
            f"{stage}/cross_ba": L_cross_ba,
            f"{stage}/kl_s_a": L_kl_s_a,
            f"{stage}/kl_m_a": L_kl_m_a,
            f"{stage}/kl_s_b": L_kl_s_b,
            f"{stage}/kl_m_b": L_kl_m_b,
            f"{stage}/contrast": L_contrast,
            f"{stage}/contrast_mod": L_contrast_mod,
        }
        
        return loss, loss_dict


    def _augment_and_forward(self, x_a, x_b):
        with torch.no_grad():
            # .clone() to avoid in-place operation on the original data
            x_a, mask_a = random_mask(x_a.clone(), self.mask_ratio, self.mask_patch_size)
            x_b, mask_b = random_mask(x_b.clone(), self.mask_ratio, self.mask_patch_size)

        # Masked regions -> 0
        results_fwd = self.model(x_a, x_b)

        return results_fwd, mask_a, mask_b, x_a, x_b