from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.heads import ClsRegHead
from models.networks.unet import UNetDecoder, UNetEncoder
from models.conv_blocks.blocks import (
    MultiLayerConvDropoutNormNonlin,
)

from yucca.modules.networks.utils.get_steps_for_sliding_window import (
    get_steps_for_sliding_window,
)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std



def gaussian_weight_nd(patch_size, sigma_scale=1./8, device="cpu"):
    """
    Create a Gaussian weight map for an N-D patch (2D or 3D).

    Args:
        patch_size (tuple): e.g. (px, py) or (px, py, pz)
        sigma_scale (float): relative sigma factor (default 1/8 of size)
        device: torch device

    Returns:
        torch.Tensor of shape (1, 1, *patch_size)
    """
    dims = len(patch_size)
    coords = [torch.arange(s, dtype=torch.float32, device=device) for s in patch_size]
    grids = torch.meshgrid(*coords, indexing="ij")
    centers = [(s - 1) / 2 for s in patch_size]
    sigmas = [s * sigma_scale for s in patch_size]

    exponent = 0
    for g, c, sig in zip(grids, centers, sigmas):
        exponent += ((g - c) / sig) ** 2

    weight = torch.exp(-0.5 * exponent)
    weight = weight / weight.max()
    return weight.unsqueeze(0).unsqueeze(0)  # [1,1,...]


class LearnableFusion3D(nn.Module):
    def __init__(self, num_modalities: int, num_channels: int = 1):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_modalities))  # Learnable scalar weights per modality
        self.softmax = nn.Softmax(dim=0)  # Ensure weights are normalized across modalities

    def forward(self, preds: torch.Tensor) -> torch.Tensor:
        """
        preds: Tensor of shape [M, B, C, D, H, W]
        returns: Tensor of shape [B, C, D, H, W] (fused prediction)
        """
        weights = self.softmax(self.weights)  # shape: [M]
        weighted_preds = preds * weights.view(-1, 1, 1, 1, 1, 1)
        fused = weighted_preds.sum(dim=0)  # sum across modality dimension
        return fused


class MultiModalUNetVAE(nn.Module):
    def __init__(
        self,
        mode: Literal[
            "mae", "classification", "segmentation", "regression", "enc"
        ] = "segmentation",  # prediction mode
        input_channels: int = 1,
        output_channels: int = 1,
        starting_filters: int = 64,
        encoder_block: nn.Module = MultiLayerConvDropoutNormNonlin.get_block_constructor(
            2
        ),
        decoder_block: nn.Module = MultiLayerConvDropoutNormNonlin.get_block_constructor(
            2
        ),
        use_skip_connections: bool = False,
        deep_supervision: bool = False,
        latent_channels_shared: int = 64,
        latent_channels_modality: int = 64,
        use_vae: bool = True,
    ):
        super().__init__()

        self.encoder_block = encoder_block
        self.decoder_block = decoder_block
        
        self.encoder = UNetEncoder(
            # input_channels=input_channels,
            input_channels=1,
            starting_filters=starting_filters,
            basic_block=encoder_block,
        )
        self.num_classes = output_channels
        self.mode = mode
        self.use_vae = use_vae 
        self.num_modalities = input_channels  # Number of modalities

        # ====== OPTIONS FOR BOTTLENECK
        last_mult = 2 ** 4
        self.use_skip_connections = use_skip_connections
        bottleneck_channels = sum([starting_filters * 2**i for i in range(5)]) # All levels!
        latent_channels_shared = (starting_filters * last_mult) // 2  # Just last one
        latent_channels_modality = (starting_filters * last_mult) - latent_channels_shared
        # ====== OPTIONS FOR BOTTLENECK

        # Set up the decoder
        self.decoder = UNetDecoder(                
                output_channels=1,
                use_skip_connections=False,
                basic_block=decoder_block,
                starting_filters=starting_filters,
            )
        if mode == "mae":
            # === Pre-training mode
            self.decoder_task = nn.Identity()  # No task-specific decoder
        elif mode == "segmentation":
            # Single channel decoder and late fusion after
            self.decoder_task = UNetDecoder(
                output_channels=output_channels,
                use_skip_connections=use_skip_connections,
                basic_block=decoder_block,
                starting_filters=starting_filters,                
            )
            # Late fusion
            self.fusion_layer = LearnableFusion3D(num_modalities=self.num_modalities)
        elif mode in ["classification", "regression"]:
            out_dim = latent_channels_modality * self.num_modalities + latent_channels_shared
            self.decoder_task = ClsRegHead(
                in_channels=out_dim, num_classes=output_channels
            )
        else:
            raise ValueError(
                "Invalid mode. Choose from 'mae', 'segmentation', 'classification', 'regression'"
            )        

        # ======= Latent space mapping from bottlenecks to a space of the size of the last bottleneck.
        self.conv_mu_shared = nn.Conv3d(bottleneck_channels, latent_channels_shared, kernel_size=1)
        self.conv_logvar_shared = nn.Conv3d(bottleneck_channels, latent_channels_shared, kernel_size=1)
        self.conv_mu_modality = nn.Conv3d(bottleneck_channels, latent_channels_modality, kernel_size=1)
        self.conv_logvar_modality = nn.Conv3d(bottleneck_channels, latent_channels_modality, kernel_size=1)


    def forward(self, x_a=None, x_b=None, x_list=None):
        if x_list is not None:
            # --------------------------
            # ---- Fine-tuning mode ----
            # --------------------------
            # This is a list of modalities, and we are in fine-tuning
            result = {}
            
            z_s_all = []
            z_m_all = []
            skips_all = []  # Only in segmentation
                        
            for ix in range(self.num_modalities):
                x = x_list[:, [ix]]  # Don't loose batch dimension
                
                # Encode modality
                skips = self.encoder(x)

                # Pool features
                features = [F.adaptive_avg_pool3d(f, output_size=skips[-1].shape[2:]) for f in skips]
                pooled = torch.cat(features, dim=1)

                # Get 'modality' and 'subject' representation                
                mu_s = self.conv_mu_shared(pooled)
                logvar_s = self.conv_logvar_shared(pooled)
                mu_m = self.conv_mu_modality(pooled)
                logvar_m = self.conv_logvar_modality(pooled)
                
                # If training use repr. trick / o.w we use the mean for prediction
                if self.training and self.use_vae:
                    z_s = reparameterize(mu_s, logvar_s)
                    z_m = reparameterize(mu_m, logvar_m)
                else: 
                    z_s = mu_s
                    z_m = mu_m
                 
                z_s_all.append(z_s)
                z_m_all.append(z_m)
                if self.use_skip_connections:
                    # Only for segmentation
                    skips_all.append(skips)

            # Mean subject representation: push it to be shared across modalities
            z_s = torch.stack(z_s_all).mean(dim=0)
            
            # Now, decoders
            if self.mode == "segmentation":
                # ===== LATE FUSION =====
                per_modality_preds = []
                for idx_m, z_m_i in enumerate(z_m_all):
                    # Concatenate each modality to the mean subject repr.
                    z_latent_i = torch.cat([z_s, z_m_i], dim=1)
                    if self.use_skip_connections:
                        # Use skip-connections + the current latent as last repr.
                        dec_input = [*skips_all[idx_m][:-1], z_latent_i]
                    else:
                        # Just the latent repr.
                        dec_input = [None, None, None, None, z_latent_i]

                    # Get prediction for this modality
                    pred_i = self.decoder_task(dec_input)
                    per_modality_preds.append(pred_i)

                # Stack predictions: shape = [num_modalities, B, C, D, H, W]
                stacked_preds = torch.stack(per_modality_preds, dim=0)

                # Get a final prediction by 'fusing' each modality pred.                
                fused_pred = self.fusion_layer(stacked_preds)
                result["task_output"] = fused_pred

            elif self.mode in ["classification", "regression"]:
                # Concatenate all modality-specific latents
                z_m = torch.cat(z_m_all, dim=1)
                # Concatenate the modality-terms to the mean subject repr.
                z_latent = torch.cat([z_s, z_m], dim=1)
                # Ger prediction
                result["task_output"] = self.decoder_task(z_latent)
                # print('output shape:')
                # print(z_latent.shape, result["task_output"].shape)

            return result
        else:
            # --------------------------
            # -- Self-supervised mode --
            # --------------------------
            # Encode each modality, independently.
            # 1st image
            skips_a = self.encoder(x_a)  # List of [x0, x1, x2, x3, x4]

            # Extract multiscale features from all encoder stages
            features_a = [F.adaptive_avg_pool3d(f, output_size=skips_a[-1].shape[2:]) for f in skips_a]
            pooled_a = torch.cat(features_a, dim=1)

            # Subject-level - A
            mu_s_a = self.conv_mu_shared(pooled_a)
            logvar_s_a = self.conv_logvar_shared(pooled_a)
            if self.training and self.use_vae:
                z_s_a = reparameterize(mu_s_a, logvar_s_a)
            else:
                z_s_a = mu_s_a  # Deterministic evaluation

            # Modality-level - A
            mu_m_a = self.conv_mu_modality(pooled_a)
            logvar_m_a = self.conv_logvar_modality(pooled_a)
            if self.training and self.use_vae:
                z_m_a = reparameterize(mu_m_a, logvar_m_a)
            else:
                z_m_a = mu_m_a  # Deterministic evaluation            

            if x_b is not None:
                # Process second-image
                skips_b = self.encoder(x_b)

                # Extract multiscale features from all encoder stages
                features_b = [F.adaptive_avg_pool3d(f, output_size=skips_b[-1].shape[2:]) for f in skips_b]
                pooled_b = torch.cat(features_b, dim=1)  # [B, C_total, D_min, H_min, W_min]        

                # Subject level - B
                mu_s_b = self.conv_mu_shared(pooled_b)
                logvar_s_b = self.conv_logvar_shared(pooled_b)
                if self.training and self.use_vae:
                    z_s_b = reparameterize(mu_s_b, logvar_s_b)
                else:
                    z_s_b = mu_s_b  # Deterministic evaluation                

                # Modality level - B
                mu_m_b = self.conv_mu_modality(pooled_b)
                logvar_m_b = self.conv_logvar_modality(pooled_b)
                if self.training and self.use_vae:
                    z_m_b = reparameterize(mu_m_b, logvar_m_b)
                else:
                    z_m_b = mu_m_b  # Deterministic evaluation

                # Average subject latent (A and B)
                z_s = 0.5 * (z_s_a + z_s_b)
            else:
                z_s = z_s_a  # Just A
                
            # Get subject A representation
            z_latent_a = torch.cat([z_s_a, z_m_a], dim=1)

            # Reconstruction of A                    
            recon_a = self.decoder([None, None, None, None, z_latent_a])

            # Initialize dict. with results
            result = {
                'x_recon': recon_a,
                'mu_s': mu_s_a,
                'logvar_s': logvar_s_a,
                'mu_m': mu_m_a,
                'logvar_m': logvar_m_a,
                'z_s': z_s_a,
                'z_m': z_m_a,
            }

            if x_b is not None:                
                # Get subject B representation
                z_latent_b = torch.cat([z_s_b, z_m_b], dim=1)
                recon_b = self.decoder([None, None, None, None, z_latent_b])

                # Cross: Subject + Modality B ; reconstruct A [A from B, ab]
                z_cross_ab = torch.cat([z_s, z_m_b], dim=1)
                cross_recon_ab = self.decoder([None, None, None, None, z_cross_ab])

                # Cross: Subject + Modality A ; reconstruct B [B from A, ba]
                z_cross_ba = torch.cat([z_s, z_m_a], dim=1)
                cross_recon_ba = self.decoder([None, None, None, None, z_cross_ba])

                # Update results dict
                result.update({
                    'x_recon_b': recon_b,
                    'mu_s_b': mu_s_b,
                    'logvar_s_b': logvar_s_b,
                    'mu_m_b': mu_m_b,
                    'logvar_m_b': logvar_m_b,
                    'z_s_b': z_s_b,
                    'z_m_b': z_m_b,
                    'x_cross_ab': cross_recon_ab,
                    'x_cross_ba': cross_recon_ba,
                })

        return result

    # ==================== FROM YUCCA =======================
    def predict(self, mode, data, patch_size, overlap, sliding_window_prediction=True, mirror=False, device="cpu"):
        if not sliding_window_prediction:
            return self._full_image_predict(data)

        elif self.mode in ["classification", "regression"]:
            # For classification/regressison collect logits per patch (as list)
            return self._patch_classification_predict(data, patch_size, overlap)
        else:
            assert mode in ["3D", "2D"]

            if mode == "3D":
                predict_fn = self._sliding_window_predict3D
            elif mode == "2D":
                predict_fn = self._sliding_window_predict2D

            pred = predict_fn(data, patch_size, overlap)
            if mirror:
                pred += torch.flip(predict_fn(torch.flip(data, (2,)), patch_size, overlap), (2,))
                pred += torch.flip(predict_fn(torch.flip(data, (3,)), patch_size, overlap), (3,))
                pred += torch.flip(predict_fn(torch.flip(data, (2, 3)), patch_size, overlap), (2, 3))
                div = 4
                if mode == "3D":
                    pred += torch.flip(predict_fn(torch.flip(data, (4,)), patch_size, overlap), (4,))
                    pred += torch.flip(predict_fn(torch.flip(data, (2, 4)), patch_size, overlap), (2, 4))
                    pred += torch.flip(predict_fn(torch.flip(data, (3, 4)), patch_size, overlap), (3, 4))
                    pred += torch.flip(
                        predict_fn(torch.flip(data, (2, 3, 4)), patch_size, overlap),
                        (2, 3, 4),
                    )
                    div += 4
                pred /= div
            return pred

    def _patch_classification_predict(self, data, patch_size, overlap):
        # For classification/regression
        logits_list = []
        x_steps, y_steps, z_steps = get_steps_for_sliding_window(data.shape[2:], patch_size, overlap)
        px, py, pz = patch_size
                
        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    patch = data[:, :, xs:xs+px, ys:ys+py, zs:zs+pz]
                    out_dict = self.forward(x_list=patch)
                    logits = out_dict["task_output"]  # shape: [B, num_classes]
                    logits_list.append(logits)

        # Aggregate        
        logits_stack = torch.stack(logits_list, dim=0)  # shape: [N_patches, B, C]
        
        if self.mode == "classification":
            # Convert logits to probabilities along class dimension (assuming dim=1 is class)
            # logits_stack: [P, 1, 2] -> [P, 2]
            logits = logits_stack.squeeze(1)
            logp = torch.log_softmax(logits, dim=-1)   # [P,2]
            agg = logp.mean(dim=0)                     # [2]  <-- average evidence
            probs = torch.softmax(agg, dim=-1)    # scalar in [0,1]
            # print('probs shape:')
            # print(probs.shape)
            return probs.unsqueeze(0)  # Batch
        
        elif self.mode == "regression":
            # Median of the stack
            return_logits = torch.median(logits_stack, dim=0).values  # shape: [C]
        else:
            raise ValueError(f"Invalid mode for patch processing: {self.mode}")

        return return_logits

    def _full_image_predict(self, data):
        """
        Standard prediction used in cases where models predict on full-size images.
        This is opposed to patch-based predictions where we use a sliding window approach to generate
        full size predictions.
        """
        out_dict = self.forward(x_list=data)
        out = out_dict['task_output']
        print(out.shape)
        return out

    def _sliding_window_predict3D(self, data, patch_size, overlap):
        """
        Sliding window prediction implementation using gaussian weights
        """
        B, _, D, H, W = data.shape
        canvas = torch.zeros(
            (data.shape[0], self.num_classes, *data.shape[2:]),
            device=data.device,
        )
        weight_map = torch.zeros((1, 1, D, H, W), device=data.device).to(data.dtype)  # broadcastable
        
        x_steps, y_steps, z_steps = get_steps_for_sliding_window(data.shape[2:], patch_size, overlap)
        px, py, pz = patch_size
        w = gaussian_weight_nd(patch_size, device=data.device)  # [1,1,px,py,pz]

        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    # check if out of bounds
                    out_dict = self.forward(x_list=data[:, :, xs : xs + px, ys : ys + py, zs : zs + pz])
                    out = out_dict['task_output']
                    canvas[:, :, xs : xs + px, ys : ys + py, zs : zs + pz] += out * w
                    weight_map[:, :, xs:xs+px, ys:ys+py, zs:zs+pz] += w
        
        canvas /= torch.clamp(weight_map, min=1e-6)
        return canvas

    def _sliding_window_predict2D(self, data, patch_size, overlap):
        """
        Sliding window prediction implementation
        """        
        canvas = torch.zeros(
            (data.shape[0], self.num_classes, *data.shape[2:]),
            device=data.device,
        )

        px, py = patch_size
        w = gaussian_weight_nd(patch_size, device=data.device).to(data.dtype)  # [1,1,px,py,pz]
        # If we have 5 dimensions we are working with 3D data, and need to predict each slice.
        if len(data.shape) == 5:
            B, _, Z, H, W = data.shape
            x_steps, y_steps = get_steps_for_sliding_window(data.shape[3:], patch_size, overlap)
            weight_map = torch.zeros((1, 1, Z, H, W), device=data.device)
            for idx in range(data.shape[2]):
                for xs in x_steps:
                    for ys in y_steps:
                        out_dict = self.forward(x_list=data[:, :, idx, xs : xs + px, ys : ys + py])                        
                        out = out_dict['task_output']
                        canvas[:, :, idx, xs : xs + px, ys : ys + py] += out * w
                        weight_map[:, :, idx, xs:xs+px, ys:ys+py] += w

            canvas /= torch.clamp(weight_map, min=1e-6)
            return canvas

        else:  # 2D case: [B, C, H, W]
            B, _, H, W = data.shape
            canvas = torch.zeros((B, self.num_classes, H, W), device=data.device)
            weight_map = torch.zeros((1, 1, H, W), device=data.device)

            x_steps, y_steps = get_steps_for_sliding_window((H, W), patch_size, overlap)
            px, py = patch_size

            for xs in x_steps:
                for ys in y_steps:
                    patch = data[:, :, xs:xs+px, ys:ys+py]
                    out_dict = self.forward(x_list=patch)
                    out = out_dict['task_output']  # [B, C, px, py]

                    canvas[:, :, xs:xs+px, ys:ys+py] += out * w
                    weight_map[:, :, xs:xs+px, ys:ys+py] += w

            canvas /= torch.clamp(weight_map, min=1e-6)
        
            return canvas


def mmunetvae(
    input_channels: int = 1,
    output_channels: int = 1,
    mode: str = "segmentation",
    use_vae: bool = True,
    use_skip_connections: bool = False,
):
    unet_model = MultiModalUNetVAE(
        input_channels=input_channels,
        output_channels=output_channels,
        decoder_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        use_skip_connections=use_skip_connections,
        starting_filters=32,
        mode=mode,
        use_vae=use_vae,
    )

    return unet_model