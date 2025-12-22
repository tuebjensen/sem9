import pytorch_lightning as pl
import torch
import torch.nn as nn
from senseiver_model import Decoder, Encoder, self_attention_block
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def sparse_fourier_encode(coords, num_bands, max_freq):
    B, N = coords.shape[:2]
    # coords: (B, N, D)
    frequencies = torch.linspace(
        1.0, max_freq / 2, num_bands, device=coords.device
    )  # (num_bands,)

    c = coords.unsqueeze(-1)  # (B, N, D, 1)

    freq_grid = c * frequencies  # (B, N, D, num_bands)
    sin_enc = torch.sin(torch.pi * freq_grid)  # (B, N, D, num_bands)
    cos_enc = torch.cos(torch.pi * freq_grid)  # (B, N, D, num_bands)

    enc = torch.cat([sin_enc, cos_enc], dim=-1)  # (B, N, D, 2 * num_bands)

    return enc.reshape(B, N, -1)  # (B, N, D * 2 * num_bands)


class MultirateTokenizer(nn.Module):
    def __init__(
        self,
        num_sensors=20,
        window_size=100,
        img_size=(192, 128),
        patch_size=16,
        dim_model=512,
        spatial_bands=16,
        time_bands=16,
    ):
        super().__init__()

        self.num_sensors = num_sensors
        self.window_size = window_size
        self.width, self.height = img_size
        self.patch_size = patch_size
        self.dim_model = dim_model
        self.spatial_bands = spatial_bands
        self.time_bands = time_bands

        self.grid_h = self.height // self.patch_size
        self.grid_w = self.width // self.patch_size

        # 2 (x, y) * spatial_bands * 2 (sin, cos)
        self.pos_dim = 2 * spatial_bands * 2
        # 1 * time_bands * 2 (sin, cos)
        self.time_dim = 1 * time_bands * 2
        self.id_dim = 64

        # 2 (u, v) + pos + time + id
        self.sensor_dim = 2 + self.pos_dim + self.time_dim + self.id_dim
        self.sensor_projector = nn.Linear(self.sensor_dim, dim_model)

        # 2 (smoke, mask) per patch
        self.patch_dim = 2 * patch_size * patch_size

        self.img_dim = self.patch_dim + self.pos_dim + self.time_dim + self.id_dim
        self.img_projector = nn.Linear(self.img_dim, dim_model)

        # Multimodal tagging
        self.id_embedding = nn.Embedding(2, self.id_dim)  # 0: sensor, 1: image
        # Time indices for fourier encoding
        self.register_buffer(
            "history_times", torch.linspace(-1.0, 0.0, window_size).reshape(1, -1, 1)
        )
        self.register_buffer("img_time", torch.zeros(1, 1, 1))

    def forward(self, sensor_values, sensor_pos, image):
        B = sensor_values.shape[0]
        device = sensor_values.device

        pos_enc = sparse_fourier_encode(sensor_pos, self.spatial_bands, max_freq=64.0)
        # Repeat over time dimension: (B, N, F_p) -> (B, N, 1, F_p) -> (B, N, T, F_p) -> (B, N*T, F_p)
        pos_enc = (
            pos_enc.unsqueeze(2)
            .expand(-1, -1, self.window_size, -1)
            .reshape(B, -1, self.pos_dim)
        )

        time_enc = sparse_fourier_encode(
            self.history_times, self.time_bands, max_freq=50.0
        )
        # Repeat over sensors: (1, T, F_t) -> (B, N, T, F_t) -> (B, N*T, F_t)
        time_enc = (
            time_enc.unsqueeze(1)
            .expand(B, self.num_sensors, -1, -1)
            .reshape(B, -1, self.time_dim)
        )

        # (B, N, T, 2) -> (B, N*T, 2)
        flat_sensor_vals = sensor_values.reshape(B, -1, 2)

        # Sensor ID embedding
        type_id = self.id_embedding(
            torch.zeros(B, flat_sensor_vals.shape[1], dtype=torch.long, device=device)
        )
        sensor_tokens = self.sensor_projector(
            torch.cat([flat_sensor_vals, pos_enc, time_enc, type_id], dim=-1)
        )

        p = self.patch_size
        # (B, C, H, W) -> (B, C, H//p, W, p) -> (B, C, H//p, W//p, p, p)
        patches = image.unfold(2, p, p).unfold(3, p, p)
        # Flatten patch dimensions into sequence of tokens
        # (B, C, H//p, W//p, p, p) -> (B, H//p, W//p, C, p, p) -> (B, H//p * W//p, C * p * p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, self.patch_dim)

        # Create positional grid for patches
        ys = torch.linspace(-1, 1, self.grid_h, device=device)
        xs = torch.linspace(-1.5, 1.5, self.grid_w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Stack into (x,y) pairs
        # (h_g, w_g, 2) -> (1, h_g * w_g, 2) -> (B, h_g * w_g, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2).expand(B, -1, -1)

        pos_enc_img = sparse_fourier_encode(grid, self.spatial_bands, max_freq=64.0)

        time_enc_img = sparse_fourier_encode(
            self.img_time, self.time_bands, max_freq=50.0
        )
        time_enc_img = time_enc_img.expand(B, self.grid_h * self.grid_w, -1)

        img_id = self.id_embedding(
            torch.ones(B, self.grid_h * self.grid_w, dtype=torch.long, device=device)
        )

        img_tokens = self.img_projector(
            torch.cat([patches, pos_enc_img, time_enc_img, img_id], dim=-1)
        )

        return torch.cat([sensor_tokens, img_tokens], dim=1)


class DynamicsPropagator(nn.Module):
    def __init__(self, dim_model, steps=100, num_layers=4, heads=8):
        super().__init__()
        self.dim_model = dim_model
        self.steps = steps
        self.num_layers = num_layers
        self.num_heads = heads

        self.attention_block = self_attention_block(
            num_layers=num_layers,
            num_channels=dim_model,
            num_heads=heads,
            dropout=0.1,
        )
        self.time_embedding = nn.Embedding(steps + 1, dim_model)

    def forward(self, z):
        # z: (B, latents, dim_model)
        states = []
        current_z = z
        for k in range(1, self.steps + 1):
            time_emb = self.time_embedding(torch.tensor(k, device=z.device))

            current_z = current_z + time_emb.reshape(1, 1, -1)

            current_z = self.attention_block(current_z)

            states.append(current_z)

        return torch.stack(states, dim=1)  # (B, steps, latents, dim_model)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out = self.conv2(out)
        out += residual
        return self.gelu(out)


class UpsampleBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, scaling_factor, upsampling_type="bilinear"
    ):
        super().__init__()
        self.reduce_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.residual_block = ResidualBlock(out_channels)
        if upsampling_type == "nearest":
            self.upsample = nn.Upsample(
                scale_factor=scaling_factor, mode=upsampling_type
            )
        else:
            self.upsample = nn.Upsample(
                scale_factor=scaling_factor, mode=upsampling_type, align_corners=False
            )
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.residual_block(x)
        x = self.upsample(x)
        x = self.out_conv(x)
        x = self.gelu(x)
        return x


class TwoStageResidualHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsampling_type="bilinear",
    ):
        super().__init__()
        scaling_factor = 4
        stage_1_channels = in_channels // 2
        stage_2_channels = stage_1_channels // 2
        self.stage_1 = UpsampleBlock(
            in_channels,
            stage_1_channels,
            scaling_factor,
            upsampling_type=upsampling_type,
        )
        self.stage_2 = UpsampleBlock(
            stage_1_channels,
            stage_2_channels,
            scaling_factor,
            upsampling_type=upsampling_type,
        )

        self.final_conv = nn.Conv2d(stage_2_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.final_conv(x)
        return x


class FourStageResidualHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upsampling_type="bilinear",
    ):
        super().__init__()
        scaling_factor = 2
        stage_1_channels = in_channels // 2
        stage_2_channels = stage_1_channels // 2
        stage_3_channels = stage_2_channels // 2
        stage_4_channels = stage_3_channels // 2

        self.stage_1 = UpsampleBlock(
            in_channels,
            stage_1_channels,
            scaling_factor,
            upsampling_type=upsampling_type,
        )
        self.stage_2 = UpsampleBlock(
            stage_1_channels,
            stage_2_channels,
            scaling_factor,
            upsampling_type=upsampling_type,
        )
        self.stage_3 = UpsampleBlock(
            stage_2_channels,
            stage_3_channels,
            scaling_factor,
            upsampling_type=upsampling_type,
        )
        self.stage_4 = UpsampleBlock(
            stage_3_channels,
            stage_4_channels,
            scaling_factor,
            upsampling_type=upsampling_type,
        )

        self.final_conv = nn.Conv2d(stage_4_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.final_conv(x)
        return x


class BaselineRefinementHead(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling_type="bilinear"):
        super().__init__()
        self.model = UpsampleBlock(
            in_channels,
            in_channels // 2,
            scaling_factor=16,
            upsampling_type=upsampling_type,
        )
        self.final_conv = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        x = self.final_conv(x)
        return x


def get_refinement_head(
    head_type,
    in_channels,
    out_channels,
    upsampling_type="bilinear",
):
    if head_type == "baseline":
        return BaselineRefinementHead(
            in_channels, out_channels, upsampling_type=upsampling_type
        )
    elif head_type == "2_step_upsampling":
        return TwoStageResidualHead(
            in_channels, out_channels, upsampling_type=upsampling_type
        )
    elif head_type == "4_step_upsampling":
        return FourStageResidualHead(
            in_channels, out_channels, upsampling_type=upsampling_type
        )


class MultirateTemporalSenseiver(pl.LightningModule):
    def __init__(
        self,
        dim_model=256,
        num_latents=128,
        downsample_factor=4,
        max_epochs=10,
        smoke_mean=0.0,
        smoke_std=1.0,
        upsampling_type="bilinear",
        image_head_type="baseline",
        image_loss_type="l1",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.spatial_bands = 16
        self.time_bands = 16
        self.patch_size = 16
        self.downsample_factor = downsample_factor
        self.model_steps = 100 // downsample_factor

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.tokenizer = MultirateTokenizer(
            dim_model=dim_model, window_size=self.model_steps
        )
        self.encoder = Encoder(
            input_ch=dim_model,
            preproc_ch=None,
            num_latents=num_latents,
            num_latent_channels=dim_model,
        )

        self.propagator = DynamicsPropagator(
            dim_model=dim_model, steps=self.model_steps
        )

        coord_dim = 2 * self.spatial_bands * 2 + 1 * self.time_bands * 2

        self.decoder_sensor = Decoder(
            ff_channels=coord_dim,
            preproc_ch=dim_model,
            num_latent_channels=dim_model,
            latent_size=1,
            num_output_channels=2,
        )

        self.decoder_feature_dim = 128
        img_coord_dim = 2 * self.spatial_bands * 2

        self.decoder_image = Decoder(
            ff_channels=img_coord_dim,
            preproc_ch=dim_model,
            num_latent_channels=dim_model,
            latent_size=1,
            num_output_channels=self.decoder_feature_dim,
        )

        self.refinement_head = get_refinement_head(
            head_type=image_head_type,
            in_channels=self.decoder_feature_dim,
            out_channels=1,
            upsampling_type=upsampling_type,
        )

        self.sensor_loss_fn = nn.MSELoss()

        if image_loss_type == "l1":
            self.image_loss_fn = nn.L1Loss()
            self.image_weight_factor = 1.0
        elif image_loss_type == "mse":
            self.image_loss_fn = nn.MSELoss()
            self.image_weight_factor = 1.0

        self.register_buffer(
            "decode_times",
            torch.linspace(0.01, 1.0, self.model_steps).reshape(
                1, self.model_steps, 1, 1
            ),
        )
        self.register_buffer("smoke_mean", torch.tensor(smoke_mean))
        self.register_buffer("smoke_std", torch.tensor(smoke_std))

    def forward(self, batch):
        # (B, N, dim_model)
        tokens = self.tokenizer(
            batch["sensor_history_vals"], batch["sensor_pos"], batch["current_image"]
        )
        # (B, latents, dim_model)
        z_0 = self.encoder(tokens)
        # (B, model_steps, latents, dim_model)
        z_traj = self.propagator(z_0)

        B = z_traj.shape[0]
        pos_enc = sparse_fourier_encode(batch["sensor_pos"], self.spatial_bands, 64.0)
        time_enc = sparse_fourier_encode(self.decode_times, self.time_bands, 50.0)
        # (1, model_steps, 1, 32) -> (B, model_steps, n_sensor, 32)
        time_enc = time_enc.unsqueeze(2).expand(B, -1, 20, -1)
        # (B, n_sensor, 64) -> (B, model_steps, n_sensor, 64)
        pos_enc_expanded = pos_enc.unsqueeze(1).expand(-1, self.model_steps, -1, -1)
        # (B, model_steps, n_sensor, 64+32)
        sensor_coords = torch.cat([pos_enc_expanded, time_enc], dim=-1)
        # (B * model_steps, n_latents, dim_model)
        z_flat = z_traj.reshape(B * self.model_steps, -1, self.hparams.dim_model)
        # (B * model_steps, n_sensor, 64+32)
        coords_flat = sensor_coords.reshape(B * self.model_steps, 20, -1)
        # (B * model_steps, n_sensor, 2 (u,v))
        sensor_preds = self.decoder_sensor(z_flat, coords_flat)
        sensor_preds = sensor_preds.reshape(B, self.model_steps, 20, 2).transpose(1, 2)

        z_final = z_traj[:, -1, :, :]

        grid_h = 128 // self.patch_size
        grid_w = 192 // self.patch_size

        ys = torch.linspace(-1, 1, grid_h, device=self.device)
        xs = torch.linspace(-1.5, 1.5, grid_w, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # (grid_h, grid_w, 2) -> (1, grid_h * grid_w, 2) -> (B, grid_h * grid_w, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2).expand(B, -1, -1)
        # (B, num_patches, 2*spatial_bands*2)
        patch_coords = sparse_fourier_encode(grid, self.spatial_bands, 64.0)
        # (B, num_patches, p*p)
        patch_features = self.decoder_image(z_final, patch_coords)

        feature_map = patch_features.reshape(
            B, grid_h, grid_w, self.decoder_feature_dim
        ).permute(0, 3, 1, 2)
        img_pred = self.refinement_head(feature_map)

        return sensor_preds, img_pred

    def training_step(self, batch, batch_idx):
        sensor_preds, img_pred = self.forward(batch)
        sensor_loss = self.sensor_loss_fn(sensor_preds, batch["sensor_target_vals"])
        image_loss = self.image_loss_fn(img_pred, batch["target_image"])
        loss = sensor_loss + self.image_weight_factor * image_loss
        self.log("mode", 1.0, on_epoch=True)
        self.log(
            "train_image_loss",
            image_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        self.log("train_loss_step", loss, on_step=True, prog_bar=False, logger=False)

        self.log(
            "train_sensor_loss",
            sensor_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log("train_total_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sensor_preds, img_pred = self.forward(batch)

        sensor_loss = self.sensor_loss_fn(sensor_preds, batch["sensor_target_vals"])

        image_loss = self.image_loss_fn(img_pred, batch["target_image"])
        loss = sensor_loss + self.image_weight_factor * image_loss
        self.log("val_image_loss", image_loss, on_epoch=True, on_step=False)

        self.log("val_loss_step", loss, on_step=True, prog_bar=False, logger=False)

        self.log("val_sensor_loss", sensor_loss, on_epoch=True, on_step=False)
        self.log("val_total_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

        img_pred_denorm = img_pred * self.smoke_std + self.smoke_mean
        target_denorm = batch["target_image"] * self.smoke_std + self.smoke_mean

        img_pred_denorm = torch.clamp(img_pred_denorm, 0.0, 1.0)
        target_denorm = torch.clamp(target_denorm, 0.0, 1.0)

        mass_pred = img_pred_denorm.sum()
        mass_target = target_denorm.sum()
        mass_error_pct = torch.abs(mass_pred - mass_target) / (mass_target + 1e-6)

        psnr_value = self.psnr(img_pred_denorm, target_denorm)
        ssim_value = self.ssim(img_pred_denorm, target_denorm)
        self.log("val_mass_error_pct", mass_error_pct, on_epoch=True, on_step=False)
        self.log("val_psnr", psnr_value, on_epoch=True, on_step=False)
        self.log("val_ssim", ssim_value, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]
