import argparse
import os

import pytorch_lightning as pl
from dataset import SmokeDataset
from model import MultirateTemporalSenseiver
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader


def main(args):
    # Weird workaround
    if "SLURM_NTASKS" in os.environ:
        del os.environ["SLURM_NTASKS"]
        del os.environ["SLURM_JOB_NAME"]
    pl.seed_everything(args.seed)

    print(f"\n--- Loading Data from {args.data_root} ---")

    train_ds = SmokeDataset(
        args.data_root, downsample_factor=args.downsample_factor, split="train"
    )
    val_ds = SmokeDataset(
        args.data_root, downsample_factor=args.downsample_factor, split="val"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    print(
        f"\n--- Initializing Model (Latents={args.num_latents}, Dim={args.dim_model}) ---"
    )
    model = MultirateTemporalSenseiver(
        dim_model=args.dim_model,
        num_latents=args.num_latents,
        downsample_factor=args.downsample_factor,
        smoke_mean=train_ds.smoke_mean.item(),
        smoke_std=train_ds.smoke_std.item(),
        image_head_type=args.image_head_type,
        upsampling_type=args.upsampling_type,
        image_loss_type=args.image_loss_type,
        max_epochs=args.max_epochs,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        dirpath=f"checkpoints/{args.run_name}",
        filename="mts-{epoch:02d}-{val_total_loss:.4f}",
        save_top_k=3,
        mode="min",
    )
    early_stop = EarlyStopping(
        monitor="val_total_loss", patience=5, verbose=True, mode="min"
    )

    bar = TQDMProgressBar(refresh_rate=1)

    logger = CSVLogger("lightning_logs", name=args.run_name)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop, bar],
        precision="bf16-mixed",
        accumulate_grad_batches=2,
    )

    trainer.fit(model, train_loader, val_loader)


def modify_parser(parser):
    parser.add_argument(
        "--data_root",
        type=str,
        default="smoke_simulation_data",
        help="Path to the generated simulation data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--dim_model",
        type=int,
        default=256,
        help="Dimension of the transformer/latent space",
    )
    parser.add_argument(
        "--num_latents",
        type=int,
        default=128,
        help="Number of latent queries (bottleneck size)",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=4,
        help="Factor to downsample the temporal dimension",
    )

    parser.add_argument(
        "--image_head_type",
        type=str,
        default="baseline",
        help="Type of image refinement head: baseline, 2_step_upsampling, 4_step_upsampling",
    )
    parser.add_argument(
        "--upsampling_type",
        type=str,
        default="bilinear",
        help="Type of upsampling: bilinear, nearest",
    )
    parser.add_argument(
        "--image_loss_type",
        type=str,
        default="l1",
        help="Type of image loss: l1, mse",
    )

    parser.add_argument("--max_epochs", type=int, default=8, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--run_name",
        type=str,
        default="multirate_run_test",
        help="Name for logging and checkpoints",
    )

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modify_parser(parser)
    args = parser.parse_args()
    main(args)
