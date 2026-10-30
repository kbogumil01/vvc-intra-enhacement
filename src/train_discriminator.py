#!/usr/bin/env python3
"""
Standalone Discriminator Training.

Train a conditional PPFFDiscriminator as a binary classifier:
  Real: cat(decoded_yuv, original_yuv) → 1
  Fake (easy): cat(decoded_yuv, decoded_yuv) → 0
  Fake (hard): cat(decoded_yuv, G(input)) → 0

This produces a strong D checkpoint that can be loaded into GAN training.
Stable training: standard classification (not adversarial), with spectral norm.

Usage:
    python train_discriminator.py \
        --temporal-memmap-dir /path/to/temporal_memmap_nofilter \
        --generator-ckpt path/to/generator.ckpt \
        --max-epochs 30
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from dataset import train_val_split_by_sequence
from dataset_temporal import VVCTemporalMmapDataset
from vvc_ppff_gan import PPFFDiscriminator, lsgan_loss_d

import model_v2
import importlib
importlib.import_module("vvc-ppff")

from registry import create_model


class DiscriminatorClassifier(pl.LightningModule):
    """
    Train PPFFDiscriminator as a standalone classifier.
    
    Each batch produces 3 pairs:
      - Real: cat(decoded, original) → target=1
      - Fake easy: cat(decoded, decoded) → target=0  
      - Fake hard: cat(decoded, G(input)) → target=0
    
    Loss: LSGAN-style MSE (same as will be used in GAN training).
    """

    def __init__(
        self,
        generator_ckpt: str,
        generator_name: str = "ppff_base",
        generator_in_channels: int = 12,
        d_num_blocks: int = 16,
        d_channels: int = 128,
        lr: float = 2e-4,
        hard_negative_ratio: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.hard_negative_ratio = hard_negative_ratio

        self.discriminator = PPFFDiscriminator(
            in_channels=6,
            num_blocks=d_num_blocks,
            channels=d_channels,
            use_spectral_norm=True,
        )

        self.generator = create_model(generator_name, in_channels=generator_in_channels)
        self._load_generator(generator_ckpt)
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False

    def _load_generator(self, ckpt_path: str):
        """Load pretrained generator weights."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        gen_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                gen_state[k.replace("model.", "", 1)] = v
        missing, unexpected = self.generator.load_state_dict(gen_state, strict=False)
        if missing:
            print(f"  WARNING: missing keys in generator: {missing}")
        if unexpected:
            print(f"  WARNING: unexpected keys: {unexpected}")
        print(f"  Generator loaded for hard negative generation "
              f"({sum(p.numel() for p in self.generator.parameters()):,} params)")

    def forward(self, x):
        return self.discriminator(x)

    def _compute_step(self, batch, stage: str):
        x = batch["input"]
        y = batch["target"]
        decoded = x[:, :3, :, :]

        B = x.shape[0]

        real_input = torch.cat([decoded, y], dim=1)
        real_pred = self.discriminator(real_input)

        use_hard = torch.rand(B, device=x.device) < self.hard_negative_ratio

        easy_input = torch.cat([decoded, decoded], dim=1)

        with torch.no_grad():
            g_output = self.generator(x)
            g_output = torch.clamp(g_output, 0.0, 1.0)
        hard_input = torch.cat([decoded, g_output], dim=1)

        use_hard_expanded = use_hard.view(B, 1, 1, 1).expand_as(easy_input)
        fake_input = torch.where(use_hard_expanded, hard_input, easy_input)
        fake_pred = self.discriminator(fake_input)

        loss = lsgan_loss_d(real_pred, fake_pred)

        with torch.no_grad():
            acc_real = (real_pred > 0.5).float().mean()
            acc_fake = (fake_pred < 0.5).float().mean()
            acc_total = (acc_real + acc_fake) / 2.0

            if use_hard.any():
                hard_only = torch.cat([decoded[use_hard], g_output[use_hard]], dim=1)
                hard_pred = self.discriminator(hard_only)
                acc_hard = (hard_pred < 0.5).float().mean()
            else:
                acc_hard = torch.tensor(0.0)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/acc_real", acc_real, on_epoch=True)
        self.log(f"{stage}/acc_fake", acc_fake, on_epoch=True)
        self.log(f"{stage}/acc_total", acc_total, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/acc_hard", acc_hard, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(0.0, 0.99),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": sched}


def parse_args():
    p = argparse.ArgumentParser(description="Standalone discriminator training")

    p.add_argument("--temporal-memmap-dir", type=str, required=True)
    p.add_argument("--generator-ckpt", type=str, required=True)
    p.add_argument("--generator-model", type=str, default="ppff_base")
    p.add_argument("--in-channels", type=int, default=12)

    p.add_argument("--d-num-blocks", type=int, default=16)
    p.add_argument("--d-channels", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hard-negative-ratio", type=float, default=0.5,
                   help="Fraction of fake samples that are G outputs (vs decoded)")

    p.add_argument("--patch-size", type=int, default=240)
    p.add_argument("--patches-per-frame", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--log-dir", type=str, default="lightning_logs")

    return p.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    print(f"  [temporal] Loading from: {args.temporal_memmap_dir}")
    idx_path = os.path.join(args.temporal_memmap_dir, "index.json")
    with open(idx_path) as f:
        index = json.load(f)

    all_records = index["records"]
    print(f"  Total temporal records: {len(all_records)}")

    train_recs, val_recs = train_val_split_by_sequence(
        all_records, val_ratio=args.val_ratio, seed=args.split_seed
    )
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    dat_path = os.path.join(args.temporal_memmap_dir, "frames.dat")
    total = index["total_elems"]
    print(f"  Loading frames ({total * 2 / 1e9:.0f} GB) ...")
    data = np.fromfile(dat_path, dtype=np.float16)
    assert data.size == total

    pidx_path = os.path.join(args.temporal_memmap_dir, "partition_index.json")
    with open(pidx_path) as f:
        pindex = json.load(f)
    pdat_path = os.path.join(args.temporal_memmap_dir, "partition_frames.dat")
    ptotal = pindex["total_elems"]
    print(f"  Loading partitions ({ptotal * 2 / 1e9:.1f} GB) ...")
    pdata = np.fromfile(pdat_path, dtype=np.float16)
    assert pdata.size == ptotal

    train_ds = VVCTemporalMmapDataset(
        memmap_dir=args.temporal_memmap_dir, records=train_recs,
        patch_size=args.patch_size, patches_per_frame=args.patches_per_frame,
        training=True, data_array=data, partition_data_array=pdata,
        temporal_mode="both",
    )
    val_ds = VVCTemporalMmapDataset(
        memmap_dir=args.temporal_memmap_dir, records=val_recs,
        patch_size=args.patch_size, patches_per_frame=1,
        training=False, data_array=data, partition_data_array=pdata,
        temporal_mode="both",
    )

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = DiscriminatorClassifier(
        generator_ckpt=args.generator_ckpt,
        generator_name=args.generator_model,
        generator_in_channels=args.in_channels,
        d_num_blocks=args.d_num_blocks,
        d_channels=args.d_channels,
        lr=args.lr,
        hard_negative_ratio=args.hard_negative_ratio,
    )

    d_params = sum(p.numel() for p in model.discriminator.parameters())
    print(f"  Discriminator: {d_params:,} params")

    callbacks = [
        ModelCheckpoint(
            monitor="val/acc_total",
            mode="max",
            save_top_k=2,
            save_last=True,
            filename="d-best-{epoch:03d}-{val/acc_total:.4f}",
        ),
        EarlyStopping(
            monitor="val/acc_total",
            mode="max",
            patience=args.early_stop_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        default_root_dir=args.log_dir,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"\n{'='*60}")
    print(f"Best D checkpoint: {best_path}")
    print(f"  val/acc_total = {trainer.checkpoint_callback.best_model_score:.4f}")
    print(f"{'='*60}")
    print(f"\nTo use in GAN training:")
    print(f"  python train_gan.py --d-checkpoint {best_path} ...")


if __name__ == "__main__":
    main()
