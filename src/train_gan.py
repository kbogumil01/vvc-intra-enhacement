#!/usr/bin/env python3
"""
GAN fine-tuning script for VVC-PPFF.

Loads a pretrained PPFF generator and fine-tunes it with adversarial training.
Validation uses MSE-only for model selection (best PSNR checkpoint).

Example:
    python train_gan.py \
        --generator-ckpt lightning_logs/lightning_logs/version_93/checkpoints/dense-best-*.ckpt \
        --decoded-root /net/tscratch/people/plgkarolbogumil/data/decoded_BVI_frames_YUV420_10bit_nofilter \
        --original-root /net/tscratch/people/plgkarolbogumil/data/splitted_DVI_BVC \
        --memmap-dir /net/tscratch/people/plgkarolbogumil/data/memmap_cache_nofilter \
        --lambda-adv 0.005
"""
from __future__ import annotations

import argparse
import os
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from dataset import (
    VVCFrameDataset,
    VVCCachedDataset,
    VVCMmapDataset,
    build_records,
    train_val_split_by_sequence,
)
from dataset_temporal import VVCTemporalMmapDataset
from vvc_ppff_gan import PPFFGANModule, GANConfig

import model_v2
import importlib
importlib.import_module("vvc-ppff")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GAN fine-tune PPFF generator")

    p.add_argument("--decoded-root", type=str,
                    default="/net/tscratch/people/plgkarolbogumil/data/decoded_BVI_frames_YUV420_10bit_nofilter")
    p.add_argument("--original-root", type=str,
                    default="/net/tscratch/people/plgkarolbogumil/data/splitted_DVI_BVC")
    p.add_argument("--extra-decoded-root", type=str, nargs="+", default=[])
    p.add_argument("--extra-original-root", type=str, nargs="+", default=[])
    p.add_argument("--qp-list", type=int, nargs="+", default=[22, 27, 32, 37])
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--patch-size", type=int, default=240)
    p.add_argument("--patches-per-frame", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--memmap-dir", type=str, default=None)
    p.add_argument("--cache-in-ram", action="store_true")
    p.add_argument("--temporal-memmap-dir", type=str, default=None,
                    help="Path to temporal memmap (12ch mode). Overrides --memmap-dir.")

    p.add_argument("--generator-ckpt", type=str, required=True,
                    help="Pretrained PPFF checkpoint (Lightning .ckpt)")
    p.add_argument("--generator-model", type=str, default="ppff_base",
                    help="Generator model name from registry")
    p.add_argument("--in-channels", type=int, default=None,
                    help="Generator input channels (auto: 4 for memmap, 12 for temporal)")

    p.add_argument("--lambda-pixel", type=float, default=1.0,
                    help="Pixel (MSE) loss weight")
    p.add_argument("--lambda-perceptual", type=float, default=0.01,
                    help="VGG perceptual loss weight (0 to disable)")
    p.add_argument("--lambda-adv", type=float, default=0.01,
                    help="Adversarial loss weight")
    p.add_argument("--lambda-r1", type=float, default=10.0,
                    help="R1 gradient penalty weight")
    p.add_argument("--lr-g", type=float, default=1e-5,
                    help="Generator learning rate (lower for fine-tuning)")
    p.add_argument("--lr-d", type=float, default=2e-4,
                    help="Discriminator learning rate")
    p.add_argument("--warmup-epochs", type=int, default=0,
                    help="Epochs of pure MSE before adversarial loss kicks in")
    p.add_argument("--d-type", type=str, default="ppff", choices=["ppff", "patch"],
                    help="Discriminator type: ppff (deep, 2.7M params) or patch (shallow)")
    p.add_argument("--ndf", type=int, default=64,
                    help="Discriminator base channels (patch type)")
    p.add_argument("--d-layers", type=int, default=3,
                    help="Number of discriminator layers (patch type)")
    p.add_argument("--d-num-blocks", type=int, default=16,
                    help="Number of FEB blocks (ppff type)")
    p.add_argument("--d-channels", type=int, default=128,
                    help="Channel width (ppff type)")
    p.add_argument("--d-conditional", action="store_true",
                    help="Conditional D: input = cat(decoded_yuv, output_yuv) = 6ch")
    p.add_argument("--d-no-spectral-norm", action="store_true",
                    help="Disable spectral norm in PPFFDiscriminator")
    p.add_argument("--d-pretrain-epochs", type=int, default=0,
                    help="Pretrain D on decoded-vs-original before GAN training")
    p.add_argument("--d-checkpoint", type=str, default=None,
                    help="Load pretrained discriminator weights from standalone D training")
    p.add_argument("--gan-loss-type", type=str, default="ragan",
                    choices=["lsgan", "ragan"],
                    help="GAN loss type: lsgan (standard) or ragan (relativistic average, ESRGAN-style)")
    p.add_argument("--r1-every", type=int, default=16,
                    help="Apply R1 penalty every N discriminator steps")

    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--precision", type=str, default="32",
                    choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--accumulate-grad-batches", type=int, default=6)
    p.add_argument("--log-dir", type=str, default="lightning_logs")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    import numpy as np

    if args.temporal_memmap_dir:
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

        train_seqs = set(r["sequence_id"] for r in train_recs)
        val_seqs = set(r["sequence_id"] for r in val_recs)
        assert len(train_seqs & val_seqs) == 0, "Train/val overlap!"

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
        num_workers = 0
        in_channels = args.in_channels or 12

    elif args.memmap_dir:
        print("Building record list ...")
        records = build_records(
            decoded_root=args.decoded_root,
            original_root=args.original_root,
            qp_list=args.qp_list,
        )
        print(f"  Main dataset: {len(records)} pairs")

        if args.extra_decoded_root:
            if len(args.extra_decoded_root) != len(args.extra_original_root):
                raise ValueError("--extra-decoded-root and --extra-original-root must have same length")
            for dec, orig in zip(args.extra_decoded_root, args.extra_original_root):
                extra = build_records(decoded_root=dec, original_root=orig, qp_list=args.qp_list)
                print(f"  Extra dataset ({dec}): {len(extra)} pairs")
                records.extend(extra)

        print(f"  Total: {len(records)} pairs")
        train_recs, val_recs = train_val_split_by_sequence(
            records, val_ratio=args.val_ratio, seed=args.split_seed
        )
        print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

        print(f"  [memmap] Loading from: {args.memmap_dir}")
        idx_path = os.path.join(args.memmap_dir, "index.json")
        dat_path = os.path.join(args.memmap_dir, "frames.dat")
        with open(idx_path) as f:
            index = json.load(f)
        total = index["total_elems"]
        print(f"  Loading {total * 2 / 1e9:.0f} GB into RAM ...")
        data = np.fromfile(dat_path, dtype=np.float16)
        assert data.size == total, f"Size mismatch: {data.size} != {total}"
        print(f"  Loaded {data.nbytes / 1e9:.0f} GB into RAM")

        train_ds = VVCMmapDataset(
            memmap_dir=args.memmap_dir, records=train_recs,
            patch_size=args.patch_size, patches_per_frame=args.patches_per_frame,
            training=True, data_array=data,
        )
        val_ds = VVCMmapDataset(
            memmap_dir=args.memmap_dir, records=val_recs,
            patch_size=args.patch_size, patches_per_frame=1,
            training=False, data_array=data,
        )
        num_workers = 0
        in_channels = args.in_channels or 4
    else:
        print("Building record list ...")
        records = build_records(
            decoded_root=args.decoded_root,
            original_root=args.original_root,
            qp_list=args.qp_list,
        )
        if args.extra_decoded_root:
            if len(args.extra_decoded_root) != len(args.extra_original_root):
                raise ValueError("--extra-decoded-root and --extra-original-root must have same length")
            for dec, orig in zip(args.extra_decoded_root, args.extra_original_root):
                extra = build_records(decoded_root=dec, original_root=orig, qp_list=args.qp_list)
                records.extend(extra)

        train_recs, val_recs = train_val_split_by_sequence(
            records, val_ratio=args.val_ratio, seed=args.split_seed
        )
        print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

        DatasetClass = VVCCachedDataset if args.cache_in_ram else VVCFrameDataset
        if args.cache_in_ram:
            print("  [cache-in-ram] Loading all frames into RAM ...")
        train_ds = DatasetClass(
            train_recs, patch_size=args.patch_size,
            patches_per_frame=args.patches_per_frame, training=True,
        )
        val_ds = DatasetClass(
            val_recs, patch_size=args.patch_size,
            patches_per_frame=1, training=False,
        )
        in_channels = args.in_channels or 4
        if args.cache_in_ram:
            num_workers = 2
        else:
            num_workers = args.num_workers

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    gan_cfg = GANConfig(
        generator_name=args.generator_model,
        generator_in_channels=in_channels,
        lambda_pixel=args.lambda_pixel,
        lambda_perceptual=args.lambda_perceptual,
        lambda_adv=args.lambda_adv,
        lambda_r1=args.lambda_r1,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        warmup_epochs=args.warmup_epochs,
        d_pretrain_epochs=args.d_pretrain_epochs,
        d_type=args.d_type,
        ndf=args.ndf,
        d_layers=args.d_layers,
        d_num_blocks=args.d_num_blocks,
        d_channels=args.d_channels,
        d_conditional=args.d_conditional,
        d_spectral_norm=not args.d_no_spectral_norm,
        gan_loss_type=args.gan_loss_type,
        r1_every=args.r1_every,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    lit = PPFFGANModule(gan_config=gan_cfg)

    print(f"Loading pretrained generator from: {args.generator_ckpt}")
    ckpt = torch.load(args.generator_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    gen_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            gen_state[k.replace("model.", "", 1)] = v
    missing, unexpected = lit.generator.load_state_dict(gen_state, strict=False)
    if missing:
        print(f"  WARNING: missing keys in generator: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")
    print(f"  Generator loaded ({sum(p.numel() for p in lit.generator.parameters()):,} params)")
    print(f"  Discriminator created ({sum(p.numel() for p in lit.discriminator.parameters()):,} params)")

    if args.d_checkpoint:
        print(f"Loading pretrained discriminator from: {args.d_checkpoint}")
        d_ckpt = torch.load(args.d_checkpoint, map_location="cpu", weights_only=False)
        d_state = d_ckpt["state_dict"]
        d_weights = {}
        for k, v in d_state.items():
            if k.startswith("discriminator."):
                d_weights[k.replace("discriminator.", "", 1)] = v
        d_missing, d_unexpected = lit.discriminator.load_state_dict(d_weights, strict=False)
        if d_missing:
            print(f"  WARNING: missing D keys: {d_missing}")
        if d_unexpected:
            print(f"  WARNING: unexpected D keys: {d_unexpected}")
        print(f"  Discriminator loaded from pretrained checkpoint")

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            filename="gan-best-{epoch:03d}-{val_loss:.5f}",
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
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
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        log_every_n_steps=50,
    )

    trainer.fit(lit, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
