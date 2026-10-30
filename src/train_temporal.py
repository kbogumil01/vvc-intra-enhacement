#!/usr/bin/env python3
"""
Training script for temporal 12-channel VVC post-processing model.

Input: [YUV_curr(3), QP(1), boundary_luma(1), boundary_chroma(1), YUV_prev(3), YUV_next(3)]
Target: [YUV_ref(3)]

Uses preprocess_temporal_memmap.py output (frames.dat + partition_frames.dat + indices).

Usage:
    python train_temporal.py --memmap-dir /path/to/temporal_memmap_nofilter
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from dataset import train_val_split_by_sequence
from dataset_temporal import VVCTemporalMmapDataset
from registry import create_lightning_module, list_models

import model_v2
import model_wcdann
import importlib
importlib.import_module("vvc-ppff")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train temporal 12ch VVC enhancer")

    p.add_argument("--memmap-dir", type=str, required=True,
                   help="Path to temporal memmap directory (from preprocess_temporal_memmap.py)")

    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--patch-size", type=int, default=240)
    p.add_argument("--patches-per-frame", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--model", type=str, default="dense_base",
                   help="Model name from registry")
    p.add_argument("--in-channels", type=int, default=12,
                   help="Number of input channels (default 12 for temporal)")
    p.add_argument("--temporal-mode", type=str, default="both",
                   choices=["both", "prev_only", "next_only"],
                   help="Which neighbors to use: both (12ch), prev_only (9ch), next_only (9ch)")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--early-stop-patience", type=int, default=50)

    p.add_argument("--precision", type=str, default="32",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--accumulate-grad-batches", type=int, default=6)
    p.add_argument("--log-dir", type=str, default="lightning_logs")
    p.add_argument("--ckpt-path", type=str, default=None,
                   help="Load weights only (no optimizer/scheduler)")
    p.add_argument("--resume-path", type=str, default=None,
                   help="Full resume (weights + optimizer + scheduler + epoch)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    print("=== Loading temporal memmap index ===")
    idx_path = os.path.join(args.memmap_dir, "index.json")
    with open(idx_path) as f:
        index = json.load(f)

    all_records = index["records"]
    print(f"  Total records: {len(all_records)}")

    train_recs, val_recs = train_val_split_by_sequence(
        all_records,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    train_seqs = set(r["sequence_id"] for r in train_recs)
    val_seqs = set(r["sequence_id"] for r in val_recs)
    assert len(train_seqs & val_seqs) == 0, "Overlap between train/val sequences!"

    os.makedirs(args.log_dir, exist_ok=True)
    split_info = {
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "num_train_sequences": len(train_seqs),
        "num_val_sequences": len(val_seqs),
        "num_train_records": len(train_recs),
        "num_val_records": len(val_recs),
        "val_sequences": sorted(val_seqs),
    }
    split_path = os.path.join(args.log_dir, "temporal_split_info.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Split info saved: {split_path}")

    print("\n=== Loading memmap data into RAM ===")
    dat_path = os.path.join(args.memmap_dir, "frames.dat")
    total_elems = index["total_elems"]
    print(f"  Loading frames ({total_elems * 2 / 1e9:.0f} GB) ...")
    data_array = np.fromfile(dat_path, dtype=np.float16)
    assert data_array.size == total_elems
    print(f"  Loaded {data_array.nbytes / 1e9:.0f} GB")

    pidx_path = os.path.join(args.memmap_dir, "partition_index.json")
    with open(pidx_path) as f:
        pindex = json.load(f)
    pdat_path = os.path.join(args.memmap_dir, "partition_frames.dat")
    ptotal = pindex["total_elems"]
    print(f"  Loading partitions ({ptotal * 2 / 1e9:.1f} GB) ...")
    partition_array = np.fromfile(pdat_path, dtype=np.float16)
    assert partition_array.size == ptotal
    print(f"  Loaded {partition_array.nbytes / 1e9:.1f} GB")

    print(f"\n=== Creating datasets (temporal_mode={args.temporal_mode}) ===")
    train_ds = VVCTemporalMmapDataset(
        memmap_dir=args.memmap_dir,
        records=train_recs,
        patch_size=args.patch_size,
        patches_per_frame=args.patches_per_frame,
        training=True,
        data_array=data_array,
        partition_data_array=partition_array,
        temporal_mode=args.temporal_mode,
    )
    val_ds = VVCTemporalMmapDataset(
        memmap_dir=args.memmap_dir,
        records=val_recs,
        patch_size=args.patch_size,
        patches_per_frame=1,
        training=False,
        data_array=data_array,
        partition_data_array=partition_array,
        temporal_mode=args.temporal_mode,
    )
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"\n=== Model: {args.model}, in_channels={args.in_channels} ===")
    lit = create_lightning_module(
        args.model, lr=args.lr, in_channels=args.in_channels
    )

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        model_state = lit.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        lit.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  Skipped {len(skipped)} incompatible keys (e.g. {skipped[0]})")
        print(f"  Loaded {len(filtered)} keys from: {args.ckpt_path}")

    n_params = sum(p.numel() for p in lit.parameters())
    print(f"  Parameters: {n_params:,}")

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="temporal-best-{epoch:03d}-{val_loss:.5f}",
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
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    trainer.fit(
        lit,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=args.resume_path,
    )


if __name__ == "__main__":
    main()
