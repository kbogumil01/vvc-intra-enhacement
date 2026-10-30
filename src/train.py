#!/usr/bin/env python3
"""
Skrypt treningowy DenseNet VVC post-processing.

Uruchomienie:
    python train.py

Opcjonalne argumenty — patrz --help.
"""
from __future__ import annotations

import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

from dataset import VVCFrameDataset, VVCCachedDataset, VVCMmapDataset, build_records, train_val_split_by_sequence
from registry import create_lightning_module, list_models

import model_v2

import model_wcdann
import importlib
importlib.import_module("vvc-ppff")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DenseNet VVC enhancer")

    p.add_argument(
        "--decoded-root",
        type=str,
        default="/path/to/decoded_root",
    )
    p.add_argument(
        "--original-root",
        type=str,
        default="/path/to/original_root",
    )
    p.add_argument(
        "--extra-decoded-root",
        type=str,
        nargs="+",
        default=[],
        help="Dodatkowe decoded roots (np. DIV2K). Musi być tyle samo co --extra-original-root.",
    )
    p.add_argument(
        "--extra-original-root",
        type=str,
        nargs="+",
        default=[],
        help="Dodatkowe original roots odpowiadające --extra-decoded-root.",
    )
    p.add_argument(
        "--qp-list",
        type=int,
        nargs="+",
        default=[22, 27, 32, 37],
    )
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--patch-size", type=int, default=240)
    p.add_argument("--patches-per-frame", type=int, default=4,
                    help="Ile patchy losować z jednej klatki na epokę")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--model", type=str, default="dense_base",
                    help="Nazwa modelu z registry (patrz --list-models)")
    p.add_argument("--list-models", action="store_true",
                    help="Wypisz dostępne modele i zakończ")

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--early-stop-patience", type=int, default=50)

    p.add_argument("--precision", type=str, default="32",
                    choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--accumulate-grad-batches", type=int, default=6)
    p.add_argument("--log-dir", type=str, default="lightning_logs")
    p.add_argument("--project-name", type=str, default="dense_vvc")
    p.add_argument("--ckpt-path", type=str, default=None,
                    help="Ścieżka do checkpointu — ładuje TYLKO wagi (bez optymizatora/schedulera)")
    p.add_argument("--resume-path", type=str, default=None,
                    help="Ścieżka do checkpointu — pełne wznowienie (wagi+optymizator+scheduler+epoka)")
    p.add_argument("--cache-in-ram", action="store_true",
                    help="Pre-load all frames into RAM (fast training, needs ~320GB RAM)")
    p.add_argument("--memmap-dir", type=str, default=None,
                    help="Path to preprocessed memmap directory (from preprocess_memmap.py)")
    p.add_argument("--partition-memmap-dir", type=str, default=None,
                    help="Path to partition memmap directory (from preprocess_partition_memmap.py). "
                         "Adds luma+chroma depth/boundary channels to model input (8ch instead of 4ch).")
    p.add_argument("--partition-channels", type=int, nargs="+", default=None,
                    help="Which partition channels to use (0=luma_depth, 1=luma_boundary, "
                         "2=chroma_depth, 3=chroma_boundary). Default: all 4. "
                         "E.g. --partition-channels 0 for luma_depth only (5ch model).")
    p.add_argument("--in-channels", type=int, default=None,
                    help="Override model in_channels (default: 4 without partitions, 8 with)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("Dostępne modele:")
        for name in list_models():
            entry = __import__("registry").MODEL_REGISTRY[name]
            cfg = entry.config_cls(**entry.default_kwargs)
            model = entry.model_cls(cfg)
            n = sum(p.numel() for p in model.parameters())
            print(f"  {name:<20s} {n:>10,} params")
        return

    torch.set_float32_matmul_precision("high")

    print("Budowanie listy rekordów …")
    records = build_records(
        decoded_root=args.decoded_root,
        original_root=args.original_root,
        qp_list=args.qp_list,
    )
    print(f"  Główny zbiór: {len(records)} par (rec, ref)")

    if args.extra_decoded_root:
        if len(args.extra_decoded_root) != len(args.extra_original_root):
            raise ValueError("--extra-decoded-root i --extra-original-root muszą mieć tę samą liczbę argumentów")
        for dec, orig in zip(args.extra_decoded_root, args.extra_original_root):
            extra = build_records(
                decoded_root=dec,
                original_root=orig,
                qp_list=args.qp_list,
            )
            print(f"  Dodatkowy zbiór ({dec}): {len(extra)} par")
            records.extend(extra)

    print(f"  Łącznie: {len(records)} par (rec, ref)")

    train_recs, val_recs = train_val_split_by_sequence(
        records,
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    train_seqs = sorted(set(r["sequence_id"] for r in train_recs))
    val_seqs = sorted(set(r["sequence_id"] for r in val_recs))
    assert len(set(train_seqs) & set(val_seqs)) == 0, "Overlap train/val!"

    import json, os
    split_info = {
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "num_train_sequences": len(train_seqs),
        "num_val_sequences": len(val_seqs),
        "num_train_records": len(train_recs),
        "num_val_records": len(val_recs),
        "val_sequences": val_seqs,
    }
    split_path = os.path.join(args.log_dir, "split_info.json")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"  Zapisano podział do: {split_path}")
    print(f"  Sekwencje walidacyjne ({len(val_seqs)}):")
    for s in val_seqs:
        print(f"    {s}")

    DatasetClass = VVCCachedDataset if args.cache_in_ram else VVCFrameDataset
    if args.cache_in_ram:
        print("  [cache-in-ram] Pre-loading all frames into RAM ...")

    if args.memmap_dir:
        print(f"  [memmap] Using preprocessed data from: {args.memmap_dir}")
        import json as _json
        import numpy as _np
        _idx_path = os.path.join(args.memmap_dir, "index.json")
        _dat_path = os.path.join(args.memmap_dir, "frames.dat")
        with open(_idx_path) as _f:
            _index = _json.load(_f)
        _total = _index["total_elems"]
        print(f"  Loading {_total * 2 / 1e9:.0f} GB into RAM (one-time sequential read) ...")
        _data = _np.fromfile(_dat_path, dtype=_np.float16)
        assert _data.size == _total, f"Size mismatch: {_data.size} != {_total}"
        print(f"  Loaded {_data.nbytes / 1e9:.0f} GB into RAM")

        _pdata = None
        if args.partition_memmap_dir:
            print(f"  [partition] Loading partition memmap from: {args.partition_memmap_dir}")
            _pidx_path = os.path.join(args.partition_memmap_dir, "partition_index.json")
            _pdat_path = os.path.join(args.partition_memmap_dir, "partition_frames.dat")
            with open(_pidx_path) as _pf:
                _pindex = _json.load(_pf)
            _ptotal = _pindex["total_elems"]
            print(f"  Loading {_ptotal * 2 / 1e9:.0f} GB partition data into RAM ...")
            _pdata = _np.fromfile(_pdat_path, dtype=_np.float16)
            assert _pdata.size == _ptotal, f"Partition size mismatch: {_pdata.size} != {_ptotal}"
            print(f"  Loaded {_pdata.nbytes / 1e9:.0f} GB partition data into RAM")

        train_ds = VVCMmapDataset(
            memmap_dir=args.memmap_dir,
            records=train_recs,
            patch_size=args.patch_size,
            patches_per_frame=args.patches_per_frame,
            training=True,
            data_array=_data,
            partition_memmap_dir=args.partition_memmap_dir,
            partition_data_array=_pdata,
            partition_channels=args.partition_channels,
        )
        val_ds = VVCMmapDataset(
            memmap_dir=args.memmap_dir,
            records=val_recs,
            patch_size=args.patch_size,
            patches_per_frame=1,
            training=False,
            data_array=_data,
            partition_memmap_dir=args.partition_memmap_dir,
            partition_data_array=_pdata,
            partition_channels=args.partition_channels,
        )
    else:
        train_ds = DatasetClass(
            train_recs,
            patch_size=args.patch_size,
            patches_per_frame=args.patches_per_frame,
            training=True,
        )
        val_ds = DatasetClass(
            val_recs,
            patch_size=args.patch_size,
            patches_per_frame=1,
            training=False,
        )

    if args.memmap_dir:
        num_workers = 0
    elif args.cache_in_ram:
        num_workers = 2
    else:
        num_workers = args.num_workers

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model_name = args.model
    if model_name not in list_models():
        print(f"Nieznany model: {model_name}")
        print(f"Dostępne: {', '.join(list_models())}")
        return

    in_channels = args.in_channels
    if in_channels is None:
        if args.partition_memmap_dir:
            n_part = len(args.partition_channels) if args.partition_channels else 4
            in_channels = 4 + n_part
        else:
            in_channels = 4
    print(f"  Model in_channels: {in_channels}")
    if args.partition_channels:
        ch_names = ["luma_depth", "luma_boundary", "chroma_depth", "chroma_boundary"]
        print(f"  Partition channels: {[ch_names[i] for i in args.partition_channels]}")

    lit = create_lightning_module(model_name, lr=args.lr, in_channels=in_channels)

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        lit.load_state_dict(ckpt["state_dict"])
        print(f"  Załadowano wagi z: {args.ckpt_path}")

    n_params = sum(p.numel() for p in lit.parameters())
    print(f"  Parametry modelu: {n_params:,}")

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="dense-best-{epoch:03d}-{val_loss:.5f}",
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
