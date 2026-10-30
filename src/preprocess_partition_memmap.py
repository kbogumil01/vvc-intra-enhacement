#!/usr/bin/env python3
"""
Pre-process partition maps (luma + chroma depth/boundary) into a separate memmap.

Reads 4 raw maps per frame directory:
  depth_map.raw, boundary_map.raw, depth_chroma_map.raw, boundary_chroma_map.raw
Packs them as float16 [4, H, W] sequentially into a single flat file.

Index uses rec_path as key (same as frames memmap) so the dataset can
look up partition data by the same key.

Output:
  {output_dir}/
    partition_frames.dat  — flat float16 memmap
    partition_index.json  — offset/shape per rec_path

Usage:
    python preprocess_partition_memmap.py \
      --decoded-root /path/to/decoded1 /path/to/decoded2 \
      --original-root /path/to/original1 /path/to/original2 \
      --output-dir /path/to/partition_cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_records


def main():
    p = argparse.ArgumentParser(description="Pre-process partition maps to memmap")
    p.add_argument("--decoded-root", type=str, nargs="+", required=True)
    p.add_argument("--original-root", type=str, nargs="+", required=True)
    p.add_argument("--qp-list", type=int, nargs="+", default=[22, 27, 32, 37])
    p.add_argument("--output-dir", type=str, required=True)
    args = p.parse_args()

    assert len(args.decoded_root) == len(args.original_root)

    all_records: List[Dict[str, Any]] = []
    for dec, orig in zip(args.decoded_root, args.original_root):
        recs = build_records(dec, orig, qp_list=args.qp_list)
        print(f"  {dec}: {len(recs)} records")
        all_records.extend(recs)
    print(f"Total: {len(all_records)} records")

    rec_info: Dict[str, Tuple[int, int, str]] = {}
    for r in all_records:
        rec_path = r["rec_path"]
        if rec_path not in rec_info:
            frame_dir = str(Path(rec_path).parent)
            rec_info[rec_path] = (r["height"], r["width"], frame_dir)

    unique_recs = sorted(rec_info.keys())
    n_files = len(unique_recs)

    N_CHANNELS = 4
    MAP_NAMES = ["depth_map.raw", "boundary_map.raw",
                 "depth_chroma_map.raw", "boundary_chroma_map.raw"]

    missing = 0
    for rec_path in unique_recs:
        h, w, frame_dir = rec_info[rec_path]
        for mname in MAP_NAMES:
            if not os.path.exists(os.path.join(frame_dir, mname)):
                missing += 1
                if missing <= 5:
                    print(f"  MISSING: {frame_dir}/{mname}")
                break
    if missing:
        print(f"ERROR: {missing} frames missing partition maps. Run csv_to_partition_maps.py first.")
        sys.exit(1)

    offsets: Dict[str, int] = {}
    total_elems = 0
    for rec_path in unique_recs:
        offsets[rec_path] = total_elems
        h, w, _ = rec_info[rec_path]
        total_elems += N_CHANNELS * h * w

    total_bytes = total_elems * 2
    print(f"Unique decoded frames: {n_files}")
    print(f"Memmap size: {total_bytes / 1e9:.1f} GB (float16, {N_CHANNELS}ch)")

    os.makedirs(args.output_dir, exist_ok=True)
    dat_path = os.path.join(args.output_dir, "partition_frames.dat")
    idx_path = os.path.join(args.output_dir, "partition_index.json")

    mmap = np.memmap(dat_path, dtype=np.float16, mode="w+", shape=(total_elems,))

    t0 = time.time()
    for i, rec_path in enumerate(unique_recs):
        h, w, frame_dir = rec_info[rec_path]
        offset = offsets[rec_path]
        n = N_CHANNELS * h * w

        channels = []
        for mname in MAP_NAMES:
            ch = np.fromfile(
                os.path.join(frame_dir, mname),
                dtype=np.float16,
            ).reshape(h, w)
            channels.append(ch)

        frame = np.stack(channels, axis=0)
        mmap[offset:offset + n] = frame.reshape(-1)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_files - i - 1) / rate
            print(f"  {i+1}/{n_files} ({rate:.0f} files/s, ETA {eta/60:.0f} min)")

    mmap.flush()
    del mmap
    elapsed = time.time() - t0
    print(f"Preprocessing done in {elapsed/60:.1f} min")

    file_index = {}
    for rec_path in unique_recs:
        h, w, _ = rec_info[rec_path]
        file_index[rec_path] = {
            "offset": offsets[rec_path],
            "height": h,
            "width": w,
        }

    index_data = {
        "total_elems": total_elems,
        "n_channels": N_CHANNELS,
        "channel_names": ["depth", "boundary", "depth_chroma", "boundary_chroma"],
        "n_files": n_files,
        "file_index": file_index,
    }

    with open(idx_path, "w") as f:
        json.dump(index_data, f)

    print(f"Saved: {dat_path} ({os.path.getsize(dat_path) / 1e9:.1f} GB)")
    print(f"Index: {idx_path}")


if __name__ == "__main__":
    main()
