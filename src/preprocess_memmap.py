#!/usr/bin/env python3
"""
Pre-process VVC dataset into a single numpy memmap file.

Reads all unique YUV420 frames, upsamples chroma to 444,
stores as float16 [3, H, W] packed sequentially (no padding).

Output:
  {output_dir}/
    frames.dat    — flat float16 memmap, frames packed end-to-end
    index.json    — offset/shape per file + training records

Usage:
    python preprocess_memmap.py \
      --decoded-root /path/to/decoded \
      --original-root /path/to/original \
      --output-dir /path/to/cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_records, VVCFrameDataset


def main():
    p = argparse.ArgumentParser(description="Pre-process VVC dataset to memmap")
    p.add_argument("--decoded-root", type=str, nargs="+", required=True)
    p.add_argument("--original-root", type=str, nargs="+", required=True)
    p.add_argument("--qp-list", type=int, nargs="+", default=[22, 27, 32, 37])
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--bit-depth", type=int, default=10)
    args = p.parse_args()

    assert len(args.decoded_root) == len(args.original_root)

    all_records: List[Dict[str, Any]] = []
    for dec, orig in zip(args.decoded_root, args.original_root):
        recs = build_records(dec, orig, qp_list=args.qp_list)
        print(f"  {dec}: {len(recs)} records")
        all_records.extend(recs)
    print(f"Total: {len(all_records)} records")

    file_info: Dict[str, Tuple[int, int]] = {}
    for r in all_records:
        file_info[r["rec_path"]] = (r["height"], r["width"])
        file_info[r["ref_path"]] = (r["height"], r["width"])

    unique_files = sorted(file_info.keys())
    n_files = len(unique_files)

    offsets: Dict[str, int] = {}
    total_elems = 0
    for path in unique_files:
        offsets[path] = total_elems
        h, w = file_info[path]
        total_elems += 3 * h * w

    total_bytes = total_elems * 2
    print(f"Unique files: {n_files}")
    print(f"Memmap size: {total_bytes / 1e9:.1f} GB (float16, packed)")

    os.makedirs(args.output_dir, exist_ok=True)
    dat_path = os.path.join(args.output_dir, "frames.dat")
    idx_path = os.path.join(args.output_dir, "index.json")

    mmap = np.memmap(dat_path, dtype=np.float16, mode="w+", shape=(total_elems,))

    max_val = float((1 << args.bit_depth) - 1)

    t0 = time.time()
    for i, path in enumerate(unique_files):
        h, w = file_info[path]
        offset = offsets[path]
        n = 3 * h * w

        y, u, v = VVCFrameDataset._read_yuv420(path, h, w)

        u_up = cv2.resize(u.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        v_up = cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        np.clip(u_up, 0, max_val, out=u_up)
        np.clip(v_up, 0, max_val, out=v_up)

        frame = np.empty((3, h, w), dtype=np.float32)
        frame[0] = y.astype(np.float32) / max_val
        frame[1] = u_up / max_val
        frame[2] = v_up / max_val

        mmap[offset : offset + n] = frame.reshape(-1).astype(np.float16)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_files - i - 1) / rate
            print(f"  {i+1}/{n_files} ({rate:.0f} files/s, ETA {eta/60:.0f} min)")

    mmap.flush()
    del mmap
    elapsed = time.time() - t0
    print(f"Preprocessing done in {elapsed/60:.1f} min")

    file_index = {}
    for path in unique_files:
        h, w = file_info[path]
        file_index[path] = {
            "offset": offsets[path],
            "height": h,
            "width": w,
        }

    index_data = {
        "total_elems": total_elems,
        "bit_depth": args.bit_depth,
        "n_files": n_files,
        "file_index": file_index,
        "records": [
            {
                "rec_path": r["rec_path"],
                "ref_path": r["ref_path"],
                "width": r["width"],
                "height": r["height"],
                "qp": r["qp"],
                "sequence_id": r["sequence_id"],
                "frame_id": r["frame_id"],
            }
            for r in all_records
        ],
    }

    with open(idx_path, "w") as f:
        json.dump(index_data, f)

    print(f"Saved: {dat_path} ({os.path.getsize(dat_path) / 1e9:.1f} GB)")
    print(f"Index: {idx_path}")


if __name__ == "__main__":
    main()
