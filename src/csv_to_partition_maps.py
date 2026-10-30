#!/usr/bin/env python3
"""
Parse VTM block-statistics CSV and produce partition maps (raw float16).

Output files (all in same dir as CSV):
  LUMA (shape = luma H × W):
    - depth_map.raw      : Depth / 7.0 per CU pixel             [0, 1]
    - boundary_map.raw   : 1.0 at inner border of each CU       {0, 1}
  CHROMA (shape = luma H × W, coords in luma space):
    - depth_chroma_map.raw    : Depth_Chroma / 10.0 per CU pixel [0, 1]
    - boundary_chroma_map.raw : 1.0 at inner border of each CU   {0, 1}

Boundary convention: each CU marks 1.0 on ALL four inner edges (top row,
bottom row, left col, right col).  Two adjacent CUs produce two adjacent
rows/columns of 1s — one from each block's inner border.

Depth normalization:
  Luma:   Depth / max(Depth)       — normalized to [0, 1] per frame.
  Chroma: Depth_Chroma / max(Depth_Chroma) — normalized to [0, 1] per frame.

Usage (single file):
    python csv_to_partition_maps.py --csv path/to/bitstream_partition.csv

Usage (batch — all CSVs under a root):
    python csv_to_partition_maps.py --root /path/to/decoded_root --workers 16
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def parse_sequence_size(csv_path: str) -> Tuple[int, int]:
    """Extract (width, height) from CSV header line: # Sequence size: [WxH]"""
    with open(csv_path, "r") as f:
        for line in f:
            if "Sequence size" in line:
                m = re.search(r"\[\s*(\d+)\s*x\s*(\d+)\s*\]", line)
                if m:
                    return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot find Sequence size in {csv_path}")


def parse_info_file(info_path: str) -> Tuple[int, int]:
    """Extract (width, height) from rec.yuv.info."""
    w, h = None, None
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("width="):
                w = int(line.split("=", 1)[1])
            elif line.startswith("height="):
                h = int(line.split("=", 1)[1])
    if w is None or h is None:
        raise ValueError(f"Cannot parse dimensions from {info_path}")
    return w, h


def process_csv(csv_path: str, force: bool = False) -> Optional[str]:
    """
    Read a partition CSV, produce 4 raw maps (luma depth/boundary + chroma depth/boundary).
    Returns None on success, error string on failure.
    """
    out_dir = os.path.dirname(csv_path)

    output_files = [
        os.path.join(out_dir, "depth_map.raw"),
        os.path.join(out_dir, "boundary_map.raw"),
        os.path.join(out_dir, "depth_chroma_map.raw"),
        os.path.join(out_dir, "boundary_chroma_map.raw"),
    ]

    if not force and all(os.path.exists(f) for f in output_files):
        return None

    info_path = os.path.join(out_dir, "rec.yuv.info")
    if os.path.exists(info_path):
        try:
            width, height = parse_info_file(info_path)
        except ValueError:
            width, height = parse_sequence_size(csv_path)
    else:
        try:
            width, height = parse_sequence_size(csv_path)
        except ValueError as e:
            return str(e)


    luma_depth = {}
    chroma_depth = {}

    target_stats = {"Depth", "Depth_Chroma"}

    with open(csv_path, "r") as f:
        for line in f:
            if not line.startswith("BlockStat"):
                continue

            parts = line.rstrip().split(";")
            if len(parts) < 8:
                continue

            stat_name = parts[6]
            if stat_name not in target_stats:
                continue

            x = int(parts[2])
            y = int(parts[3])
            w = int(parts[4])
            h = int(parts[5])
            val = int(parts[7])

            key = (x, y, w, h)
            if stat_name == "Depth":
                luma_depth[key] = val
            else:
                chroma_depth[key] = val

    def build_maps(cu_dict):
        """Build depth and boundary maps from a CU dictionary."""
        depth_map = np.zeros((height, width), dtype=np.float32)
        boundary_map = np.zeros((height, width), dtype=np.float32)

        for (x, y, w, h), d_val in cu_dict.items():
            x_end = min(x + w, width)
            y_end = min(y + h, height)
            x = max(x, 0)
            y = max(y, 0)

            if x_end <= x or y_end <= y:
                continue

            depth_map[y:y_end, x:x_end] = d_val

            boundary_map[y, x:x_end] = 1.0
            if y_end - 1 >= y:
                boundary_map[y_end - 1, x:x_end] = 1.0
            boundary_map[y:y_end, x] = 1.0
            if x_end - 1 >= x:
                boundary_map[y:y_end, x_end - 1] = 1.0

        d_max = depth_map.max()
        if d_max > 0:
            depth_map /= d_max

        return depth_map, boundary_map

    l_depth, l_boundary = build_maps(luma_depth)
    l_depth.astype(np.float16).tofile(output_files[0])
    l_boundary.astype(np.float16).tofile(output_files[1])

    c_depth, c_boundary = build_maps(chroma_depth)
    c_depth.astype(np.float16).tofile(output_files[2])
    c_boundary.astype(np.float16).tofile(output_files[3])

    return None


def main():
    p = argparse.ArgumentParser(description="Convert VTM partition CSV to raw maps")
    p.add_argument("--csv", type=str, help="Single CSV file to process")
    p.add_argument("--root", type=str, help="Root dir to scan for bitstream_partition.csv")
    p.add_argument("--workers", type=int, default=16, help="Parallel workers for batch mode")
    p.add_argument("--force", action="store_true", help="Overwrite existing maps")
    args = p.parse_args()

    if args.csv:
        err = process_csv(args.csv, force=args.force)
        if err:
            print(f"ERROR: {err}", file=sys.stderr)
            sys.exit(1)
        print("Done.")
        return

    if not args.root:
        p.error("Either --csv or --root is required")

    csv_files = sorted(glob.glob(
        os.path.join(args.root, "**", "bitstream_partition.csv"),
        recursive=True,
    ))
    print(f"Found {len(csv_files)} CSV files under {args.root}")

    if not csv_files:
        return

    done = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_csv, f, args.force): f for f in csv_files}
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result:
                errors += 1
                print(f"ERROR: {futures[fut]}: {result}", file=sys.stderr)
            if done % 1000 == 0:
                print(f"  {done}/{len(csv_files)}  (errors: {errors})")

    print(f"\nDone: {done}/{len(csv_files)}, errors: {errors}")


if __name__ == "__main__":
    main()
