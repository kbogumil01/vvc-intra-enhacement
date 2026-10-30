#!/usr/bin/env python3
"""
Pre-process temporal VVC dataset into numpy memmap files.

Creates a memmap with all frames (curr rec, curr ref, prev rec, next rec)
and an index with temporal links for each record.

Input data:
  - BVI curr (nofilter): decoded_BVI_frames_YUV420_10bit_nofilter/
  - BVI originals: splitted_DVI_BVC/
  - BVI neighbors (nofilter): dograna_data/decoded_BVI_neighbors_nofilter/
  - XIPH curr (nofilter): dograna_data/decoded_XIPH_nofilter/
  - XIPH originals: dograna_data/splitted_XIPH/
  (XIPH neighbors are in the same decoded_XIPH_nofilter dir)

Model input (12ch):
  [YUV_curr(3), QP(1), binary_luma(1), binary_chroma(1), YUV_prev(3), YUV_next(3)]

Output:
  {output_dir}/
    frames.dat           — flat float16 memmap (YUV444 [3,H,W] per frame)
    index.json           — file offsets + temporal records
    partition_frames.dat — flat float16 memmap (boundary maps [2,H,W])
    partition_index.json — partition offsets per rec_path

Usage (Phase 1, no 4K):
  python preprocess_temporal_memmap.py \
    --bvi-decoded /net/tscratch/.../decoded_BVI_frames_YUV420_10bit_nofilter \
    --bvi-original /net/tscratch/.../splitted_DVI_BVC \
    --bvi-neighbors-decoded /net/tscratch/.../dograna_data/decoded_BVI_neighbors_nofilter \
    --xiph-decoded /net/tscratch/.../dograna_data/decoded_XIPH_nofilter \
    --xiph-original /net/tscratch/.../dograna_data/splitted_XIPH \
    --output-dir /net/tscratch/.../temporal_memmap_nofilter \
    --no-4k
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_records, VVCFrameDataset

TARGET_FRAMES = [4, 19, 34, 49, 60]
NEIGHBOR_OFFSETS = [-1, +1]
PARTITION_CHANNELS = 2
PARTITION_MAP_NAMES = ["boundary_map.raw", "boundary_chroma_map.raw"]


def find_neighbor_path(
    rec_path: str,
    frame_id: int,
    offset: int,
    neighbors_decoded_root: str,
) -> Optional[str]:
    """
    Given a curr rec_path like .../QP22/seq_frame4/rec.yuv,
    find the neighbor rec.yuv in neighbors_decoded_root.
    """
    frame_dir = Path(rec_path).parent
    qp_dir_name = frame_dir.parent.name
    frame_name = frame_dir.name

    m = re.match(r"^(.+)_frame(\d+)$", frame_name)
    if not m:
        return None
    seq_base = m.group(1)
    nb_idx = frame_id + offset
    nb_frame_name = f"{seq_base}_frame{nb_idx}"
    nb_path = Path(neighbors_decoded_root) / qp_dir_name / nb_frame_name / "rec.yuv"
    if nb_path.exists():
        return str(nb_path)
    return None


def find_xiph_neighbor_path(
    rec_path: str,
    frame_id: int,
    offset: int,
) -> Optional[str]:
    """
    For XIPH, neighbors are in the same decoded dir as curr.
    .../QP22/seq_frame4/rec.yuv -> .../QP22/seq_frame3/rec.yuv
    """
    frame_dir = Path(rec_path).parent
    qp_dir = frame_dir.parent
    frame_name = frame_dir.name

    m = re.match(r"^(.+)_frame(\d+)$", frame_name)
    if not m:
        return None
    seq_base = m.group(1)
    nb_idx = frame_id + offset
    nb_path = qp_dir / f"{seq_base}_frame{nb_idx}" / "rec.yuv"
    if nb_path.exists():
        return str(nb_path)
    return None


def build_temporal_records(
    bvi_decoded: str,
    bvi_original: str,
    bvi_neighbors_decoded: str,
    xiph_decoded: str,
    xiph_original: str,
    qp_list: List[int],
    no_4k: bool = False,
    only_4k: bool = False,
) -> List[Dict[str, Any]]:
    """Build records with temporal links (prev_path, next_path)."""
    temporal_records = []

    if bvi_decoded and bvi_original:
        bvi_recs = build_records(bvi_decoded, bvi_original, qp_list=qp_list)
        print(f"  BVI base records: {len(bvi_recs)}")

        if no_4k:
            bvi_recs = [r for r in bvi_recs if r["width"] != 3840]
        elif only_4k:
            bvi_recs = [r for r in bvi_recs if r["width"] == 3840]

        print(f"  BVI filtered records: {len(bvi_recs)}")

        for r in bvi_recs:
            prev_path = find_neighbor_path(
                r["rec_path"], r["frame_id"], -1, bvi_neighbors_decoded
            )
            next_path = find_neighbor_path(
                r["rec_path"], r["frame_id"], +1, bvi_neighbors_decoded
            )
            if prev_path and next_path:
                r["prev_path"] = prev_path
                r["next_path"] = next_path
                temporal_records.append(r)

    if xiph_decoded and xiph_original:
        xiph_recs = build_records(xiph_decoded, xiph_original, qp_list=qp_list)
        print(f"  XIPH base records: {len(xiph_recs)}")

        xiph_curr = [r for r in xiph_recs if r["frame_id"] in TARGET_FRAMES]
        print(f"  XIPH curr records: {len(xiph_curr)}")

        for r in xiph_curr:
            prev_path = find_xiph_neighbor_path(r["rec_path"], r["frame_id"], -1)
            next_path = find_xiph_neighbor_path(r["rec_path"], r["frame_id"], +1)
            if prev_path and next_path:
                r["prev_path"] = prev_path
                r["next_path"] = next_path
                temporal_records.append(r)

    print(f"  Total temporal records (with both neighbors): {len(temporal_records)}")
    return temporal_records


def main():
    p = argparse.ArgumentParser(description="Pre-process temporal VVC dataset to memmap")
    p.add_argument("--bvi-decoded", type=str, default="")
    p.add_argument("--bvi-original", type=str, default="")
    p.add_argument("--bvi-neighbors-decoded", type=str, default="")
    p.add_argument("--xiph-decoded", type=str, default="")
    p.add_argument("--xiph-original", type=str, default="")
    p.add_argument("--qp-list", type=int, nargs="+", default=[22, 27, 32, 37])
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--bit-depth", type=int, default=10)
    p.add_argument("--no-4k", action="store_true")
    p.add_argument("--only-4k", action="store_true")
    args = p.parse_args()

    print("=== Building temporal records ===")
    all_records = build_temporal_records(
        bvi_decoded=args.bvi_decoded,
        bvi_original=args.bvi_original,
        bvi_neighbors_decoded=args.bvi_neighbors_decoded,
        xiph_decoded=args.xiph_decoded,
        xiph_original=args.xiph_original,
        qp_list=args.qp_list,
        no_4k=args.no_4k,
        only_4k=args.only_4k,
    )

    if not all_records:
        print("ERROR: No records found!")
        sys.exit(1)

    print("\n=== Collecting unique files ===")
    file_info: Dict[str, Tuple[int, int]] = {}
    for r in all_records:
        h, w = r["height"], r["width"]
        file_info[r["rec_path"]] = (h, w)
        file_info[r["ref_path"]] = (h, w)
        file_info[r["prev_path"]] = (h, w)
        file_info[r["next_path"]] = (h, w)

    unique_files = sorted(file_info.keys())
    n_files = len(unique_files)

    offsets: Dict[str, int] = {}
    total_elems = 0
    for path in unique_files:
        offsets[path] = total_elems
        h, w = file_info[path]
        total_elems += 3 * h * w

    total_bytes = total_elems * 2
    print(f"  Unique files: {n_files}")
    print(f"  Frames memmap size: {total_bytes / 1e9:.1f} GB")

    print("\n=== Collecting partition maps ===")
    rec_paths_with_maps: Dict[str, Tuple[int, int, str]] = {}
    missing_maps = 0
    for r in all_records:
        rec_path = r["rec_path"]
        if rec_path in rec_paths_with_maps:
            continue
        frame_dir = str(Path(rec_path).parent)
        has_all = all(
            os.path.exists(os.path.join(frame_dir, mn))
            for mn in PARTITION_MAP_NAMES
        )
        if has_all:
            rec_paths_with_maps[rec_path] = (r["height"], r["width"], frame_dir)
        else:
            missing_maps += 1
            if missing_maps <= 3:
                print(f"    MISSING maps: {frame_dir}")

    if missing_maps:
        print(f"  WARNING: {missing_maps} rec frames missing partition maps")
        all_records = [r for r in all_records if r["rec_path"] in rec_paths_with_maps]
        print(f"  Records after filter: {len(all_records)}")

    unique_recs_for_parts = sorted(rec_paths_with_maps.keys())
    n_part_files = len(unique_recs_for_parts)

    part_offsets: Dict[str, int] = {}
    part_total_elems = 0
    for rec_path in unique_recs_for_parts:
        part_offsets[rec_path] = part_total_elems
        h, w, _ = rec_paths_with_maps[rec_path]
        part_total_elems += PARTITION_CHANNELS * h * w

    part_bytes = part_total_elems * 2
    print(f"  Partition files: {n_part_files}")
    print(f"  Partition memmap size: {part_bytes / 1e9:.1f} GB")

    print("\n=== Creating frames memmap ===")
    os.makedirs(args.output_dir, exist_ok=True)
    dat_path = os.path.join(args.output_dir, "frames.dat")
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

        mmap[offset:offset + n] = frame.reshape(-1).astype(np.float16)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_files - i - 1) / rate
            print(f"  Frames: {i+1}/{n_files} ({rate:.1f} files/s, ETA {eta/60:.0f} min)")

    mmap.flush()
    del mmap
    elapsed = time.time() - t0
    print(f"  Frames done in {elapsed/60:.1f} min")

    print("\n=== Creating partition memmap ===")
    pdat_path = os.path.join(args.output_dir, "partition_frames.dat")
    pmmap = np.memmap(pdat_path, dtype=np.float16, mode="w+", shape=(part_total_elems,))

    t0 = time.time()
    for i, rec_path in enumerate(unique_recs_for_parts):
        h, w, frame_dir = rec_paths_with_maps[rec_path]
        offset = part_offsets[rec_path]
        n = PARTITION_CHANNELS * h * w

        channels = []
        for mname in PARTITION_MAP_NAMES:
            ch = np.fromfile(
                os.path.join(frame_dir, mname), dtype=np.float16
            ).reshape(h, w)
            channels.append(ch)

        frame = np.stack(channels, axis=0)
        pmmap[offset:offset + n] = frame.reshape(-1)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_part_files - i - 1) / rate
            print(f"  Partitions: {i+1}/{n_part_files} ({rate:.0f} files/s, ETA {eta/60:.0f} min)")

    pmmap.flush()
    del pmmap
    elapsed = time.time() - t0
    print(f"  Partitions done in {elapsed/60:.1f} min")

    print("\n=== Saving indices ===")

    file_index = {}
    for path in unique_files:
        h, w = file_info[path]
        file_index[path] = {"offset": offsets[path], "height": h, "width": w}

    partition_file_index = {}
    for rec_path in unique_recs_for_parts:
        h, w, _ = rec_paths_with_maps[rec_path]
        partition_file_index[rec_path] = {
            "offset": part_offsets[rec_path], "height": h, "width": w
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
                "prev_path": r["prev_path"],
                "next_path": r["next_path"],
                "width": r["width"],
                "height": r["height"],
                "qp": r["qp"],
                "sequence_id": r["sequence_id"],
                "frame_id": r["frame_id"],
            }
            for r in all_records
        ],
    }

    idx_path = os.path.join(args.output_dir, "index.json")
    with open(idx_path, "w") as f:
        json.dump(index_data, f)

    partition_index_data = {
        "total_elems": part_total_elems,
        "n_channels": PARTITION_CHANNELS,
        "channel_names": ["boundary", "boundary_chroma"],
        "n_files": n_part_files,
        "file_index": partition_file_index,
    }

    pidx_path = os.path.join(args.output_dir, "partition_index.json")
    with open(pidx_path, "w") as f:
        json.dump(partition_index_data, f)

    print(f"\n=== DONE ===")
    print(f"  Records: {len(all_records)}")
    print(f"  Frames: {dat_path} ({os.path.getsize(dat_path) / 1e9:.1f} GB)")
    print(f"  Partitions: {pdat_path} ({os.path.getsize(pdat_path) / 1e9:.1f} GB)")
    print(f"  Index: {idx_path}")
    print(f"  Partition index: {pidx_path}")


if __name__ == "__main__":
    main()
