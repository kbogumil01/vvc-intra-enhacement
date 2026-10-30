#!/usr/bin/env python3
"""
Verify partition maps have correct dimensions and visualize samples.
Run after fix_and_preprocess_partition_slurm.sh completes.
"""

import argparse
import json
import os
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = "/net/tscratch/people/plgkarolbogumil/data"


def verify_raw_files():
    """Check that all 4 partition map files match rec.yuv.info dims."""
    print("=" * 60)
    print("1. Verifying raw file dimensions vs rec.yuv.info")
    print("=" * 60)

    MAP_NAMES = ["depth_map.raw", "boundary_map.raw",
                 "depth_chroma_map.raw", "boundary_chroma_map.raw"]
    mismatches = 0
    total = 0

    for variant in ["", "_nofilter", "_dbonly"]:
        for ds in [f"decoded_BVI_frames_YUV420_10bit{variant}",
                    f"decoded_DIV2K_YUV420_10bit{variant}",
                    f"decoded_CTC_YUV420_10bit{variant}"]:
            root = os.path.join(DATA, ds)
            ds_total = 0
            ds_mismatch = 0
            for info_path in sorted(glob.glob(os.path.join(root, "QP*", "*", "rec.yuv.info"))):
                d = {}
                for line in open(info_path):
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        d[k] = v
                w, h = int(d["width"]), int(d["height"])
                frame_dir = os.path.dirname(info_path)

                for map_name in MAP_NAMES:
                    map_path = os.path.join(frame_dir, map_name)
                    if not os.path.exists(map_path):
                        continue
                    raw_size = os.path.getsize(map_path) // 2
                    expected = w * h
                    ds_total += 1
                    total += 1
                    if raw_size != expected:
                        ds_mismatch += 1
                        mismatches += 1
                        if mismatches <= 5:
                            print(f"  MISMATCH: {map_path}  {w}x{h}  raw={raw_size}  expected={expected}")

            status = "OK" if ds_mismatch == 0 else f"FAIL ({ds_mismatch} mismatches)"
            print(f"  {ds}: {ds_total // 4} frames checked — {status}")

    print(f"\nTotal: {total} maps, {mismatches} mismatches")
    return mismatches == 0


def verify_memmap_index():
    """Check that partition memmap index keys match frames memmap index keys."""
    print("\n" + "=" * 60)
    print("2. Verifying partition memmap index consistency")
    print("=" * 60)

    ok = True
    for variant in ["", "_nofilter", "_dbonly"]:
        frames_dir = os.path.join(DATA, f"memmap_cache{variant}")
        part_dir = os.path.join(DATA, f"partition_memmap_cache_v2{variant}")

        frames_idx_path = os.path.join(frames_dir, "index.json")
        part_idx_path = os.path.join(part_dir, "partition_index.json")

        if not os.path.exists(part_idx_path):
            print(f"  MISSING: {part_idx_path}")
            ok = False
            continue

        if not os.path.exists(frames_idx_path):
            print(f"  MISSING frames index: {frames_idx_path}")
            ok = False
            continue

        with open(frames_idx_path) as f:
            frames_raw = json.load(f)
        with open(part_idx_path) as f:
            part_raw = json.load(f)

        frames_fi = frames_raw.get("file_index", {})
        part_fi = part_raw.get("file_index", {})

        frames_keys = set(frames_fi.keys())
        part_keys = set(part_fi.keys())

        common = frames_keys & part_keys
        only_frames = frames_keys - part_keys
        only_part = part_keys - frames_keys

        label = f"memmap_cache{variant}"
        print(f"  {label}: frames={len(frames_keys)} partition={len(part_keys)} "
              f"common={len(common)} only_frames={len(only_frames)} only_part={len(only_part)}")

        if only_frames:
            print(f"    WARNING: {len(only_frames)} keys in frames but not in partition")
            for k in list(only_frames)[:3]:
                print(f"      {k}")
        if only_part:
            print(f"    WARNING: {len(only_part)} keys in partition but not in frames")
            ok = False

        dim_mismatches = 0
        for key in list(common)[:100]:
            f_entry = frames_fi[key]
            p_entry = part_fi[key]
            f_h, f_w = f_entry["height"], f_entry["width"]
            p_h, p_w = p_entry["height"], p_entry["width"]
            if f_h != p_h or f_w != p_w:
                dim_mismatches += 1
                if dim_mismatches <= 3:
                    print(f"    DIM MISMATCH: {key}  frames={f_w}x{f_h}  partition={p_w}x{p_h}")
        if dim_mismatches:
            print(f"    {dim_mismatches}/100 sampled keys have dimension mismatches!")
            ok = False
        else:
            print(f"    Dimensions consistent (100 sampled)")

    return ok


def verify_memmap_data():
    """Spot-check that partition memmap data is valid (in range, not all zeros)."""
    print("\n" + "=" * 60)
    print("3. Verifying partition memmap data quality")
    print("=" * 60)

    ok = True
    N_CHANNELS = 4
    CHANNEL_NAMES = ["depth", "boundary", "depth_chroma", "boundary_chroma"]
    for variant in ["", "_nofilter", "_dbonly"]:
        part_dir = os.path.join(DATA, f"partition_memmap_cache_v2{variant}")
        dat_path = os.path.join(part_dir, "partition_frames.dat")
        idx_path = os.path.join(part_dir, "partition_index.json")

        if not os.path.exists(dat_path) or not os.path.exists(idx_path):
            print(f"  SKIP: {part_dir} (missing files)")
            ok = False
            continue

        with open(idx_path) as f:
            idx_raw = json.load(f)

        idx = idx_raw.get("file_index", {})
        dat = np.memmap(dat_path, dtype=np.float16, mode="r")
        label = f"partition_memmap_cache_v2{variant}"

        keys = list(idx.keys())
        np.random.seed(42)
        sample_keys = np.random.choice(keys, min(20, len(keys)), replace=False)

        all_zero = 0
        out_of_range = 0
        for key in sample_keys:
            entry = idx[key]
            offset = entry["offset"]
            h, w = entry["height"], entry["width"]
            n_pixels = N_CHANNELS * h * w
            chunk = np.array(dat[offset: offset + n_pixels]).reshape(N_CHANNELS, h, w)

            if all(chunk[c].max() == 0 for c in range(N_CHANNELS)):
                all_zero += 1
            for c in range(N_CHANNELS):
                if chunk[c].min() < 0 or chunk[c].max() > 1.0:
                    out_of_range += 1
                    if out_of_range <= 3:
                        print(f"    OUT OF RANGE: {key} ch={CHANNEL_NAMES[c]} "
                              f"min={chunk[c].min():.4f} max={chunk[c].max():.4f}")

        print(f"  {label}: {len(idx)} entries, dat={os.path.getsize(dat_path) / 1e9:.1f} GB")
        print(f"    Sampled {len(sample_keys)}: all_zero={all_zero}, out_of_range={out_of_range}")
        if all_zero > 5:
            print(f"    WARNING: many all-zero frames!")
            ok = False
        if out_of_range > 0:
            print(f"    WARNING: values out of [0,1] range!")
            ok = False

    return ok


def visualize(output_dir):
    """Visualize partition maps from memmap for sample frames."""
    print("\n" + "=" * 60)
    print("4. Generating visualization")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    frames_dir = os.path.join(DATA, "memmap_cache")
    part_dir = os.path.join(DATA, "partition_memmap_cache_v2")

    frames_idx_path = os.path.join(frames_dir, "index.json")
    part_idx_path = os.path.join(part_dir, "partition_index.json")

    if not os.path.exists(part_idx_path):
        print("  Partition memmap not ready, skipping visualization")
        return

    with open(frames_idx_path) as f:
        frames_idx = json.load(f).get("file_index", {})
    with open(part_idx_path) as f:
        part_idx = json.load(f).get("file_index", {})

    frames_dat = np.memmap(os.path.join(frames_dir, "frames.dat"), dtype=np.float16, mode="r")
    part_dat = np.memmap(os.path.join(part_dir, "partition_frames.dat"), dtype=np.float16, mode="r")

    ctc_keys = [k for k in part_idx.keys() if "decoded_CTC" in k]
    from collections import defaultdict
    seq_map = defaultdict(list)
    for k in ctc_keys:
        parts = k.split("/")
        for p in parts:
            if "QP" in p:
                qp = p
            if any(c.isdigit() for c in p) and "x" in p.lower():
                pass
        seq_name = parts[-2] if parts[-1] == "rec.yuv" else parts[-1]
        seq_base = seq_name.rsplit("_frame", 1)[0] if "_frame" in seq_name else seq_name
        seq_map[seq_base].append(k)

    target_seqs = ["BasketballPass_416x240_50", "BlowingBubbles_416x240_50",
                   "RaceHorses_416x240_30", "BQSquare_416x240_60"]

    for seq_base in target_seqs:
        keys = seq_map.get(seq_base, [])
        if not keys:
            continue

        qp_keys = {}
        for k in keys:
            for qp in [22, 27, 32, 37]:
                if f"QP{qp}" in k:
                    qp_keys[qp] = k
                    break

        if len(qp_keys) < 2:
            continue

        fig, axes = plt.subplots(len(qp_keys), 4, figsize=(16, 4 * len(qp_keys)))
        if len(qp_keys) == 1:
            axes = axes[np.newaxis, :]

        for row, qp in enumerate(sorted(qp_keys.keys())):
            key = qp_keys[qp]

            f_entry = frames_idx[key]
            f_off = f_entry["offset"]
            f_h, f_w = f_entry["height"], f_entry["width"]
            f_n = 4 * f_h * f_w
            frame = np.array(frames_dat[f_off: f_off + f_n]).reshape(4, f_h, f_w)
            y_chan = frame[0]

            p_entry = part_idx[key]
            p_off = p_entry["offset"]
            p_h, p_w = p_entry["height"], p_entry["width"]
            p_n = 4 * p_h * p_w
            part = np.array(part_dat[p_off: p_off + p_n]).reshape(4, p_h, p_w)
            depth = part[0]
            boundary = part[1]
            depth_chroma = part[2]
            boundary_chroma = part[3]

            assert f_h == p_h and f_w == p_w, \
                f"Dimension mismatch for {key}: frame={f_w}x{f_h} partition={p_w}x{p_h}"

            axes[row, 0].imshow(y_chan, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"QP{qp} — Y channel")
            axes[row, 0].axis("off")

            im_d = axes[row, 1].imshow(depth, cmap="viridis", vmin=0, vmax=1)
            axes[row, 1].set_title(f"QP{qp} — Luma depth")
            axes[row, 1].axis("off")
            plt.colorbar(im_d, ax=axes[row, 1], fraction=0.046)

            axes[row, 2].imshow(boundary, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title(f"QP{qp} — Luma boundary")
            axes[row, 2].axis("off")

            overlay = np.stack([y_chan, y_chan, y_chan], axis=-1)
            overlay = np.clip(overlay, 0, 1).astype(np.float32)
            mask = boundary > 0.5
            overlay[mask] = [1.0, 0.2, 0.2]
            axes[row, 3].imshow(overlay)
            axes[row, 3].set_title(f"QP{qp} — Y + luma bounds")
            axes[row, 3].axis("off")

        fig.suptitle(f"{seq_base} — Partition maps v2 (luma)", fontsize=14, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{seq_base}_partition_v2_luma.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    div2k_keys = [k for k in part_idx.keys() if "DIV2K" in k]
    if div2k_keys:
        np.random.seed(123)
        sample = np.random.choice(div2k_keys, min(4, len(div2k_keys)), replace=False)

        fig, axes = plt.subplots(len(sample), 6, figsize=(24, 4 * len(sample)))
        if len(sample) == 1:
            axes = axes[np.newaxis, :]

        for row, key in enumerate(sample):
            f_entry = frames_idx.get(key)
            p_entry = part_idx[key]
            if f_entry is None:
                continue

            f_off = f_entry["offset"]
            f_h, f_w = f_entry["height"], f_entry["width"]
            frame = np.array(frames_dat[f_off: f_off + 4 * f_h * f_w]).reshape(4, f_h, f_w)
            y_chan = frame[0]

            p_off = p_entry["offset"]
            p_h, p_w = p_entry["height"], p_entry["width"]
            part = np.array(part_dat[p_off: p_off + 4 * p_h * p_w]).reshape(4, p_h, p_w)
            depth = part[0]
            boundary = part[1]
            depth_chroma = part[2]
            boundary_chroma = part[3]

            assert f_h == p_h and f_w == p_w, \
                f"Dimension mismatch for {key}: frame={f_w}x{f_h} partition={p_w}x{p_h}"

            seq_name = key.split("/")[-2] if key.endswith("rec.yuv") else os.path.basename(key)

            axes[row, 0].imshow(y_chan, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"{seq_name} — Y ({f_w}x{f_h})")
            axes[row, 0].axis("off")

            im_d = axes[row, 1].imshow(depth, cmap="viridis", vmin=0, vmax=1)
            axes[row, 1].set_title(f"Luma depth")
            axes[row, 1].axis("off")
            plt.colorbar(im_d, ax=axes[row, 1], fraction=0.046)

            axes[row, 2].imshow(boundary, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title(f"Luma boundary")
            axes[row, 2].axis("off")

            im_dc = axes[row, 3].imshow(depth_chroma, cmap="viridis", vmin=0, vmax=1)
            axes[row, 3].set_title(f"Chroma depth")
            axes[row, 3].axis("off")
            plt.colorbar(im_dc, ax=axes[row, 3], fraction=0.046)

            axes[row, 4].imshow(boundary_chroma, cmap="gray", vmin=0, vmax=1)
            axes[row, 4].set_title(f"Chroma boundary")
            axes[row, 4].axis("off")

            overlay = np.stack([y_chan, y_chan, y_chan], axis=-1)
            overlay = np.clip(overlay, 0, 1).astype(np.float32)
            mask = boundary > 0.5
            overlay[mask] = [1.0, 0.2, 0.2]
            mask_c = boundary_chroma > 0.5
            overlay[mask_c] = [0.2, 0.2, 1.0]
            axes[row, 5].imshow(overlay)
            axes[row, 5].set_title(f"Y + bounds (red=luma, blue=chroma)")
            axes[row, 5].axis("off")

        fig.suptitle("DIV2K — Partition maps v2 (4ch)", fontsize=14, y=1.01)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "DIV2K_partition_v2.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Verify and visualize partition data")
    p.add_argument("--viz-dir", default="viz_partition_verify",
                   help="Directory for visualization output")
    p.add_argument("--skip-viz", action="store_true", help="Skip visualization")
    args = p.parse_args()

    raw_ok = verify_raw_files()
    idx_ok = verify_memmap_index()
    data_ok = verify_memmap_data()

    if not args.skip_viz:
        visualize(args.viz_dir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Raw file dimensions:    {'PASS' if raw_ok else 'FAIL'}")
    print(f"  Memmap index consistency: {'PASS' if idx_ok else 'FAIL'}")
    print(f"  Memmap data quality:    {'PASS' if data_ok else 'FAIL'}")

    if raw_ok and idx_ok and data_ok:
        print("\n  ALL CHECKS PASSED — data is ready for training")
    else:
        print("\n  SOME CHECKS FAILED — investigate before training")
        sys.exit(1)


if __name__ == "__main__":
    main()
