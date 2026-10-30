#!/usr/bin/env python3
"""
Visualize partition maps (luma/chroma depth + boundaries) alongside the decoded frame.
Generates a figure suitable for thesis insertion.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

MEMMAP_DIR = "/net/tscratch/people/plgkarolbogumil/data/memmap_cache_nofilter"
PART_DIR = "/net/tscratch/people/plgkarolbogumil/data/partition_memmap_cache_v2_nofilter"

TARGET_QP = "QP37"
TARGET_RES = (1088, 1920)
TARGET_SEQUENCE = "Basketball"

CROP_SIZE = 384

OUTPUT_PATH = "/net/people/plgrid/plgkarolbogumil/mgr/dense/partition_visualization.pdf"
OUTPUT_PNG = "/net/people/plgrid/plgkarolbogumil/mgr/dense/partition_visualization.png"


def load_indices():
    with open(f"{MEMMAP_DIR}/index.json") as f:
        frame_idx = json.load(f)
    with open(f"{PART_DIR}/partition_index.json") as f:
        part_idx = json.load(f)
    return frame_idx, part_idx


def find_frame(frame_idx, part_idx):
    """Find a suitable 1080p QP37 frame from target sequence."""
    for key, info in frame_idx["file_index"].items():
        if (TARGET_QP in key and 
            TARGET_SEQUENCE in key and
            info["height"] == TARGET_RES[0] and 
            info["width"] == TARGET_RES[1] and
            key in part_idx["file_index"]):
            return key, info
    for key, info in frame_idx["file_index"].items():
        if (TARGET_QP in key and 
            info["height"] == TARGET_RES[0] and 
            info["width"] == TARGET_RES[1] and
            key in part_idx["file_index"]):
            return key, info
    raise RuntimeError("No suitable frame found")


def read_frame_crop(dat_path, total_elems, info, cy, cx, size):
    """Read a [3, size, size] crop from memmap."""
    data = np.memmap(dat_path, dtype=np.float16, mode="r", shape=(total_elems,))
    h, w = info["height"], info["width"]
    offset = info["offset"]
    
    patch = np.empty((3, size, size), dtype=np.float32)
    for c in range(3):
        ch_offset = offset + c * h * w
        for row in range(size):
            start = ch_offset + (cy + row) * w + cx
            patch[c, row] = data[start:start + size].astype(np.float32)
    return patch


def read_partition_crop(dat_path, total_elems, info, cy, cx, size):
    """Read [4, size, size] partition crop from memmap."""
    data = np.memmap(dat_path, dtype=np.float16, mode="r", shape=(total_elems,))
    h, w = info["height"], info["width"]
    offset = info["offset"]
    
    patch = np.empty((4, size, size), dtype=np.float32)
    for c in range(4):
        ch_offset = offset + c * h * w
        for row in range(size):
            start = ch_offset + (cy + row) * w + cx
            patch[c, row] = data[start:start + size].astype(np.float32)
    return patch


def main():
    frame_idx, part_idx = load_indices()
    
    key, frame_info = find_frame(frame_idx, part_idx)
    part_info = part_idx["file_index"][key]
    
    print(f"Frame: {key}")
    print(f"  Size: {frame_info['height']}x{frame_info['width']}")
    
    h, w = frame_info["height"], frame_info["width"]
    cy = (h - CROP_SIZE) // 2
    cx = (w - CROP_SIZE) // 2
    cy = cy - (cy % 2)
    cx = cx - (cx % 2)
    
    print(f"  Crop: ({cy}, {cx}) size {CROP_SIZE}x{CROP_SIZE}")
    
    frame_total = frame_idx["total_elems"]
    part_total = part_idx["total_elems"]
    
    frame_crop = read_frame_crop(
        f"{MEMMAP_DIR}/frames.dat", frame_total, frame_info, cy, cx, CROP_SIZE
    )
    part_crop = read_partition_crop(
        f"{PART_DIR}/partition_frames.dat", part_total, part_info, cy, cx, CROP_SIZE
    )
    
    
    luma_y = frame_crop[0]
    luma_depth = part_crop[0]
    luma_boundary = part_crop[1]
    chroma_depth = part_crop[2]
    chroma_boundary = part_crop[3]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    
    ax = axes[0]
    y_rgb = np.stack([luma_y, luma_y, luma_y], axis=-1)
    mask = luma_boundary > 0.5
    overlay = y_rgb.copy()
    overlay[mask] = [1.0, 0.2, 0.2]
    ax.imshow(overlay)
    ax.set_title("(a) Luminancja Y\nz granicami CU", fontsize=10)
    ax.axis("off")
    
    ax = axes[1]
    im = ax.imshow(luma_depth, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("(b) Głębokość\npartycji – luma", fontsize=10)
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("d / 7", fontsize=8)
    
    ax = axes[2]
    ax.imshow(luma_boundary, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("(c) Granice CU\n– luma", fontsize=10)
    ax.axis("off")
    
    ax = axes[3]
    im = ax.imshow(chroma_depth, cmap="magma", vmin=0, vmax=1)
    ax.set_title("(d) Głębokość\npartycji – chroma", fontsize=10)
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("d / 10", fontsize=8)
    
    ax = axes[4]
    ax.imshow(chroma_boundary, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("(e) Granice CU\n– chroma", fontsize=10)
    ax.axis("off")
    
    fig.suptitle(
        f"Mapy partycji VVC – podział klatki na jednostki kodowania (QP = 37, nofilter, crop {CROP_SIZE}×{CROP_SIZE})",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    
    fig.savefig(OUTPUT_PDF := OUTPUT_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
