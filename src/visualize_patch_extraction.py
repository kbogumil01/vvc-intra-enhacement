#!/usr/bin/env python3
"""
Visualize a full DIV2K frame with a 256x256 patch highlighted,
and the extracted patch shown next to it.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MEMMAP_DIR = "/net/tscratch/people/plgkarolbogumil/data/memmap_cache_nofilter"
TARGET_NAME = "DIV2K_0010"
TARGET_QP = "QP37"
PATCH_SIZE = 256

OUTPUT_PNG = "/net/people/plgrid/plgkarolbogumil/mgr/dense/patch_extraction_vis.png"
OUTPUT_PDF = "/net/people/plgrid/plgkarolbogumil/mgr/dense/patch_extraction_vis.pdf"


def main():
    with open(f"{MEMMAP_DIR}/index.json") as f:
        frame_idx = json.load(f)

    key = None
    for k in frame_idx["file_index"]:
        if TARGET_NAME in k and TARGET_QP in k:
            key = k
            break
    if key is None:
        raise RuntimeError(f"{TARGET_NAME} {TARGET_QP} not found")

    fi = frame_idx["file_index"][key]
    h, w = fi["height"], fi["width"]
    print(f"Frame: {key}")
    print(f"  Size: {h}x{w}")

    total = frame_idx["total_elems"]
    fdata = np.memmap(f"{MEMMAP_DIR}/frames.dat", dtype=np.float16, mode="r", shape=(total,))

    yuv = np.empty((3, h, w), dtype=np.float32)
    for c in range(3):
        off = fi["offset"] + c * h * w
        yuv[c] = fdata[off : off + h * w].astype(np.float32).reshape(h, w)

    Y = yuv[0]
    Cb = yuv[1] - 0.5
    Cr = yuv[2] - 0.5
    R = Y + 1.5748 * Cr
    G = Y - 0.1873 * Cb - 0.4681 * Cr
    B = Y + 1.8556 * Cb
    rgb = np.stack([R, G, B], axis=-1).clip(0, 1)

    cy = int(h * 0.35)
    cx = int(w * 0.55)
    cy -= cy % 2
    cx -= cx % 2
    ps = PATCH_SIZE

    patch = rgb[cy:cy+ps, cx:cx+ps]
    print(f"  Patch at ({cy}, {cx}), size {ps}x{ps}")

    fig = plt.figure(figsize=(16, 16 * h / w * 0.55))
    
    gs = fig.add_gridspec(1, 2, width_ratios=[w, h * 0.6], wspace=0.03)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.imshow(rgb, aspect="equal")
    rect = Rectangle(
        (cx, cy), ps, ps,
        linewidth=2.5, edgecolor="#FF3333", facecolor="none", linestyle="-"
    )
    ax1.add_patch(rect)
    ax1.axis("off")

    ax2.imshow(patch, aspect="equal")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#FF3333")
        spine.set_linewidth(2.5)
        spine.set_visible(True)
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.savefig(OUTPUT_PDF, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"  Saved: {OUTPUT_PDF}")
    print(f"  Saved: {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
