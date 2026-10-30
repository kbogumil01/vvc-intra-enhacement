#!/usr/bin/env python3
"""
Visualize temporal context: prev frame | current frame | next frame.
Shows the three frames that form the temporal input to the 12-channel model.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEMMAP_DIR = "/net/tscratch/people/plgkarolbogumil/data/temporal_memmap_nofilter"

TARGET_SEQ = "XIPH"
TARGET_QP = 37
TARGET_FRAME = 34

OUTPUT_PNG = "/net/people/plgrid/plgkarolbogumil/mgr/dense/temporal_vis.png"
OUTPUT_PDF = "/net/people/plgrid/plgkarolbogumil/mgr/dense/temporal_vis.pdf"


def read_frame_rgb(fdata, file_index, path):
    """Read full frame from memmap and convert to RGB."""
    fi = file_index[path]
    h, w = fi["height"], fi["width"]
    offset = fi["offset"]

    yuv = np.empty((3, h, w), dtype=np.float32)
    for c in range(3):
        off = offset + c * h * w
        yuv[c] = fdata[off : off + h * w].astype(np.float32).reshape(h, w)

    Y = yuv[0]
    Cb = yuv[1] - 0.5
    Cr = yuv[2] - 0.5
    R = Y + 1.5748 * Cr
    G = Y - 0.1873 * Cb - 0.4681 * Cr
    B = Y + 1.8556 * Cb
    return np.stack([R, G, B], axis=-1).clip(0, 1)


def main():
    with open(f"{MEMMAP_DIR}/index.json") as f:
        idx = json.load(f)

    file_index = idx["file_index"]
    records = idx["records"]

    rec = None
    for r in records:
        if (r["qp"] == TARGET_QP and "Johnny" in r["sequence_id"]
            and r["prev_path"] in file_index and r["next_path"] in file_index):
            rec = r
            break
    if rec is None:
        raise RuntimeError("No suitable temporal record found")

    print(f"Sequence: {rec['sequence_id']}")
    print(f"  Frame {rec['frame_id']}, QP={rec['qp']}, {rec['height']}x{rec['width']}")
    print(f"  Current: {rec['rec_path'].split('/')[-2]}")
    print(f"  Prev:    {rec['prev_path'].split('/')[-2]}")
    print(f"  Next:    {rec['next_path'].split('/')[-2]}")

    total = idx["total_elems"]
    fdata = np.memmap(f"{MEMMAP_DIR}/frames.dat", dtype=np.float16, mode="r", shape=(total,))

    prev_rgb = read_frame_rgb(fdata, file_index, rec["prev_path"])
    curr_rgb = read_frame_rgb(fdata, file_index, rec["rec_path"])
    next_rgb = read_frame_rgb(fdata, file_index, rec["next_path"])

    h, w = rec["height"], rec["width"]
    print(f"  Loaded 3 frames: {h}x{w}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 18 * h / w / 3))

    ax1.imshow(prev_rgb, aspect="equal")
    ax1.axis("off")

    ax2.imshow(curr_rgb, aspect="equal")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#22CC22")
        spine.set_linewidth(3)
        spine.set_visible(True)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(next_rgb, aspect="equal")
    ax3.axis("off")

    plt.subplots_adjust(wspace=0.02, left=0, right=1, top=1, bottom=0)

    fig.savefig(OUTPUT_PDF, dpi=150, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", pad_inches=0.02)
    print(f"\n  Saved: {OUTPUT_PDF}")
    print(f"  Saved: {OUTPUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()
