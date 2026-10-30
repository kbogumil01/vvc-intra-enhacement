"""
Visualize CTC frame with partition boundary overlay for thesis.
Produces a side-by-side figure: original crop | crop with boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA = Path("/net/tscratch/people/plgkarolbogumil/data")
SEQ_NAME = "BasketballDrive_1920x1080_10bit_420_frame12"
QP = 37
W, H = 1920, 1080

CROP_X, CROP_Y, CROP_W, CROP_H = 700, 300, 512, 512

OUTPUT_PATH = Path("/net/people/plgrid/plgkarolbogumil/mgr/dense/partition_visualization.pdf")
OUTPUT_PNG = OUTPUT_PATH.with_suffix(".png")

orig_path = DATA / "splitted_CTC" / f"{SEQ_NAME}.yuv"
y_size = W * H
frame_bytes = np.fromfile(orig_path, dtype=np.uint16, count=y_size)
y_orig = frame_bytes.reshape(H, W).astype(np.float32) / 1023.0

decoded_dir = DATA / f"decoded_CTC_YUV420_10bit/QP{QP}/{SEQ_NAME}"
boundary_path = decoded_dir / "boundary_map.raw"
boundary = np.fromfile(boundary_path, dtype=np.float16).reshape(H, W).astype(np.float32)

depth_path = decoded_dir / "depth_map.raw"
depth = np.fromfile(depth_path, dtype=np.float16).reshape(H, W).astype(np.float32)

y_crop = y_orig[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
boundary_crop = boundary[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]
depth_crop = depth[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

axes[0].imshow(y_crop, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("(a) Fragment klatki (luma)", fontsize=12)
axes[0].axis('off')

axes[1].imshow(y_crop, cmap='gray', vmin=0, vmax=1)
boundary_rgba = np.zeros((CROP_H, CROP_W, 4), dtype=np.float32)
boundary_rgba[..., 0] = 1.0
boundary_rgba[..., 3] = boundary_crop * 0.8
axes[1].imshow(boundary_rgba)
axes[1].set_title("(b) Fragment z granicami bloków CU", fontsize=12)
axes[1].axis('off')

plt.tight_layout(pad=1.0)
plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=150)
plt.savefig(OUTPUT_PATH, bbox_inches='tight')
print(f"Saved: {OUTPUT_PNG}")
print(f"Saved: {OUTPUT_PATH}")

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

axes2[0].imshow(y_crop, cmap='gray', vmin=0, vmax=1)
axes2[0].set_title("(a) Fragment klatki (luma)", fontsize=11)
axes2[0].axis('off')

axes2[1].imshow(y_crop, cmap='gray', vmin=0, vmax=1)
axes2[1].imshow(boundary_rgba)
axes2[1].set_title("(b) Granice bloków CU", fontsize=11)
axes2[1].axis('off')

im = axes2[2].imshow(depth_crop, cmap='viridis', vmin=0, vmax=1)
axes2[2].set_title("(c) Mapa głębokości podziału", fontsize=11)
axes2[2].axis('off')
cbar = plt.colorbar(im, ax=axes2[2], fraction=0.046, pad=0.04)
cbar.set_label("Głębokość / 7", fontsize=9)

plt.tight_layout(pad=1.0)
out3 = OUTPUT_PATH.with_name("partition_visualization_3panel.png")
plt.savefig(out3, bbox_inches='tight', dpi=150)
plt.savefig(out3.with_suffix(".pdf"), bbox_inches='tight')
print(f"Saved: {out3}")
print(f"Saved: {out3.with_suffix('.pdf')}")

plt.close('all')
print("Done!")
