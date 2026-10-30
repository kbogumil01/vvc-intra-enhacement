"""
Visualize VVC compression artifacts at high QP.
Generates 4 side-by-side images (original vs compressed) with tight crops:
1. Blocking artifacts - nofilter for maximum visibility
2. Ringing artifacts - sharp edges with oscillations
3. Mosquito noise - complex edge regions
4. Color bleeding - chroma artifacts at sharp color transitions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("/net/tscratch/people/plgkarolbogumil/data")
DECODED = DATA / "decoded_CTC_YUV420_10bit" / "QP37"
DECODED_NOFILTER = DATA / "decoded_CTC_YUV420_10bit_nofilter" / "QP37"
ORIGINAL = DATA / "splitted_CTC"
OUTPUT_DIR = Path("/net/people/plgrid/plgkarolbogumil/mgr/dense")


def read_yuv420_10bit(path: Path, width: int, height: int) -> tuple:
    """Read YUV420 10-bit frame, return Y, U, V as float [0, 1]."""
    y_size = width * height
    uv_width = width // 2
    uv_height = height // 2
    uv_size = uv_width * uv_height

    raw = np.fromfile(path, dtype=np.uint16)
    y = raw[:y_size].reshape(height, width).astype(np.float32) / 1023.0
    u = raw[y_size:y_size + uv_size].reshape(uv_height, uv_width).astype(np.float32) / 1023.0
    v = raw[y_size + uv_size:y_size + 2 * uv_size].reshape(uv_height, uv_width).astype(np.float32) / 1023.0
    return y, u, v


def yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Convert YUV420 to RGB. Upsamples U, V to luma resolution."""
    u_up = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_up = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

    h, w = y.shape
    u_up = u_up[:h, :w]
    v_up = v_up[:h, :w]

    r = y + 1.5748 * (v_up - 0.5)
    g = y - 0.1873 * (u_up - 0.5) - 0.4681 * (v_up - 0.5)
    b = y + 1.8556 * (u_up - 0.5)

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)


def load_frame(sequence: str, qp: int = 37, nofilter: bool = False):
    """Load original and decoded frame pair."""
    orig_path = ORIGINAL / f"{sequence}.yuv"
    if nofilter:
        dec_dir = DATA / "decoded_CTC_YUV420_10bit_nofilter" / f"QP{qp}" / sequence
    else:
        dec_dir = DATA / "decoded_CTC_YUV420_10bit" / f"QP{qp}" / sequence
    dec_path = dec_dir / "rec.yuv"

    parts = sequence.split("_")
    res = parts[1]
    width, height = map(int, res.split("x"))

    y_orig, u_orig, v_orig = read_yuv420_10bit(orig_path, width, height)
    y_dec, u_dec, v_dec = read_yuv420_10bit(dec_path, width, height)

    rgb_orig = yuv_to_rgb(y_orig, u_orig, v_orig)
    rgb_dec = yuv_to_rgb(y_dec, u_dec, v_dec)

    return rgb_orig, rgb_dec


def crop(img, y, x, h, w):
    return img[y:y+h, x:x+w]


def make_comparison(orig_crop, dec_crop, title, filename, subtitle="VTM QP=37", figsize=(10, 4.5)):
    """Create side-by-side comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(orig_crop, interpolation='nearest')
    axes[0].set_title("Oryginał", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(dec_crop, interpolation='nearest')
    axes[1].set_title(subtitle, fontsize=13)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("1. Blocking artifacts (nofilter, tight crop on sky)...")
    rgb_orig, rgb_dec = load_frame("BQTerrace_1920x1080_10bit_420_frame104", nofilter=True)
    orig_crop = crop(rgb_orig, 20, 700, 128, 256)
    dec_crop = crop(rgb_dec, 20, 700, 128, 256)
    make_comparison(orig_crop, dec_crop,
                    "Artefakty blokowe (blocking)",
                    "artifact_blocking.png",
                    subtitle="VTM QP=37 (bez filtrów pętli)")

    print("2. Ringing artifacts (tight crop on court line)...")
    rgb_orig, rgb_dec = load_frame("BasketballDrill_832x480_10bit_420_frame114")
    orig_crop = crop(rgb_orig, 380, 150, 90, 180)
    dec_crop = crop(rgb_dec, 380, 150, 90, 180)
    make_comparison(orig_crop, dec_crop,
                    "Artefakty dzwonienia (ringing)",
                    "artifact_ringing.png")

    print("3. Mosquito noise (tight crop on text/detail)...")
    rgb_orig, rgb_dec = load_frame("PartyScene_832x480_10bit_420_frame114")
    orig_crop = crop(rgb_orig, 5, 30, 120, 200)
    dec_crop = crop(rgb_dec, 5, 30, 120, 200)
    make_comparison(orig_crop, dec_crop,
                    "Szum komara (mosquito noise)",
                    "artifact_mosquito.png")

    print("4. Color bleeding (tight crop on colorful signs)...")
    rgb_orig, rgb_dec = load_frame("BQMall_832x480_10bit_420_frame104")
    orig_crop = crop(rgb_orig, 20, 400, 100, 180)
    dec_crop = crop(rgb_dec, 20, 400, 100, 180)
    make_comparison(orig_crop, dec_crop,
                    "Rozlewanie koloru (color bleeding)",
                    "artifact_color_bleeding.png")

    print("5. Creating combined figure with amplified differences...")
    sequences_info = [
        ("BQTerrace_1920x1080_10bit_420_frame104", (20, 700, 128, 256),
         "Artefakty blokowe", True),
        ("BasketballDrill_832x480_10bit_420_frame114", (380, 150, 90, 180),
         "Dzwonienie (ringing)", False),
        ("PartyScene_832x480_10bit_420_frame114", (5, 30, 120, 200),
         "Szum komara", False),
        ("BQMall_832x480_10bit_420_frame104", (20, 400, 100, 180),
         "Rozlewanie koloru", False),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(11, 12))

    for i, (seq, (cy, cx, ch, cw), title, use_nofilter) in enumerate(sequences_info):
        rgb_orig, rgb_dec = load_frame(seq, nofilter=use_nofilter)
        orig_c = crop(rgb_orig, cy, cx, ch, cw)
        dec_c = crop(rgb_dec, cy, cx, ch, cw)

        diff = np.abs(dec_c.astype(np.float32) - orig_c.astype(np.float32))
        diff_amp = np.clip(diff * 8, 0, 1)

        axes[i, 0].imshow(orig_c, interpolation='nearest')
        axes[i, 0].set_ylabel(title, fontsize=10, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(dec_c, interpolation='nearest')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        axes[i, 2].imshow(diff_amp, interpolation='nearest')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

    axes[0, 0].set_title("Oryginał", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("VTM QP=37", fontsize=12, fontweight="bold")
    axes[0, 2].set_title("|Różnica| × 8", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "artifacts_all_with_diff.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print("\nDone!")
    print("  - artifact_blocking.png")
    print("  - artifact_ringing.png")
    print("  - artifact_mosquito.png")
    print("  - artifact_color_bleeding.png")
    print("  - artifacts_all_with_diff.png (combined 3-column)")


if __name__ == "__main__":
    main()
