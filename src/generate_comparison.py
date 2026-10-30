#!/usr/bin/env python3
"""
Generate visual comparison: decoded vs MSE-enhanced vs GAN-enhanced.
Crops a region showing clear GAN improvement (sharper textures).
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model_v2
import model_wcdann
import importlib
importlib.import_module("vvc-ppff")

from registry import create_model


def read_yuv420(path, h, w):
    y_size = w * h
    uv_w, uv_h = w // 2, h // 2
    uv_size = uv_w * uv_h
    data = np.fromfile(path, dtype=np.uint16)
    y = data[:y_size].reshape(h, w)
    u = data[y_size:y_size + uv_size].reshape(uv_h, uv_w)
    v = data[y_size + uv_size:y_size + 2 * uv_size].reshape(uv_h, uv_w)
    return y, u, v


def upsample_chroma(y, u, v):
    h, w = y.shape
    max_val = 1023.0
    u_up = cv2.resize(u.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    v_up = cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    np.clip(u_up, 0, max_val, out=u_up)
    np.clip(v_up, 0, max_val, out=v_up)
    return np.stack([y.astype(np.float32) / max_val, u_up / max_val, v_up / max_val], axis=0)


def yuv444_to_rgb(yuv444):
    """[3,H,W] float32 [0,1] (normalized by /1023) YUV → RGB [0,1].
    Handles BT.709 limited range: Y=[64,940]/1023, C=[64,960]/1023."""
    y_full = (yuv444[0] * 1023.0 - 64.0) / 876.0
    cb = (yuv444[1] * 1023.0 - 512.0) / 896.0
    cr = (yuv444[2] * 1023.0 - 512.0) / 896.0

    r = y_full + 1.5748 * cr
    g = y_full - 0.1873 * cb - 0.4681 * cr
    b = y_full + 1.8556 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)


def load_model_from_ckpt(ckpt_path, device, is_gan=False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    hp = ckpt.get("hyper_parameters", {})
    if is_gan:
        gan_cfg = hp.get("gan_config", None)
        model_name = gan_cfg.generator_name if gan_cfg else "ppff_base"
    else:
        model_name = hp.get("model_name", "ppff_base")
    in_channels = 12
    for k, v in state.items():
        if "initial_conv" in k and v.dim() == 4:
            in_channels = v.shape[1]
            break
    model = create_model(model_name, in_channels=in_channels)
    cleaned = {}
    for k, v in state.items():
        if is_gan and k.startswith("generator."):
            cleaned[k[len("generator."):]] = v
        elif k.startswith("model."):
            cleaned[k[6:]] = v
        elif k.startswith("discriminator.") or k.startswith("vgg_loss.") or k.startswith("criterion."):
            continue
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned)
    model.to(device).eval()
    return model


@torch.no_grad()
def enhance_frame(model, full_input, device, tile_size=512, overlap=32):
    """Tile-based inference on [C,H,W] input."""
    _, h, w = full_input.shape
    if h <= tile_size and w <= tile_size:
        inp_t = torch.from_numpy(full_input).unsqueeze(0).to(device)
        out = model(inp_t).squeeze(0).cpu().numpy()
        return np.clip(out, 0.0, 1.0)

    out = np.zeros((3, h, w), dtype=np.float32)
    step = tile_size - overlap
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            y0a = max(0, y1 - tile_size)
            x0a = max(0, x1 - tile_size)
            tile = full_input[:, y0a:y1, x0a:x1]
            inp_t = torch.from_numpy(tile).unsqueeze(0).to(device)
            out_tile = model(inp_t).squeeze(0).cpu().numpy()
            oy_s = overlap // 2 if y0a > 0 else 0
            ox_s = overlap // 2 if x0a > 0 else 0
            oy_e = out_tile.shape[1] - (overlap // 2 if y1 < h else 0)
            ox_e = out_tile.shape[2] - (overlap // 2 if x1 < w else 0)
            out[:, y0a + oy_s:y0a + oy_e, x0a + ox_s:x0a + ox_e] = \
                out_tile[:, oy_s:oy_e, ox_s:ox_e]
    return np.clip(out, 0.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-root", required=True)
    p.add_argument("--neighbor-root", required=True)
    p.add_argument("--original-root", required=True)
    p.add_argument("--mse-ckpt", required=True)
    p.add_argument("--gan-ckpt", required=True)
    p.add_argument("--sequence", default="RitualDance_1920x1080_10bit_420")
    p.add_argument("--qp", type=int, default=37)
    p.add_argument("--crop", type=int, nargs=4, default=None,
                   help="y0 x0 h w crop region")
    p.add_argument("--output", default="comparison_gan.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq = args.sequence
    parts = seq.split("_")
    res_part = [pp for pp in parts if "x" in pp and pp[0].isdigit()]
    w, h = map(int, res_part[0].split("x"))

    qp_dir = Path(args.decoded_root) / f"QP{args.qp}"
    frame_dirs = sorted(qp_dir.glob(f"{seq}_frame*"))
    if not frame_dirs:
        print(f"No frames found for {seq} QP{args.qp}")
        return
    frame_dir = frame_dirs[len(frame_dirs)//2]
    frame_name = frame_dir.name
    frame_num = int(frame_name.rsplit("_frame", 1)[1])
    print(f"Using: {frame_name} ({w}x{h})")

    rec_path = frame_dir / "rec.yuv"
    y_rec, u_rec, v_rec = read_yuv420(str(rec_path), h, w)
    rec_444 = upsample_chroma(y_rec, u_rec, v_rec)

    orig_path = Path(args.original_root) / f"{frame_name}.yuv"
    y_orig, u_orig, v_orig = read_yuv420(str(orig_path), h, w)
    orig_444 = upsample_chroma(y_orig, u_orig, v_orig)

    prev_dir = Path(args.neighbor_root) / f"QP{args.qp}" / f"{seq}_frame{frame_num - 1}"
    next_dir = Path(args.neighbor_root) / f"QP{args.qp}" / f"{seq}_frame{frame_num + 1}"
    neighbors_444 = None
    if (prev_dir / "rec.yuv").exists() and (next_dir / "rec.yuv").exists():
        yp, up, vp = read_yuv420(str(prev_dir / "rec.yuv"), h, w)
        yn, un, vn = read_yuv420(str(next_dir / "rec.yuv"), h, w)
        prev_444 = upsample_chroma(yp, up, vp)
        next_444 = upsample_chroma(yn, un, vn)
        neighbors_444 = np.concatenate([prev_444, next_444], axis=0)
    else:
        neighbors_444 = np.zeros((6, h, w), dtype=np.float32)
        print("WARNING: neighbors not found, using zeros")

    bnd_l = frame_dir / "boundary_map.raw"
    bnd_c = frame_dir / "boundary_chroma_map.raw"
    if bnd_l.exists() and bnd_c.exists():
        bl = np.fromfile(str(bnd_l), dtype=np.float16).astype(np.float32).reshape(h, w)
        bc = np.fromfile(str(bnd_c), dtype=np.float16).astype(np.float32).reshape(h, w)
        partition_maps = np.stack([bl, bc], axis=0)
    else:
        partition_maps = np.zeros((2, h, w), dtype=np.float32)

    qp_val = float(args.qp) / 63.0
    qp_map = np.full((1, h, w), qp_val, dtype=np.float32)
    full_input = np.concatenate([rec_444, qp_map, partition_maps, neighbors_444], axis=0)
    print(f"Input shape: {full_input.shape}")

    print("Loading MSE model...")
    mse_model = load_model_from_ckpt(args.mse_ckpt, device, is_gan=False)
    print("Loading GAN model...")
    gan_model = load_model_from_ckpt(args.gan_ckpt, device, is_gan=True)

    print("Running MSE inference...")
    mse_enh = enhance_frame(mse_model, full_input, device)
    print("Running GAN inference...")
    gan_enh = enhance_frame(gan_model, full_input, device)

    rgb_orig = yuv444_to_rgb(orig_444)
    rgb_dec = yuv444_to_rgb(rec_444)
    rgb_mse = yuv444_to_rgb(mse_enh)
    rgb_gan = yuv444_to_rgb(gan_enh)

    if args.crop:
        cy, cx, ch, cw = args.crop
    else:
        ch, cw = min(128, h//3), min(192, w//3)
        gray = (rgb_orig[:, :, 0] * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        best_var = 0
        cy, cx = h//4, w//4
        step = 16
        for yy in range(0, h - ch, step):
            for xx in range(0, w - cw, step):
                var = np.var(laplacian[yy:yy+ch, xx:xx+cw])
                if var > best_var:
                    best_var = var
                    cy, cx = yy, xx
        print(f"Auto-crop: y={cy}, x={cx}, h={ch}, w={cw} (var={best_var:.1f})")

    rgb_orig_c = rgb_orig[cy:cy+ch, cx:cx+cw]
    rgb_dec_c = rgb_dec[cy:cy+ch, cx:cx+cw]
    rgb_mse_c = rgb_mse[cy:cy+ch, cx:cx+cw]
    rgb_gan_c = rgb_gan[cy:cy+ch, cx:cx+cw]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=200)
    titles = ['Oryginał', 'Zdekodowany (bez filtrów)', 'MSE (12ch)', 'RaGAN (12ch)']
    images = [rgb_orig_c, rgb_dec_c, rgb_mse_c, rgb_gan_c]

    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout(pad=0.3)
    plt.savefig(args.output, bbox_inches='tight', dpi=200)
    print(f"Saved to {args.output}")

    base = Path(args.output).stem
    for name, img in zip(['original', 'decoded', 'mse', 'gan'], images):
        out_path = f"{base}_{name}.png"
        cv2.imwrite(out_path, (img[:, :, ::-1] * 255).astype(np.uint8))
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
