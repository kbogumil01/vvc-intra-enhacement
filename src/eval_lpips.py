#!/usr/bin/env python3
"""
LPIPS comparison: baseline (decoded) vs MSE model vs RaGAN model.
Evaluates on CTC sequences (nofilter, temporal 12ch).
Outputs per-sequence and average LPIPS for each variant.
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import lpips

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


def yuv420_to_rgb_tensor(y, u, v):
    """Convert YUV420 uint16 10-bit to RGB float tensor [1,3,H,W] in [-1,1] for LPIPS."""
    h, w = y.shape
    max_val = 1023.0

    u_up = cv2.resize(u.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    v_up = cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

    y_f = y.astype(np.float32) / max_val
    u_f = np.clip(u_up / max_val, 0, 1)
    v_f = np.clip(v_up / max_val, 0, 1)

    y_c = y_f
    cb = u_f - 0.5
    cr = v_f - 0.5
    r = y_c + 1.5748 * cr
    g = y_c - 0.1873 * cb - 0.4681 * cr
    b = y_c + 1.8556 * cb

    rgb = np.stack([r, g, b], axis=0)
    rgb = np.clip(rgb, 0, 1)

    rgb = rgb * 2.0 - 1.0
    return torch.from_numpy(rgb).unsqueeze(0).float()


def enhanced_to_rgb_tensor(enh_444):
    """Convert enhanced [3,H,W] float32 [0,1] YUV444 to RGB tensor [-1,1]."""
    y_c = enh_444[0]
    cb = enh_444[1] - 0.5
    cr = enh_444[2] - 0.5
    r = y_c + 1.5748 * cr
    g = y_c - 0.1873 * cb - 0.4681 * cr
    b = y_c + 1.8556 * cb
    rgb = np.stack([r, g, b], axis=0)
    rgb = np.clip(rgb, 0, 1) * 2.0 - 1.0
    return torch.from_numpy(rgb).unsqueeze(0).float()


def upsample_chroma(y, u, v):
    h, w = y.shape
    max_val = 1023.0
    u_up = cv2.resize(u.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    v_up = cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    np.clip(u_up, 0, max_val, out=u_up)
    np.clip(v_up, 0, max_val, out=v_up)
    return np.stack([y.astype(np.float32) / max_val, u_up / max_val, v_up / max_val], axis=0)


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
    return model, in_channels


@torch.no_grad()
def enhance_frame(model, rec_444, neighbors_444, qp, partition_maps, device,
                  tile_size=512, overlap=32):
    """Run model on frame with tiling. Builds 12ch input."""
    _, h, w = rec_444.shape
    qp_val = float(qp) / 63.0
    qp_map = np.full((1, h, w), qp_val, dtype=np.float32)

    parts = [rec_444, qp_map]
    if partition_maps is not None:
        parts.append(partition_maps)
    else:
        parts.append(np.zeros((2, h, w), dtype=np.float32))
    if neighbors_444 is not None:
        parts.append(neighbors_444)
    else:
        parts.append(np.zeros((6, h, w), dtype=np.float32))

    full_input = np.concatenate(parts, axis=0)

    if h <= tile_size and w <= tile_size:
        inp_t = torch.from_numpy(full_input).unsqueeze(0).to(device)
        out = model(inp_t).squeeze(0).cpu().numpy()
        return np.clip(out, 0.0, 1.0)

    out = np.zeros_like(rec_444)
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


def find_ctc_sequences(decoded_root, original_root, neighbor_root, qp_list):
    """Find CTC sequences with all QPs available.
    Structure: decoded_root/QP{qp}/{seq_name}_frame{N}/rec.yuv
    """
    from collections import defaultdict
    seq_frames = defaultdict(lambda: defaultdict(list))

    for qp in qp_list:
        qp_dir = Path(decoded_root) / f"QP{qp}"
        if not qp_dir.exists():
            continue
        for frame_dir in sorted(qp_dir.iterdir()):
            if not frame_dir.is_dir():
                continue
            name = frame_dir.name
            parts = name.rsplit("_frame", 1)
            if len(parts) != 2:
                continue
            seq_base = parts[0]
            frame_num = int(parts[1])
            rec_path = frame_dir / "rec.yuv"
            if rec_path.exists():
                seq_frames[seq_base][qp].append({
                    "frame_num": frame_num,
                    "rec_path": str(rec_path),
                    "decoded_dir": str(frame_dir),
                })

    valid = {}
    for seq_base, qp_data in seq_frames.items():
        if len(qp_data) < len(qp_list):
            continue
        entry = {}
        for qp in qp_list:
            if qp in qp_data and qp_data[qp]:
                frame_info = qp_data[qp][0]
                entry[qp] = frame_info
        if len(entry) == len(qp_list):
            valid[seq_base] = entry
    return valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-root", required=True)
    p.add_argument("--neighbor-root", required=True)
    p.add_argument("--original-root", required=True)
    p.add_argument("--mse-ckpt", required=True)
    p.add_argument("--gan-ckpt", required=True)
    p.add_argument("--qp-list", type=int, nargs="+", default=[22, 27, 32, 37])
    p.add_argument("--output", default="lpips_comparison.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    print("LPIPS model loaded (AlexNet)")

    print(f"Loading MSE model: {args.mse_ckpt}")
    mse_model, mse_ch = load_model_from_ckpt(args.mse_ckpt, device, is_gan=False)
    print(f"  in_channels={mse_ch}")

    print(f"Loading GAN model: {args.gan_ckpt}")
    gan_model, gan_ch = load_model_from_ckpt(args.gan_ckpt, device, is_gan=True)
    print(f"  in_channels={gan_ch}")

    sequences = find_ctc_sequences(args.decoded_root, args.original_root, args.neighbor_root, args.qp_list)
    print(f"Found {len(sequences)} sequences with all QPs")

    excluded = {"FourPeople", "Johnny", "KristenAndSara"}

    results = []

    for seq_name in sorted(sequences.keys()):
        if any(seq_name.startswith(e) for e in excluded):
            print(f"  [SKIP] {seq_name} (class E)")
            continue

        lpips_base_list = []
        lpips_mse_list = []
        lpips_gan_list = []

        for qp in args.qp_list:
            if qp not in sequences[seq_name]:
                continue

            info = sequences[seq_name][qp]
            rec_path = info["rec_path"]
            decoded_dir = info["decoded_dir"]
            frame_num = info["frame_num"]

            parts = seq_name.split("_")
            res_part = [p for p in parts if "x" in p and p[0].isdigit()]
            if not res_part:
                continue
            w, h = map(int, res_part[0].split("x"))

            orig_path = Path(args.original_root) / f"{seq_name}_frame{frame_num}.yuv"
            if not orig_path.exists():
                print(f"    WARN: no original for {seq_name} frame{frame_num}")
                continue

            y_ref, u_ref, v_ref = read_yuv420(str(orig_path), h, w)
            y_rec, u_rec, v_rec = read_yuv420(rec_path, h, w)

            ref_rgb = yuv420_to_rgb_tensor(y_ref, u_ref, v_ref).to(device)

            base_rgb = yuv420_to_rgb_tensor(y_rec, u_rec, v_rec).to(device)
            lpips_base = lpips_fn(ref_rgb, base_rgb).item()

            rec_444 = upsample_chroma(y_rec, u_rec, v_rec)

            prev_path = Path(args.neighbor_root) / f"QP{qp}" / f"{seq_name}_frame{frame_num - 1}" / "rec.yuv"
            next_path = Path(args.neighbor_root) / f"QP{qp}" / f"{seq_name}_frame{frame_num + 1}" / "rec.yuv"

            neighbors_444 = None
            if prev_path.exists() and next_path.exists():
                yp, up, vp = read_yuv420(str(prev_path), h, w)
                yn, un, vn = read_yuv420(str(next_path), h, w)
                prev_444 = upsample_chroma(yp, up, vp)
                next_444 = upsample_chroma(yn, un, vn)
                neighbors_444 = np.concatenate([prev_444, next_444], axis=0)

            bd_dir = Path(decoded_dir)
            bnd_luma_path = bd_dir / "boundary_map.raw"
            bnd_chroma_path = bd_dir / "boundary_chroma_map.raw"
            partition_maps = None
            if bnd_luma_path.exists() and bnd_chroma_path.exists():
                bl = np.fromfile(str(bnd_luma_path), dtype=np.float16).astype(np.float32).reshape(h, w)
                bc = np.fromfile(str(bnd_chroma_path), dtype=np.float16).astype(np.float32).reshape(h, w)
                partition_maps = np.stack([bl, bc], axis=0)

            mse_enh = enhance_frame(mse_model, rec_444, neighbors_444, qp, partition_maps, device)
            mse_rgb = enhanced_to_rgb_tensor(mse_enh).to(device)
            lpips_mse = lpips_fn(ref_rgb, mse_rgb).item()

            gan_enh = enhance_frame(gan_model, rec_444, neighbors_444, qp, partition_maps, device)
            gan_rgb = enhanced_to_rgb_tensor(gan_enh).to(device)
            lpips_gan = lpips_fn(ref_rgb, gan_rgb).item()

            lpips_base_list.append(lpips_base)
            lpips_mse_list.append(lpips_mse)
            lpips_gan_list.append(lpips_gan)

            del ref_rgb, base_rgb, mse_rgb, gan_rgb
            torch.cuda.empty_cache()

        if lpips_base_list:
            avg_base = np.mean(lpips_base_list)
            avg_mse = np.mean(lpips_mse_list)
            avg_gan = np.mean(lpips_gan_list)
            results.append((seq_name, avg_base, avg_mse, avg_gan))
            print(f"  {seq_name:<50} base={avg_base:.4f}  MSE={avg_mse:.4f}  GAN={avg_gan:.4f}")

    print("\n" + "=" * 80)
    print(f"{'AVERAGE':<50} {'Baseline':>10} {'MSE 12ch':>10} {'RaGAN 12ch':>10}")
    print("=" * 80)
    if results:
        avg_b = np.mean([r[1] for r in results])
        avg_m = np.mean([r[2] for r in results])
        avg_g = np.mean([r[3] for r in results])
        print(f"{'LPIPS (lower=better)':<50} {avg_b:>10.4f} {avg_m:>10.4f} {avg_g:>10.4f}")
        print(f"{'Δ vs baseline':<50} {'---':>10} {avg_m - avg_b:>+10.4f} {avg_g - avg_b:>+10.4f}")
        print(f"{'Δ GAN vs MSE':<50} {'':>10} {'':>10} {avg_g - avg_m:>+10.4f}")

    with open(args.output, "w") as f:
        f.write("sequence,lpips_baseline,lpips_mse,lpips_gan\n")
        for seq, b, m, g in results:
            f.write(f"{seq},{b:.6f},{m:.6f},{g:.6f}\n")
        if results:
            f.write(f"AVERAGE,{avg_b:.6f},{avg_m:.6f},{avg_g:.6f}\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
