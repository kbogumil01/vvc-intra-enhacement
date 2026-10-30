#!/usr/bin/env python3
"""
BD-Rate evaluation for the 12-channel temporal model on CTC sequences.

Anchor: VTM without filters (nofilter) — no network.
Test:   VTM nofilter + temporal network (12ch).

Input channels: [YUV_curr(3), QP(1), boundary_luma(1), boundary_chroma(1),
                 YUV_prev(3), YUV_next(3)] = 12

Usage:
    python evaluate_bd_temporal_ctc.py \
        --ckpt-path <checkpoint.ckpt> \
        --decoded-root /data/decoded_CTC_YUV420_10bit_nofilter \
        --neighbor-root /data/dograna_data/decoded_CTC_neighbors_nofilter \
        --original-root /data/splitted_CTC \
        --output bd_results_temporal_ctc.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from registry import create_model

import model_v2, model_wcdann
import importlib
importlib.import_module("vvc-ppff")


def bd_rate(rate_a, psnr_a, rate_t, psnr_t):
    rate_a, psnr_a = np.asarray(rate_a, np.float64), np.asarray(psnr_a, np.float64)
    rate_t, psnr_t = np.asarray(rate_t, np.float64), np.asarray(psnr_t, np.float64)
    lr_a, lr_t = np.log(rate_a), np.log(rate_t)
    pa = np.polyfit(psnr_a, lr_a, 3)
    pt = np.polyfit(psnr_t, lr_t, 3)
    lo = max(psnr_a.min(), psnr_t.min())
    hi = min(psnr_a.max(), psnr_t.max())
    if hi <= lo:
        return 0.0
    ia = np.polyval(np.polyint(pa), hi) - np.polyval(np.polyint(pa), lo)
    it = np.polyval(np.polyint(pt), hi) - np.polyval(np.polyint(pt), lo)
    return (np.exp((it - ia) / (hi - lo)) - 1.0) * 100.0


def bd_psnr(rate_a, psnr_a, rate_t, psnr_t):
    rate_a, psnr_a = np.asarray(rate_a, np.float64), np.asarray(psnr_a, np.float64)
    rate_t, psnr_t = np.asarray(rate_t, np.float64), np.asarray(psnr_t, np.float64)
    lr_a, lr_t = np.log(rate_a), np.log(rate_t)
    pa = np.polyfit(lr_a, psnr_a, 3)
    pt = np.polyfit(lr_t, psnr_t, 3)
    lo = max(lr_a.min(), lr_t.min())
    hi = min(lr_a.max(), lr_t.max())
    if hi <= lo:
        return 0.0
    ia = np.polyval(np.polyint(pa), hi) - np.polyval(np.polyint(pa), lo)
    it = np.polyval(np.polyint(pt), hi) - np.polyval(np.polyint(pt), lo)
    return (it - ia) / (hi - lo)


def psnr_channel(ref, test, bit_depth=10):
    mx = float((1 << bit_depth) - 1)
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(mx ** 2 / mse)


def ssim_channel(ref, test, bit_depth=10):
    mx = float((1 << bit_depth) - 1)
    r = ref.astype(np.float64) / mx
    t = test.astype(np.float64) / mx
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_r = cv2.GaussianBlur(r, (11, 11), 1.5)
    mu_t = cv2.GaussianBlur(t, (11, 11), 1.5)
    s_r = cv2.GaussianBlur(r ** 2, (11, 11), 1.5) - mu_r ** 2
    s_t = cv2.GaussianBlur(t ** 2, (11, 11), 1.5) - mu_t ** 2
    s_rt = cv2.GaussianBlur(r * t, (11, 11), 1.5) - mu_r * mu_t
    ssim_map = ((2 * mu_r * mu_t + C1) * (2 * s_rt + C2)) / \
               ((mu_r ** 2 + mu_t ** 2 + C1) * (s_r + s_t + C2))
    return float(np.mean(ssim_map))


def read_yuv420(path, h, w):
    y_sz = w * h
    uv_w, uv_h = w // 2, h // 2
    uv_sz = uv_w * uv_h
    data = np.fromfile(path, dtype=np.uint16)
    return data[:y_sz].reshape(h, w), \
           data[y_sz:y_sz + uv_sz].reshape(uv_h, uv_w), \
           data[y_sz + uv_sz:y_sz + 2 * uv_sz].reshape(uv_h, uv_w)


def upsample_chroma(y, u, v):
    h, w = y.shape
    mx = 1023.0
    u_up = np.clip(cv2.resize(u.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), 0, mx)
    v_up = np.clip(cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), 0, mx)
    return np.stack([y.astype(np.float32) / mx, u_up / mx, v_up / mx], axis=0)


def enhanced_to_yuv420(enh, h, w):
    mx = 1023.0
    y = np.clip(enh[0] * mx + 0.5, 0, mx).astype(np.uint16)
    uv_h, uv_w = h // 2, w // 2
    u = np.clip(cv2.resize(enh[1] * mx, (uv_w, uv_h), interpolation=cv2.INTER_CUBIC) + 0.5, 0, mx).astype(np.uint16)
    v = np.clip(cv2.resize(enh[2] * mx, (uv_w, uv_h), interpolation=cv2.INTER_CUBIC) + 0.5, 0, mx).astype(np.uint16)
    return y, u, v


def read_partition_boundary(decoded_dir: str, h: int, w: int):
    """Read boundary_map.raw + boundary_chroma_map.raw → [2, H, W] float32."""
    bnd_luma = os.path.join(decoded_dir, "boundary_map.raw")
    bnd_chroma = os.path.join(decoded_dir, "boundary_chroma_map.raw")
    if not (os.path.exists(bnd_luma) and os.path.exists(bnd_chroma)):
        return None
    bl = np.fromfile(bnd_luma, dtype=np.float16).astype(np.float32).reshape(h, w)
    bc = np.fromfile(bnd_chroma, dtype=np.float16).astype(np.float32).reshape(h, w)
    return np.stack([bl, bc], axis=0)


@torch.no_grad()
def enhance_frame_temporal(
    model, rec_444, qp, boundary_maps, prev_444, next_444,
    device, tile_size=512, overlap=32, temporal_mode="both"
):
    """
    Run temporal model on full frame with tiling.

    Input channels depend on temporal_mode:
      both:      [YUV_curr(3), QP(1), bnd_luma(1), bnd_chroma(1), YUV_prev(3), YUV_next(3)] = 12
      prev_only: [YUV_curr(3), QP(1), bnd_luma(1), bnd_chroma(1), YUV_prev(3)] = 9
      next_only: [YUV_curr(3), QP(1), bnd_luma(1), bnd_chroma(1), YUV_next(3)] = 9
    """
    _, h, w = rec_444.shape
    qp_val = float(qp) / 63.0

    def _build_input(y0a, x0a, th, tw):
        tile_rec = rec_444[:, y0a:y0a+th, x0a:x0a+tw]
        qp_map = np.full((1, th, tw), qp_val, dtype=np.float32)
        tile_bnd = boundary_maps[:, y0a:y0a+th, x0a:x0a+tw]
        parts = [tile_rec, qp_map, tile_bnd]
        if temporal_mode in ("both", "prev_only"):
            parts.append(prev_444[:, y0a:y0a+th, x0a:x0a+tw])
        if temporal_mode in ("both", "next_only"):
            parts.append(next_444[:, y0a:y0a+th, x0a:x0a+tw])
        return np.concatenate(parts, axis=0)

    if h <= tile_size and w <= tile_size:
        inp = _build_input(0, 0, h, w)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        return np.clip(model(inp_t).squeeze(0).cpu().numpy(), 0.0, 1.0)

    out = np.zeros_like(rec_444)
    step = tile_size - overlap
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            y0a = max(0, y1 - tile_size)
            x0a = max(0, x1 - tile_size)
            th, tw = y1 - y0a, x1 - x0a

            inp = _build_input(y0a, x0a, th, tw)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            ot = model(inp_t).squeeze(0).cpu().numpy()

            os_y = overlap // 2 if y0a > 0 else 0
            os_x = overlap // 2 if x0a > 0 else 0
            oe_y = th - (overlap // 2 if y1 < h else 0)
            oe_x = tw - (overlap // 2 if x1 < w else 0)
            out[:, y0a + os_y:y0a + oe_y, x0a + os_x:x0a + oe_x] = \
                ot[:, os_y:oe_y, os_x:oe_x]

    return np.clip(out, 0.0, 1.0)


def discover_ctc_frames(decoded_root, neighbor_root, original_root, qp_list):
    """
    Discover CTC frames that have both prev and next neighbors.
    Returns list of dicts with all paths and metadata.
    """
    records = []
    for qp in qp_list:
        qp_dir = os.path.join(decoded_root, f"QP{qp}")
        if not os.path.isdir(qp_dir):
            print(f"  WARN: missing {qp_dir}")
            continue

        for frame_dir_name in sorted(os.listdir(qp_dir)):
            frame_dir = os.path.join(qp_dir, frame_dir_name)
            if not os.path.isdir(frame_dir):
                continue

            m = re.match(r"(.+)_frame(\d+)$", frame_dir_name)
            if not m:
                continue
            seq_id = m.group(1)
            frame_num = int(m.group(2))

            prev_dir = os.path.join(neighbor_root, f"QP{qp}", f"{seq_id}_frame{frame_num - 1}")
            next_dir = os.path.join(neighbor_root, f"QP{qp}", f"{seq_id}_frame{frame_num + 1}")
            if not (os.path.isdir(prev_dir) and os.path.isdir(next_dir)):
                continue

            rec_path = os.path.join(frame_dir, "rec.yuv")
            prev_path = os.path.join(prev_dir, "rec.yuv")
            next_path = os.path.join(next_dir, "rec.yuv")
            if not (os.path.exists(rec_path) and os.path.exists(prev_path) and os.path.exists(next_path)):
                continue

            ref_path = os.path.join(original_root, f"{frame_dir_name}.yuv")
            if not os.path.exists(ref_path):
                continue

            bs_path = os.path.join(frame_dir, "bitstream.bin")
            if not os.path.exists(bs_path):
                continue

            info_path = os.path.join(frame_dir, "rec.yuv.info")
            w, h = None, None
            if os.path.exists(info_path):
                with open(info_path) as f:
                    for line in f:
                        if line.startswith("width="):
                            w = int(line.strip().split("=")[1])
                        elif line.startswith("height="):
                            h = int(line.strip().split("=")[1])
            if w is None or h is None:
                dm = re.search(r"_(\d+)x(\d+)_", seq_id)
                if dm:
                    w, h = int(dm.group(1)), int(dm.group(2))
                else:
                    continue

            records.append({
                "sequence_id": seq_id,
                "frame_id": frame_num,
                "qp": qp,
                "width": w,
                "height": h,
                "rec_path": rec_path,
                "prev_path": prev_path,
                "next_path": next_path,
                "ref_path": ref_path,
                "bs_path": bs_path,
                "frame_dir": frame_dir,
            })

    return records


def load_model(ckpt_path, model_name, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if model_name is None:
        hp = ckpt.get("hyper_parameters", {})
        gan_cfg = hp.get("gan_config", None)
        if gan_cfg is not None and hasattr(gan_cfg, "generator_name"):
            model_name = gan_cfg.generator_name
        else:
            model_name = hp.get("model_name", "ppff_base")
    state = ckpt["state_dict"]
    in_channels = 12
    for k, v in state.items():
        if "initial_conv" in k and v.dim() == 4:
            in_channels = v.shape[1]
            break
        elif "conv" in k and v.dim() == 4:
            in_channels = v.shape[1]
            break
    overrides = {"in_channels": in_channels}
    model = create_model(model_name, **overrides)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("generator."):
            cleaned[k[len("generator."):]] = v
        elif k.startswith("model."):
            cleaned[k[6:]] = v
        elif k.startswith("discriminator.") or k.startswith("vgg_loss.") or k.startswith("criterion."):
            continue
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned)
    model.to(device).eval()
    print(f"Model loaded: {model_name}, in_channels={in_channels}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Temporal model BD-Rate on CTC (anchor=VTM nofilter)")
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--decoded-root", required=True,
                   help="VTM nofilter decoded CTC root")
    p.add_argument("--neighbor-root", required=True,
                   help="Decoded CTC neighbor frames root")
    p.add_argument("--original-root", required=True,
                   help="Original CTC frames")
    p.add_argument("--anchor-decoded-root", default=None,
                   help="Anchor decoded root (default: same as --decoded-root). "
                        "Use to compare vs VTM with filters.")
    p.add_argument("--model", default="ppff_base")
    p.add_argument("--qp-list", type=int, nargs="+", default=[22, 27, 32, 37])
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--temporal-mode", default="both",
                   choices=["both", "prev_only", "next_only"],
                   help="Which neighbors to use")
    p.add_argument("--output", default="bd_results_temporal_ctc.csv")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.ckpt_path, args.model, device)

    print("Discovering CTC frames with temporal neighbors...")
    records = discover_ctc_frames(
        args.decoded_root, args.neighbor_root, args.original_root, args.qp_list
    )
    print(f"Total records with neighbors: {len(records)}")

    frame_groups = defaultdict(dict)
    for r in records:
        frame_groups[(r["sequence_id"], r["frame_id"])][r["qp"]] = r

    valid_groups = {k: v for k, v in frame_groups.items() if len(v) == len(args.qp_list)}
    print(f"Frame groups with all QPs: {len(valid_groups)}")

    seq_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    total = len(valid_groups)
    anchor_root = args.anchor_decoded_root
    if anchor_root:
        print(f"Using cross-anchor: {anchor_root}")

    for idx, ((seq_id, frame_id), qp_recs) in enumerate(sorted(valid_groups.items())):
        print(f"\r[{idx+1}/{total}] {seq_id} frame{frame_id}       ", end="", flush=True)

        for qp in sorted(args.qp_list):
            rec = qp_recs[qp]
            w, h = rec["width"], rec["height"]

            bits = os.path.getsize(rec["bs_path"]) * 8

            y_ref, u_ref, v_ref = read_yuv420(rec["ref_path"], h, w)
            y_rec, u_rec, v_rec = read_yuv420(rec["rec_path"], h, w)
            y_prev, u_prev, v_prev = read_yuv420(rec["prev_path"], h, w)
            y_next, u_next, v_next = read_yuv420(rec["next_path"], h, w)

            if anchor_root:
                anchor_rec_path = os.path.join(
                    anchor_root, f"QP{qp}",
                    f"{seq_id}_frame{frame_id}", "rec.yuv"
                )
                if not os.path.exists(anchor_rec_path):
                    print(f"\n  WARN: anchor not found: {anchor_rec_path}")
                    continue
                y_anc, u_anc, v_anc = read_yuv420(anchor_rec_path, h, w)
                base_psnr_y = psnr_channel(y_ref, y_anc)
                base_psnr_u = psnr_channel(u_ref, u_anc)
                base_psnr_v = psnr_channel(v_ref, v_anc)
                base_ssim_y = ssim_channel(y_ref, y_anc)
                base_ssim_u = ssim_channel(u_ref, u_anc)
                base_ssim_v = ssim_channel(v_ref, v_anc)
            else:
                base_psnr_y = psnr_channel(y_ref, y_rec)
                base_psnr_u = psnr_channel(u_ref, u_rec)
                base_psnr_v = psnr_channel(v_ref, v_rec)
                base_ssim_y = ssim_channel(y_ref, y_rec)
                base_ssim_u = ssim_channel(u_ref, u_rec)
                base_ssim_v = ssim_channel(v_ref, v_rec)

            rec_444 = upsample_chroma(y_rec, u_rec, v_rec)
            prev_444 = upsample_chroma(y_prev, u_prev, v_prev)
            next_444 = upsample_chroma(y_next, u_next, v_next)

            boundary_maps = read_partition_boundary(rec["frame_dir"], h, w)
            if boundary_maps is None:
                print(f"\n  WARN: no boundary maps for {seq_id} QP{qp} frame{frame_id}, using zeros")
                boundary_maps = np.zeros((2, h, w), dtype=np.float32)

            enh_444 = enhance_frame_temporal(
                model, rec_444, qp, boundary_maps, prev_444, next_444,
                device, args.tile_size, temporal_mode=args.temporal_mode
            )
            y_enh, u_enh, v_enh = enhanced_to_yuv420(enh_444, h, w)

            enh_psnr_y = psnr_channel(y_ref, y_enh)
            enh_psnr_u = psnr_channel(u_ref, u_enh)
            enh_psnr_v = psnr_channel(v_ref, v_enh)
            enh_ssim_y = ssim_channel(y_ref, y_enh)
            enh_ssim_u = ssim_channel(u_ref, u_enh)
            enh_ssim_v = ssim_channel(v_ref, v_enh)

            d = seq_data[seq_id][qp]
            d["bits"].append(bits)
            d["base_psnr_y"].append(base_psnr_y)
            d["base_psnr_u"].append(base_psnr_u)
            d["base_psnr_v"].append(base_psnr_v)
            d["base_ssim_y"].append(base_ssim_y)
            d["base_ssim_u"].append(base_ssim_u)
            d["base_ssim_v"].append(base_ssim_v)
            d["enh_psnr_y"].append(enh_psnr_y)
            d["enh_psnr_u"].append(enh_psnr_u)
            d["enh_psnr_v"].append(enh_psnr_v)
            d["enh_ssim_y"].append(enh_ssim_y)
            d["enh_ssim_u"].append(enh_ssim_u)
            d["enh_ssim_v"].append(enh_ssim_v)

    print()

    print("\n" + "=" * 120)
    print(f"{'Sequence':<40} {'BD-Rate Y':>10} {'BD-Rate U':>10} {'BD-Rate V':>10} "
          f"{'BD-PSNR Y':>10} {'BD-PSNR U':>10} {'BD-PSNR V':>10} "
          f"{'BD-Rate SSIM Y':>14} {'BD-Rate SSIM U':>14} {'BD-Rate SSIM V':>14}")
    print("=" * 120)

    all_bd = defaultdict(list)
    results_rows = []

    for seq_id in sorted(seq_data.keys()):
        qp_dict = seq_data[seq_id]
        if len(qp_dict) < len(args.qp_list):
            continue

        qps = sorted(qp_dict.keys())
        rates = np.array([np.mean(qp_dict[q]["bits"]) for q in qps])

        base_psnr_y = np.array([np.mean(qp_dict[q]["base_psnr_y"]) for q in qps])
        base_psnr_u = np.array([np.mean(qp_dict[q]["base_psnr_u"]) for q in qps])
        base_psnr_v = np.array([np.mean(qp_dict[q]["base_psnr_v"]) for q in qps])
        base_ssim_y = np.array([np.mean(qp_dict[q]["base_ssim_y"]) for q in qps])
        base_ssim_u = np.array([np.mean(qp_dict[q]["base_ssim_u"]) for q in qps])
        base_ssim_v = np.array([np.mean(qp_dict[q]["base_ssim_v"]) for q in qps])

        enh_psnr_y = np.array([np.mean(qp_dict[q]["enh_psnr_y"]) for q in qps])
        enh_psnr_u = np.array([np.mean(qp_dict[q]["enh_psnr_u"]) for q in qps])
        enh_psnr_v = np.array([np.mean(qp_dict[q]["enh_psnr_v"]) for q in qps])
        enh_ssim_y = np.array([np.mean(qp_dict[q]["enh_ssim_y"]) for q in qps])
        enh_ssim_u = np.array([np.mean(qp_dict[q]["enh_ssim_u"]) for q in qps])
        enh_ssim_v = np.array([np.mean(qp_dict[q]["enh_ssim_v"]) for q in qps])

        bdr_y = bd_rate(rates, base_psnr_y, rates, enh_psnr_y)
        bdr_u = bd_rate(rates, base_psnr_u, rates, enh_psnr_u)
        bdr_v = bd_rate(rates, base_psnr_v, rates, enh_psnr_v)

        bdp_y = bd_psnr(rates, base_psnr_y, rates, enh_psnr_y)
        bdp_u = bd_psnr(rates, base_psnr_u, rates, enh_psnr_u)
        bdp_v = bd_psnr(rates, base_psnr_v, rates, enh_psnr_v)

        base_ssim_y_db = -10.0 * np.log10(np.clip(1.0 - base_ssim_y, 1e-12, 1.0))
        base_ssim_u_db = -10.0 * np.log10(np.clip(1.0 - base_ssim_u, 1e-12, 1.0))
        base_ssim_v_db = -10.0 * np.log10(np.clip(1.0 - base_ssim_v, 1e-12, 1.0))
        enh_ssim_y_db = -10.0 * np.log10(np.clip(1.0 - enh_ssim_y, 1e-12, 1.0))
        enh_ssim_u_db = -10.0 * np.log10(np.clip(1.0 - enh_ssim_u, 1e-12, 1.0))
        enh_ssim_v_db = -10.0 * np.log10(np.clip(1.0 - enh_ssim_v, 1e-12, 1.0))

        bdr_ssim_y = bd_rate(rates, base_ssim_y_db, rates, enh_ssim_y_db)
        bdr_ssim_u = bd_rate(rates, base_ssim_u_db, rates, enh_ssim_u_db)
        bdr_ssim_v = bd_rate(rates, base_ssim_v_db, rates, enh_ssim_v_db)

        print(f"{seq_id:<40} {bdr_y:>10.2f}% {bdr_u:>10.2f}% {bdr_v:>10.2f}% "
              f"{bdp_y:>10.4f} {bdp_u:>10.4f} {bdp_v:>10.4f} "
              f"{bdr_ssim_y:>14.2f}% {bdr_ssim_u:>14.2f}% {bdr_ssim_v:>14.2f}%")

        all_bd["bdr_y"].append(bdr_y)
        all_bd["bdr_u"].append(bdr_u)
        all_bd["bdr_v"].append(bdr_v)
        all_bd["bdp_y"].append(bdp_y)
        all_bd["bdp_u"].append(bdp_u)
        all_bd["bdp_v"].append(bdp_v)
        all_bd["bdr_ssim_y"].append(bdr_ssim_y)
        all_bd["bdr_ssim_u"].append(bdr_ssim_u)
        all_bd["bdr_ssim_v"].append(bdr_ssim_v)

        results_rows.append({
            "sequence": seq_id,
            "bdr_psnr_y": bdr_y, "bdr_psnr_u": bdr_u, "bdr_psnr_v": bdr_v,
            "bdp_y": bdp_y, "bdp_u": bdp_u, "bdp_v": bdp_v,
            "bdr_ssim_y": bdr_ssim_y, "bdr_ssim_u": bdr_ssim_u, "bdr_ssim_v": bdr_ssim_v,
        })

    print("-" * 120)
    avg_bdr_y = np.mean(all_bd["bdr_y"])
    avg_bdr_u = np.mean(all_bd["bdr_u"])
    avg_bdr_v = np.mean(all_bd["bdr_v"])
    avg_bdp_y = np.mean(all_bd["bdp_y"])
    avg_bdp_u = np.mean(all_bd["bdp_u"])
    avg_bdp_v = np.mean(all_bd["bdp_v"])
    avg_bdr_ssim_y = np.mean(all_bd["bdr_ssim_y"])
    avg_bdr_ssim_u = np.mean(all_bd["bdr_ssim_u"])
    avg_bdr_ssim_v = np.mean(all_bd["bdr_ssim_v"])

    print(f"{'AVERAGE':<40} {avg_bdr_y:>10.2f}% {avg_bdr_u:>10.2f}% {avg_bdr_v:>10.2f}% "
          f"{avg_bdp_y:>10.4f} {avg_bdp_u:>10.4f} {avg_bdp_v:>10.4f} "
          f"{avg_bdr_ssim_y:>14.2f}% {avg_bdr_ssim_u:>14.2f}% {avg_bdr_ssim_v:>14.2f}%")

    results_rows.append({
        "sequence": "AVERAGE",
        "bdr_psnr_y": avg_bdr_y, "bdr_psnr_u": avg_bdr_u, "bdr_psnr_v": avg_bdr_v,
        "bdp_y": avg_bdp_y, "bdp_u": avg_bdp_u, "bdp_v": avg_bdp_v,
        "bdr_ssim_y": avg_bdr_ssim_y, "bdr_ssim_u": avg_bdr_ssim_u, "bdr_ssim_v": avg_bdr_ssim_v,
    })

    fieldnames = ["sequence", "bdr_psnr_y", "bdr_psnr_u", "bdr_psnr_v",
                  "bdp_y", "bdp_u", "bdp_v",
                  "bdr_ssim_y", "bdr_ssim_u", "bdr_ssim_v"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
