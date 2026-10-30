#!/usr/bin/env python3
"""
Model evaluation for VVC post-processing: BD-Rate (PSNR, SSIM) per channel (Y, U, V).

For each validation sequence and each QP:
  1. Load the decoded frame (rec) and the original (ref)
  2. Run rec through the model → enhanced
  3. Compute PSNR and SSIM per channel (Y, U, V) for baseline and enhanced
  4. Read the bitstream size → bits (rate)
  5. Compute BD-Rate from 4 RD points

Usage:
    python evaluate_bd.py --ckpt-path <checkpoint.ckpt>
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from dataset import build_records, train_val_split_by_sequence
from registry import create_model, get_entry, MODEL_REGISTRY

import model_v2
import model_wcdann
import importlib
importlib.import_module("vvc-ppff")


def bd_rate(
    rate_anchor: np.ndarray,
    psnr_anchor: np.ndarray,
    rate_test: np.ndarray,
    psnr_test: np.ndarray,
) -> float:
    """
    Compute BD-Rate from RD curves.
    rate_anchor/test: bitrate (bpp or bits) — 4 points
    psnr_anchor/test: PSNR (or SSIM in dB) — 4 points
    Returns BD-Rate in percent (negative = saving).
    """
    rate_anchor = np.array(rate_anchor, dtype=np.float64)
    psnr_anchor = np.array(psnr_anchor, dtype=np.float64)
    rate_test = np.array(rate_test, dtype=np.float64)
    psnr_test = np.array(psnr_test, dtype=np.float64)

    log_rate_a = np.log(rate_anchor)
    log_rate_t = np.log(rate_test)

    pa = np.polyfit(psnr_anchor, log_rate_a, 3)
    pt = np.polyfit(psnr_test, log_rate_t, 3)

    min_psnr = max(psnr_anchor.min(), psnr_test.min())
    max_psnr = min(psnr_anchor.max(), psnr_test.max())

    if max_psnr <= min_psnr:
        return 0.0

    def polyint(p, a, b):
        pi = np.polyint(p)
        return np.polyval(pi, b) - np.polyval(pi, a)

    int_a = polyint(pa, min_psnr, max_psnr)
    int_t = polyint(pt, min_psnr, max_psnr)

    avg_diff = (int_t - int_a) / (max_psnr - min_psnr)
    return (np.exp(avg_diff) - 1.0) * 100.0


def bd_psnr(
    rate_anchor: np.ndarray,
    psnr_anchor: np.ndarray,
    rate_test: np.ndarray,
    psnr_test: np.ndarray,
) -> float:
    """
    Compute BD-PSNR from RD curves.
    Returns the average PSNR difference (positive = improvement).
    """
    rate_anchor = np.array(rate_anchor, dtype=np.float64)
    psnr_anchor = np.array(psnr_anchor, dtype=np.float64)
    rate_test = np.array(rate_test, dtype=np.float64)
    psnr_test = np.array(psnr_test, dtype=np.float64)

    log_rate_a = np.log(rate_anchor)
    log_rate_t = np.log(rate_test)

    pa = np.polyfit(log_rate_a, psnr_anchor, 3)
    pt = np.polyfit(log_rate_t, psnr_test, 3)

    min_rate = max(log_rate_a.min(), log_rate_t.min())
    max_rate = min(log_rate_a.max(), log_rate_t.max())

    if max_rate <= min_rate:
        return 0.0

    def polyint(p, a, b):
        pi = np.polyint(p)
        return np.polyval(pi, b) - np.polyval(pi, a)

    int_a = polyint(pa, min_rate, max_rate)
    int_t = polyint(pt, min_rate, max_rate)

    return (int_t - int_a) / (max_rate - min_rate)


def psnr_channel(ref: np.ndarray, test: np.ndarray, bit_depth: int = 10) -> float:
    """PSNR of a single channel (2D array, uint16)."""
    max_val = float((1 << bit_depth) - 1)
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse < 1e-12:
        return 99.0
    return 10.0 * np.log10(max_val**2 / mse)


def ssim_channel(ref: np.ndarray, test: np.ndarray, bit_depth: int = 10) -> float:
    """SSIM of a single channel (2D array, uint16 → float64)."""
    max_val = float((1 << bit_depth) - 1)
    r = ref.astype(np.float64) / max_val
    t = test.astype(np.float64) / max_val

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu_r = cv2.GaussianBlur(r, (11, 11), 1.5)
    mu_t = cv2.GaussianBlur(t, (11, 11), 1.5)

    mu_r_sq = mu_r ** 2
    mu_t_sq = mu_t ** 2
    mu_rt = mu_r * mu_t

    sigma_r_sq = cv2.GaussianBlur(r ** 2, (11, 11), 1.5) - mu_r_sq
    sigma_t_sq = cv2.GaussianBlur(t ** 2, (11, 11), 1.5) - mu_t_sq
    sigma_rt = cv2.GaussianBlur(r * t, (11, 11), 1.5) - mu_rt

    ssim_map = ((2 * mu_rt + C1) * (2 * sigma_rt + C2)) / \
               ((mu_r_sq + mu_t_sq + C1) * (sigma_r_sq + sigma_t_sq + C2))
    return float(np.mean(ssim_map))


def read_yuv420(path: str, h: int, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read YUV420 10-bit planar → (Y, U, V) as uint16."""
    y_size = w * h
    uv_w, uv_h = w // 2, h // 2
    uv_size = uv_w * uv_h
    data = np.fromfile(path, dtype=np.uint16)
    y = data[:y_size].reshape(h, w)
    u = data[y_size:y_size + uv_size].reshape(uv_h, uv_w)
    v = data[y_size + uv_size:y_size + 2 * uv_size].reshape(uv_h, uv_w)
    return y, u, v


def upsample_chroma(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """YUV420 → YUV444, returns [3, H, W] float32 [0, 1]."""
    h, w = y.shape
    max_val = 1023.0
    u_up = cv2.resize(u.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    v_up = cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    np.clip(u_up, 0, max_val, out=u_up)
    np.clip(v_up, 0, max_val, out=v_up)
    yuv = np.stack([y.astype(np.float32) / max_val,
                    u_up / max_val,
                    v_up / max_val], axis=0)
    return yuv


def enhanced_to_yuv420(
    enhanced: np.ndarray, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    enhanced: [3, H, W] float32 [0,1] (YUV444)
    Returns (Y, U, V) as uint16, UV at 420 resolution.
    INTER_CUBIC minimizes the 420→444→420 round-trip loss.
    """
    max_val = 1023.0
    y = np.clip(enhanced[0] * max_val + 0.5, 0, max_val).astype(np.uint16)
    u444 = enhanced[1] * max_val
    v444 = enhanced[2] * max_val

    uv_h, uv_w = h // 2, w // 2
    u = np.clip(cv2.resize(u444, (uv_w, uv_h), interpolation=cv2.INTER_CUBIC) + 0.5,
                0, max_val).astype(np.uint16)
    v = np.clip(cv2.resize(v444, (uv_w, uv_h), interpolation=cv2.INTER_CUBIC) + 0.5,
                0, max_val).astype(np.uint16)
    return y, u, v


def read_partition_maps(decoded_dir: str, h: int, w: int, channels: list = None):
    """Read partition maps → [N, H, W] float32 or None. channels selects subset."""
    names = ["depth_map.raw", "boundary_map.raw",
             "depth_chroma_map.raw", "boundary_chroma_map.raw"]
    paths = [os.path.join(decoded_dir, n) for n in names]
    if not all(os.path.exists(p) for p in paths):
        return None
    all_channels = []
    for p in paths:
        ch = np.fromfile(p, dtype=np.float16).astype(np.float32).reshape(h, w)
        all_channels.append(ch)
    if channels is not None:
        all_channels = [all_channels[i] for i in channels]
    return np.stack(all_channels, axis=0)


@torch.no_grad()
def enhance_frame(
    model: DenseVVCEnhancer,
    rec_444: np.ndarray,
    qp: int,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 32,
    partition_maps: np.ndarray = None,
) -> np.ndarray:
    """
    Run a full frame through the model with tiling (avoids OOM).
    rec_444: [3, H, W] float32 [0,1]
    partition_maps: [2, H, W] float32 (depth, boundary) or None
    Returns: [3, H, W] float32 [0,1]
    """
    _, h, w = rec_444.shape
    qp_val = float(qp) / 63.0

    def _build_input(tile_yuv, tile_h, tile_w, y0a=None, x0a=None):
        qp_map = np.full((1, tile_h, tile_w), qp_val, dtype=np.float32)
        parts = [tile_yuv, qp_map]
        if partition_maps is not None:
            if y0a is not None:
                parts.append(partition_maps[:, y0a:y0a+tile_h, x0a:x0a+tile_w])
            else:
                parts.append(partition_maps)
        return np.concatenate(parts, axis=0)

    if h <= tile_size and w <= tile_size:
        inp = _build_input(rec_444, h, w)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        out_t = model(inp_t)
        return np.clip(out_t.squeeze(0).cpu().numpy(), 0.0, 1.0)

    out = np.zeros_like(rec_444)
    step = tile_size - overlap

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            y0_adj = max(0, y1 - tile_size)
            x0_adj = max(0, x1 - tile_size)

            tile = rec_444[:, y0_adj:y1, x0_adj:x1]
            th, tw = tile.shape[1], tile.shape[2]
            inp = _build_input(tile, th, tw, y0_adj, x0_adj)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out_tile = model(inp_t).squeeze(0).cpu().numpy()

            oy_start = overlap // 2 if y0_adj > 0 else 0
            ox_start = overlap // 2 if x0_adj > 0 else 0
            oy_end = th - (overlap // 2 if y1 < h else 0)
            ox_end = tw - (overlap // 2 if x1 < w else 0)

            out[:, y0_adj + oy_start:y0_adj + oy_end,
                   x0_adj + ox_start:x0_adj + ox_end] = \
                out_tile[:, oy_start:oy_end, ox_start:ox_end]

    return np.clip(out, 0.0, 1.0)


def parse_args():
    p = argparse.ArgumentParser(description="Ewaluacja BD-Rate modelu DenseNet VVC")
    p.add_argument("--ckpt-path", type=str, required=True,
                   help="Ścieżka do checkpointu (.ckpt)")
    p.add_argument("--decoded-root", type=str,
                   default="/path/to/decoded_root")
    p.add_argument("--original-root", type=str,
                   default="/path/to/original_root")
    p.add_argument("--qp-list", type=int, nargs="+", default=[22, 27, 32, 37])
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--output", type=str, default="bd_results.csv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-split", action="store_true",
                   help="Użyj wszystkich rekordów (bez podziału train/val, np. dla CTC)")
    p.add_argument("--use-train", action="store_true",
                   help="Ewaluuj na zbiorze treningowym (zamiast walidacyjnym)")
    p.add_argument("--model", type=str, default=None,
                   help="Nazwa modelu z registry (auto-detect z checkpointu jeśli pominięte)")
    p.add_argument("--partition-maps", action="store_true",
                   help="Użyj map partycji (depth_map.raw + boundary_map.raw) jako dodatkowych kanałów")
    p.add_argument("--partition-channels", type=int, nargs="+", default=None,
                   help="Which partition channels to use (0=luma_depth, 1=luma_boundary, "
                        "2=chroma_depth, 3=chroma_boundary). Default: all from checkpoint.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

    model_name = args.model
    if model_name is None:
        hp = ckpt.get("hyper_parameters", {})
        model_name = hp.get("model_name", "dense_base")
    print(f"Model: {model_name}")

    state = ckpt["state_dict"]

    in_channels = 4
    for k, v in state.items():
        if "initial_conv" in k and v.dim() == 4:
            in_channels = v.shape[1]
            break
        elif "conv" in k and v.dim() == 4:
            in_channels = v.shape[1]
            break
    use_partition = args.partition_maps or in_channels > 4
    part_ch = args.partition_channels
    if part_ch is None and in_channels > 4:
        n_part = in_channels - 4
        if n_part < 4:
            part_ch = None
        else:
            part_ch = None
    if use_partition:
        print(f"Partition maps: ENABLED (in_channels={in_channels}, channels={part_ch})")

    overrides = {"in_channels": in_channels} if in_channels != 4 else {}
    model = create_model(model_name, **overrides)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        elif k.startswith("criterion."):
            continue
        else:
            cleaned[k] = v
    model.load_state_dict(cleaned)
    model.to(device).eval()
    print(f"Model załadowany z: {args.ckpt_path}")

    records = build_records(args.decoded_root, args.original_root, args.qp_list)
    if args.no_split:
        eval_recs = records
        print(f"Rekordów (wszystkie, --no-split): {len(eval_recs)}")
    elif args.use_train:
        eval_recs, _ = train_val_split_by_sequence(records, args.val_ratio, args.split_seed)
        print(f"Rekordów treningowych: {len(eval_recs)}")
    else:
        _, eval_recs = train_val_split_by_sequence(records, args.val_ratio, args.split_seed)
        print(f"Rekordów walidacyjnych: {len(eval_recs)}")

    frame_groups: Dict[Tuple[str, int], Dict[int, dict]] = defaultdict(dict)
    for r in eval_recs:
        key = (r["sequence_id"], r["frame_id"])
        frame_groups[key][r["qp"]] = r

    valid_groups = {k: v for k, v in frame_groups.items() if len(v) == len(args.qp_list)}
    print(f"Grup (sekw+klatka) z pełnym zestawem QP: {len(valid_groups)}")

    seq_data: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    total = len(valid_groups)
    for idx, ((seq_id, frame_id), qp_recs) in enumerate(sorted(valid_groups.items())):
        print(f"\r[{idx+1}/{total}] {seq_id} frame{frame_id}", end="", flush=True)

        for qp in sorted(args.qp_list):
            rec = qp_recs[qp]
            w, h = rec["width"], rec["height"]
            decoded_dir = Path(rec["rec_path"]).parent

            bs_path = decoded_dir / "bitstream.bin"
            if bs_path.exists():
                bits = os.path.getsize(bs_path) * 8
            else:
                print(f"\n  WARN: brak bitstream.bin dla {seq_id} QP{qp}")
                continue

            y_ref, u_ref, v_ref = read_yuv420(rec["ref_path"], h, w)
            y_rec, u_rec, v_rec = read_yuv420(rec["rec_path"], h, w)

            base_psnr_y = psnr_channel(y_ref, y_rec)
            base_psnr_u = psnr_channel(u_ref, u_rec)
            base_psnr_v = psnr_channel(v_ref, v_rec)
            base_ssim_y = ssim_channel(y_ref, y_rec)
            base_ssim_u = ssim_channel(u_ref, u_rec)
            base_ssim_v = ssim_channel(v_ref, v_rec)

            rec_444 = upsample_chroma(y_rec, u_rec, v_rec)
            part_maps = None
            if use_partition:
                part_maps = read_partition_maps(str(decoded_dir), h, w, channels=part_ch)
                if part_maps is None:
                    print(f"\n  WARN: brak partition maps dla {seq_id} QP{qp}")
            enh_444 = enhance_frame(model, rec_444, qp, device, args.tile_size,
                                    partition_maps=part_maps)
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

    print("\n" + "=" * 100)
    print(f"{'Sequence':<60} {'BD-Rate Y':>10} {'BD-Rate U':>10} {'BD-Rate V':>10} "
          f"{'BD-PSNR Y':>10} {'BD-PSNR U':>10} {'BD-PSNR V':>10}")
    print("=" * 100)

    all_bd = defaultdict(list)
    results_rows = []

    for seq_id in sorted(seq_data.keys()):
        qp_dict = seq_data[seq_id]
        if len(qp_dict) < 4:
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

        seq_short = seq_id[:58]
        print(f"{seq_short:<60} {bdr_y:>+10.2f}% {bdr_u:>+10.2f}% {bdr_v:>+10.2f}% "
              f"{bdp_y:>+10.3f} {bdp_u:>+10.3f} {bdp_v:>+10.3f}")

        all_bd["bdr_psnr_y"].append(bdr_y)
        all_bd["bdr_psnr_u"].append(bdr_u)
        all_bd["bdr_psnr_v"].append(bdr_v)
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

    print("=" * 100)
    print(f"{'AVERAGE':<60} "
          f"{np.mean(all_bd['bdr_psnr_y']):>+10.2f}% "
          f"{np.mean(all_bd['bdr_psnr_u']):>+10.2f}% "
          f"{np.mean(all_bd['bdr_psnr_v']):>+10.2f}% "
          f"{np.mean(all_bd['bdp_y']):>+10.3f} "
          f"{np.mean(all_bd['bdp_u']):>+10.3f} "
          f"{np.mean(all_bd['bdp_v']):>+10.3f}")
    print()
    print("BD-Rate SSIM (średnia):")
    print(f"  Y: {np.mean(all_bd['bdr_ssim_y']):+.2f}%  "
          f"U: {np.mean(all_bd['bdr_ssim_u']):+.2f}%  "
          f"V: {np.mean(all_bd['bdr_ssim_v']):+.2f}%")

    import csv
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence",
            "bdr_psnr_y", "bdr_psnr_u", "bdr_psnr_v",
            "bdp_y", "bdp_u", "bdp_v",
            "bdr_ssim_y", "bdr_ssim_u", "bdr_ssim_v",
        ])
        w.writeheader()
        w.writerows(results_rows)
        w.writerow({
            "sequence": "AVERAGE",
            "bdr_psnr_y": np.mean(all_bd["bdr_psnr_y"]),
            "bdr_psnr_u": np.mean(all_bd["bdr_psnr_u"]),
            "bdr_psnr_v": np.mean(all_bd["bdr_psnr_v"]),
            "bdp_y": np.mean(all_bd["bdp_y"]),
            "bdp_u": np.mean(all_bd["bdp_u"]),
            "bdp_v": np.mean(all_bd["bdp_v"]),
            "bdr_ssim_y": np.mean(all_bd["bdr_ssim_y"]),
            "bdr_ssim_u": np.mean(all_bd["bdr_ssim_u"]),
            "bdr_ssim_v": np.mean(all_bd["bdr_ssim_v"]),
        })
    print(f"\nZapisano wyniki do: {args.output}")


if __name__ == "__main__":
    main()
