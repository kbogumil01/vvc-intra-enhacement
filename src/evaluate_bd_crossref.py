#!/usr/bin/env python3
"""
Cross-reference BD-Rate evaluation.

Computes BD-Rate with a COMMON anchor (VTM with filters, no network)
for fair comparison between filter and nofilter strategies.

Outputs per sequence:
  1. BD-Rate of (VTM_nofilter) vs (VTM_filter)           → degradation (positive %)
  2. BD-Rate of (VTM_filter + network) vs (VTM_filter)    → filter-net improvement (negative %)
  3. BD-Rate of (VTM_nofilter + network) vs (VTM_filter)  → nofilter-net improvement (negative %)

Usage:
    python evaluate_bd_crossref.py \
        --anchor-decoded-root /data/decoded_CTC_YUV420_10bit \
        --nofilter-decoded-root /data/decoded_CTC_YUV420_10bit_nofilter \
        --original-root /data/splitted_CTC \
        --filter-ckpt-path <v92.ckpt> \
        --nofilter-ckpt-path <v93.ckpt> \
        --model ppff_base \
        --no-split \
        --output bd_results_crossref.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from dataset import build_records, train_val_split_by_sequence
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
def enhance_frame(model, rec_444, qp, device, tile_size=512, overlap=32, partition_maps=None):
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
        inp = torch.from_numpy(inp).unsqueeze(0).to(device)
        return np.clip(model(inp).squeeze(0).cpu().numpy(), 0.0, 1.0)

    out = np.zeros_like(rec_444)
    step = tile_size - overlap
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            y0a = max(0, y1 - tile_size)
            x0a = max(0, x1 - tile_size)
            tile = rec_444[:, y0a:y1, x0a:x1]
            th, tw = tile.shape[1], tile.shape[2]
            inp = _build_input(tile, th, tw, y0a, x0a)
            inp = torch.from_numpy(inp).unsqueeze(0).to(device)
            ot = model(inp).squeeze(0).cpu().numpy()
            os_y = overlap // 2 if y0a > 0 else 0
            os_x = overlap // 2 if x0a > 0 else 0
            oe_y = th - (overlap // 2 if y1 < h else 0)
            oe_x = tw - (overlap // 2 if x1 < w else 0)
            out[:, y0a + os_y:y0a + oe_y, x0a + os_x:x0a + oe_x] = ot[:, os_y:oe_y, os_x:oe_x]
    return np.clip(out, 0.0, 1.0)


def load_model(ckpt_path, model_name, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if model_name is None:
        model_name = ckpt.get("hyper_parameters", {}).get("model_name", "dense_base")
    state = ckpt["state_dict"]
    in_channels = 4
    for k, v in state.items():
        if "initial_conv" in k and v.dim() == 4:
            in_channels = v.shape[1]
            break
        elif "conv" in k and v.dim() == 4:
            in_channels = v.shape[1]
            break
    overrides = {"in_channels": in_channels} if in_channels != 4 else {}
    model = create_model(model_name, **overrides)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k[6:]] = v
        elif not k.startswith("criterion."):
            cleaned[k] = v
    model.load_state_dict(cleaned)
    model.to(device).eval()
    uses_partition = in_channels > 4
    return model, uses_partition, in_channels


def parse_args():
    p = argparse.ArgumentParser(description="Cross-reference BD-Rate: common anchor = VTM with filters")
    p.add_argument("--anchor-decoded-root", required=True,
                   help="VTM with filters decoded root (anchor)")
    p.add_argument("--nofilter-decoded-root", required=True,
                   help="VTM without filters decoded root")
    p.add_argument("--original-root", required=True,
                   help="Original (uncompressed) frames root")
    p.add_argument("--filter-ckpt-path", default=None,
                   help="Checkpoint for filter-trained model (optional)")
    p.add_argument("--nofilter-ckpt-path", required=True,
                   help="Checkpoint for nofilter-trained model")
    p.add_argument("--dbonly-decoded-root", default=None,
                   help="VTM with DB-only (no SAO) decoded root")
    p.add_argument("--dbonly-ckpt-path", default=None,
                   help="Checkpoint for dbonly-trained model")
    p.add_argument("--model", default="ppff_base")
    p.add_argument("--qp-list", type=int, nargs="+", default=[22, 27, 32, 37])
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--output", default="bd_results_crossref.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-split", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--anchor-nofilter", action="store_true",
                   help="Anchor = VTM bez filtrow (zamiast z filtrami)")
    p.add_argument("--partition-channels-nofilter", type=int, nargs="+", default=None,
                   help="Partition channels for nofilter model")
    p.add_argument("--partition-channels-filter", type=int, nargs="+", default=None,
                   help="Partition channels for filter model")
    p.add_argument("--partition-channels-dbonly", type=int, nargs="+", default=None,
                   help="Partition channels for dbonly model")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    nf_model, nf_uses_part, _ = load_model(args.nofilter_ckpt_path, args.model, device)
    nf_part_ch = args.partition_channels_nofilter
    print(f"Nofilter model loaded: {args.nofilter_ckpt_path} (partition={nf_uses_part}, ch={nf_part_ch})")

    f_model, f_uses_part, f_part_ch = None, False, None
    if args.filter_ckpt_path:
        f_model, f_uses_part, _ = load_model(args.filter_ckpt_path, args.model, device)
        f_part_ch = args.partition_channels_filter
        print(f"Filter model loaded: {args.filter_ckpt_path} (partition={f_uses_part}, ch={f_part_ch})")

    db_model, db_uses_part, db_part_ch = None, False, None
    if args.dbonly_ckpt_path:
        db_model, db_uses_part, _ = load_model(args.dbonly_ckpt_path, args.model, device)
        db_part_ch = args.partition_channels_dbonly
        print(f"DB-only model loaded: {args.dbonly_ckpt_path} (partition={db_uses_part}, ch={db_part_ch})")

    anchor_recs = build_records(args.anchor_decoded_root, args.original_root, args.qp_list)
    nf_recs = build_records(args.nofilter_decoded_root, args.original_root, args.qp_list)
    print(f"Anchor records (filter, all): {len(anchor_recs)}")
    print(f"Nofilter records (all):       {len(nf_recs)}")

    db_recs = []
    if args.dbonly_decoded_root:
        db_recs = build_records(args.dbonly_decoded_root, args.original_root, args.qp_list)
        print(f"DB-only records (all):        {len(db_recs)}")

    if not args.no_split:
        _, anchor_recs = train_val_split_by_sequence(anchor_recs, args.val_ratio, args.split_seed)
        _, nf_recs = train_val_split_by_sequence(nf_recs, args.val_ratio, args.split_seed)
        if db_recs:
            _, db_recs = train_val_split_by_sequence(db_recs, args.val_ratio, args.split_seed)
        print(f"After val split: anchor={len(anchor_recs)}, nofilter={len(nf_recs)}")

    nf_index = {}
    for r in nf_recs:
        nf_index[(r["sequence_id"], r["frame_id"], r["qp"])] = r

    db_index = {}
    for r in db_recs:
        db_index[(r["sequence_id"], r["frame_id"], r["qp"])] = r

    anchor_groups = defaultdict(dict)
    for r in anchor_recs:
        anchor_groups[(r["sequence_id"], r["frame_id"])][r["qp"]] = r

    valid_groups = {}
    for key, qp_dict in anchor_groups.items():
        if len(qp_dict) < len(args.qp_list):
            continue
        seq_id, frame_id = key
        all_nf = all((seq_id, frame_id, qp) in nf_index for qp in args.qp_list)
        all_db = (not db_index) or all((seq_id, frame_id, qp) in db_index for qp in args.qp_list)
        if all_nf and all_db:
            valid_groups[key] = qp_dict

    print(f"Valid groups (full QP + matched nofilter): {len(valid_groups)}")

    seq_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    total = len(valid_groups)
    for idx, ((seq_id, frame_id), qp_recs) in enumerate(sorted(valid_groups.items())):
        print(f"\r[{idx+1}/{total}] {seq_id} frame{frame_id}", end="", flush=True)

        for qp in sorted(args.qp_list):
            a_rec = qp_recs[qp]
            nf_rec = nf_index[(seq_id, frame_id, qp)]
            w, h = a_rec["width"], a_rec["height"]

            a_bs = Path(a_rec["rec_path"]).parent / "bitstream.bin"
            nf_bs = Path(nf_rec["rec_path"]).parent / "bitstream.bin"
            if not a_bs.exists() or not nf_bs.exists():
                print(f"\n  WARN: missing bitstream for {seq_id} QP{qp}")
                continue
            a_bits = os.path.getsize(a_bs) * 8
            nf_bits = os.path.getsize(nf_bs) * 8

            y_ref, u_ref, v_ref = read_yuv420(a_rec["ref_path"], h, w)
            y_a, u_a, v_a = read_yuv420(a_rec["rec_path"], h, w)
            y_nf, u_nf, v_nf = read_yuv420(nf_rec["rec_path"], h, w)

            aq_psnr_y = psnr_channel(y_ref, y_a)
            aq_psnr_u = psnr_channel(u_ref, u_a)
            aq_psnr_v = psnr_channel(v_ref, v_a)
            aq_ssim_y = ssim_channel(y_ref, y_a)
            aq_ssim_u = ssim_channel(u_ref, u_a)
            aq_ssim_v = ssim_channel(v_ref, v_a)

            nfq_psnr_y = psnr_channel(y_ref, y_nf)
            nfq_psnr_u = psnr_channel(u_ref, u_nf)
            nfq_psnr_v = psnr_channel(v_ref, v_nf)
            nfq_ssim_y = ssim_channel(y_ref, y_nf)
            nfq_ssim_u = ssim_channel(u_ref, u_nf)
            nfq_ssim_v = ssim_channel(v_ref, v_nf)

            nf_444 = upsample_chroma(y_nf, u_nf, v_nf)
            nf_part = read_partition_maps(str(Path(nf_rec["rec_path"]).parent), h, w, channels=nf_part_ch) if nf_uses_part else None
            nf_enh_444 = enhance_frame(nf_model, nf_444, qp, device, args.tile_size, partition_maps=nf_part)
            y_nfe, u_nfe, v_nfe = enhanced_to_yuv420(nf_enh_444, h, w)
            nfe_psnr_y = psnr_channel(y_ref, y_nfe)
            nfe_psnr_u = psnr_channel(u_ref, u_nfe)
            nfe_psnr_v = psnr_channel(v_ref, v_nfe)
            nfe_ssim_y = ssim_channel(y_ref, y_nfe)
            nfe_ssim_u = ssim_channel(u_ref, u_nfe)
            nfe_ssim_v = ssim_channel(v_ref, v_nfe)

            if f_model is not None:
                a_444 = upsample_chroma(y_a, u_a, v_a)
                a_part = read_partition_maps(str(Path(a_rec["rec_path"]).parent), h, w, channels=f_part_ch) if f_uses_part else None
                a_enh_444 = enhance_frame(f_model, a_444, qp, device, args.tile_size, partition_maps=a_part)
                y_ae, u_ae, v_ae = enhanced_to_yuv420(a_enh_444, h, w)
                fe_psnr_y = psnr_channel(y_ref, y_ae)
                fe_psnr_u = psnr_channel(u_ref, u_ae)
                fe_psnr_v = psnr_channel(v_ref, v_ae)
                fe_ssim_y = ssim_channel(y_ref, y_ae)
                fe_ssim_u = ssim_channel(u_ref, u_ae)
                fe_ssim_v = ssim_channel(v_ref, v_ae)

            if db_model is not None and db_index:
                db_rec = db_index[(seq_id, frame_id, qp)]
                y_db, u_db, v_db = read_yuv420(db_rec["rec_path"], h, w)
                db_bs = Path(db_rec["rec_path"]).parent / "bitstream.bin"
                db_bits = os.path.getsize(db_bs) * 8
                db_444 = upsample_chroma(y_db, u_db, v_db)
                db_part = read_partition_maps(str(Path(db_rec["rec_path"]).parent), h, w, channels=db_part_ch) if db_uses_part else None
                db_enh_444 = enhance_frame(db_model, db_444, qp, device, args.tile_size, partition_maps=db_part)
                y_dbe, u_dbe, v_dbe = enhanced_to_yuv420(db_enh_444, h, w)
                dbe_psnr_y = psnr_channel(y_ref, y_dbe)
                dbe_psnr_u = psnr_channel(u_ref, u_dbe)
                dbe_psnr_v = psnr_channel(v_ref, v_dbe)
                dbe_ssim_y = ssim_channel(y_ref, y_dbe)
                dbe_ssim_u = ssim_channel(u_ref, u_dbe)
                dbe_ssim_v = ssim_channel(v_ref, v_dbe)

            d = seq_data[seq_id][qp]
            d["a_bits"].append(a_bits)
            d["nf_bits"].append(nf_bits)
            d["a_psnr_y"].append(aq_psnr_y); d["a_psnr_u"].append(aq_psnr_u); d["a_psnr_v"].append(aq_psnr_v)
            d["a_ssim_y"].append(aq_ssim_y); d["a_ssim_u"].append(aq_ssim_u); d["a_ssim_v"].append(aq_ssim_v)
            d["nf_psnr_y"].append(nfq_psnr_y); d["nf_psnr_u"].append(nfq_psnr_u); d["nf_psnr_v"].append(nfq_psnr_v)
            d["nf_ssim_y"].append(nfq_ssim_y); d["nf_ssim_u"].append(nfq_ssim_u); d["nf_ssim_v"].append(nfq_ssim_v)
            d["nfe_psnr_y"].append(nfe_psnr_y); d["nfe_psnr_u"].append(nfe_psnr_u); d["nfe_psnr_v"].append(nfe_psnr_v)
            d["nfe_ssim_y"].append(nfe_ssim_y); d["nfe_ssim_u"].append(nfe_ssim_u); d["nfe_ssim_v"].append(nfe_ssim_v)
            if f_model is not None:
                d["fe_psnr_y"].append(fe_psnr_y); d["fe_psnr_u"].append(fe_psnr_u); d["fe_psnr_v"].append(fe_psnr_v)
                d["fe_ssim_y"].append(fe_ssim_y); d["fe_ssim_u"].append(fe_ssim_u); d["fe_ssim_v"].append(fe_ssim_v)
            if db_model is not None and db_index:
                d["db_bits"].append(db_bits)
                d["dbe_psnr_y"].append(dbe_psnr_y); d["dbe_psnr_u"].append(dbe_psnr_u); d["dbe_psnr_v"].append(dbe_psnr_v)
                d["dbe_ssim_y"].append(dbe_ssim_y); d["dbe_ssim_u"].append(dbe_ssim_u); d["dbe_ssim_v"].append(dbe_ssim_v)

    print()

    has_filter_model = f_model is not None
    has_dbonly_model = db_model is not None
    rev = args.anchor_nofilter

    if rev:
        anchor_label = "VTM bez filtrow"
    else:
        anchor_label = "VTM z filtrami"

    hdr = f"{'Sekwencja':<40}"
    if has_filter_model and has_dbonly_model:
        hdr += f" {'filt+siec Y':>12} {'db+siec Y':>12} {'filtry Y':>12} {'bezf+siec Y':>12}"
    elif has_filter_model:
        hdr += f" {'filt+siec Y':>12} {'filtry Y':>12} {'bezf+siec Y':>12}"
    elif has_dbonly_model:
        hdr += f" {'db+siec Y':>12} {'filtry Y':>12} {'bezf+siec Y':>12}"
    else:
        hdr += f" {'filtry Y':>12} {'bezf+siec Y':>12}"
    print("\n" + "=" * len(hdr))
    print(f"BD-Rate (PSNR) wzgl. {anchor_label} (anchor)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    all_bd = defaultdict(list)
    rows = []

    for seq_id in sorted(seq_data.keys()):
        qp_dict = seq_data[seq_id]
        if len(qp_dict) < len(args.qp_list):
            continue

        qps = sorted(qp_dict.keys())
        a_rates = np.array([np.mean(qp_dict[q]["a_bits"]) for q in qps])
        nf_rates = np.array([np.mean(qp_dict[q]["nf_bits"]) for q in qps])

        def avg(key):
            return np.array([np.mean(qp_dict[q][key]) for q in qps])

        def ssim_db(key):
            s = avg(key)
            return -10.0 * np.log10(np.clip(1.0 - s, 1e-12, 1.0))

        if rev:
            anc_rates = nf_rates
            anc_py, anc_pu, anc_pv = avg("nf_psnr_y"), avg("nf_psnr_u"), avg("nf_psnr_v")
            anc_sy, anc_su, anc_sv = "nf_ssim_y", "nf_ssim_u", "nf_ssim_v"
        else:
            anc_rates = a_rates
            anc_py, anc_pu, anc_pv = avg("a_psnr_y"), avg("a_psnr_u"), avg("a_psnr_v")
            anc_sy, anc_su, anc_sv = "a_ssim_y", "a_ssim_u", "a_ssim_v"

        if rev:
            bdr_other_y = bd_rate(anc_rates, anc_py, a_rates, avg("a_psnr_y"))
            bdr_other_u = bd_rate(anc_rates, anc_pu, a_rates, avg("a_psnr_u"))
            bdr_other_v = bd_rate(anc_rates, anc_pv, a_rates, avg("a_psnr_v"))
        else:
            bdr_other_y = bd_rate(anc_rates, anc_py, nf_rates, avg("nf_psnr_y"))
            bdr_other_u = bd_rate(anc_rates, anc_pu, nf_rates, avg("nf_psnr_u"))
            bdr_other_v = bd_rate(anc_rates, anc_pv, nf_rates, avg("nf_psnr_v"))

        bdr_nfnet_y = bd_rate(anc_rates, anc_py, nf_rates, avg("nfe_psnr_y"))
        bdr_nfnet_u = bd_rate(anc_rates, anc_pu, nf_rates, avg("nfe_psnr_u"))
        bdr_nfnet_v = bd_rate(anc_rates, anc_pv, nf_rates, avg("nfe_psnr_v"))

        bdp_other_y = bd_psnr(anc_rates, anc_py, (a_rates if rev else nf_rates),
                              (avg("a_psnr_y") if rev else avg("nf_psnr_y")))
        bdp_nfnet_y = bd_psnr(anc_rates, anc_py, nf_rates, avg("nfe_psnr_y"))

        anc_sdb_y, anc_sdb_u, anc_sdb_v = ssim_db(anc_sy), ssim_db(anc_su), ssim_db(anc_sv)

        if rev:
            bdr_other_ssim_y = bd_rate(anc_rates, anc_sdb_y, a_rates, ssim_db("a_ssim_y"))
            bdr_other_ssim_u = bd_rate(anc_rates, anc_sdb_u, a_rates, ssim_db("a_ssim_u"))
            bdr_other_ssim_v = bd_rate(anc_rates, anc_sdb_v, a_rates, ssim_db("a_ssim_v"))
        else:
            bdr_other_ssim_y = bd_rate(anc_rates, anc_sdb_y, nf_rates, ssim_db("nf_ssim_y"))
            bdr_other_ssim_u = bd_rate(anc_rates, anc_sdb_u, nf_rates, ssim_db("nf_ssim_u"))
            bdr_other_ssim_v = bd_rate(anc_rates, anc_sdb_v, nf_rates, ssim_db("nf_ssim_v"))

        bdr_nfnet_ssim_y = bd_rate(anc_rates, anc_sdb_y, nf_rates, ssim_db("nfe_ssim_y"))
        bdr_nfnet_ssim_u = bd_rate(anc_rates, anc_sdb_u, nf_rates, ssim_db("nfe_ssim_u"))
        bdr_nfnet_ssim_v = bd_rate(anc_rates, anc_sdb_v, nf_rates, ssim_db("nfe_ssim_v"))

        row = {
            "sequence": seq_id,
            "other_bdr_y": bdr_other_y, "other_bdr_u": bdr_other_u, "other_bdr_v": bdr_other_v,
            "other_bdr_ssim_y": bdr_other_ssim_y, "other_bdr_ssim_u": bdr_other_ssim_u, "other_bdr_ssim_v": bdr_other_ssim_v,
            "nfnet_bdr_y": bdr_nfnet_y, "nfnet_bdr_u": bdr_nfnet_u, "nfnet_bdr_v": bdr_nfnet_v,
            "nfnet_bdr_ssim_y": bdr_nfnet_ssim_y, "nfnet_bdr_ssim_u": bdr_nfnet_ssim_u, "nfnet_bdr_ssim_v": bdr_nfnet_ssim_v,
            "bdp_other_y": bdp_other_y, "bdp_nfnet_y": bdp_nfnet_y,
        }

        if has_filter_model:
            bdr_fnet_y = bd_rate(anc_rates, anc_py, a_rates, avg("fe_psnr_y"))
            bdr_fnet_u = bd_rate(anc_rates, anc_pu, a_rates, avg("fe_psnr_u"))
            bdr_fnet_v = bd_rate(anc_rates, anc_pv, a_rates, avg("fe_psnr_v"))
            bdr_fnet_ssim_y = bd_rate(anc_rates, anc_sdb_y, a_rates, ssim_db("fe_ssim_y"))
            bdr_fnet_ssim_u = bd_rate(anc_rates, anc_sdb_u, a_rates, ssim_db("fe_ssim_u"))
            bdr_fnet_ssim_v = bd_rate(anc_rates, anc_sdb_v, a_rates, ssim_db("fe_ssim_v"))
            row.update({
                "fnet_bdr_y": bdr_fnet_y, "fnet_bdr_u": bdr_fnet_u, "fnet_bdr_v": bdr_fnet_v,
                "fnet_bdr_ssim_y": bdr_fnet_ssim_y, "fnet_bdr_ssim_u": bdr_fnet_ssim_u, "fnet_bdr_ssim_v": bdr_fnet_ssim_v,
            })
            all_bd["fnet_bdr_y"].append(bdr_fnet_y)
            all_bd["fnet_bdr_u"].append(bdr_fnet_u)
            all_bd["fnet_bdr_v"].append(bdr_fnet_v)
            all_bd["fnet_bdr_ssim_y"].append(bdr_fnet_ssim_y)
            all_bd["fnet_bdr_ssim_u"].append(bdr_fnet_ssim_u)
            all_bd["fnet_bdr_ssim_v"].append(bdr_fnet_ssim_v)

        if has_dbonly_model:
            db_rates = np.array([np.mean(qp_dict[q]["db_bits"]) for q in qps])
            bdr_dbnet_y = bd_rate(anc_rates, anc_py, db_rates, avg("dbe_psnr_y"))
            bdr_dbnet_u = bd_rate(anc_rates, anc_pu, db_rates, avg("dbe_psnr_u"))
            bdr_dbnet_v = bd_rate(anc_rates, anc_pv, db_rates, avg("dbe_psnr_v"))
            bdr_dbnet_ssim_y = bd_rate(anc_rates, anc_sdb_y, db_rates, ssim_db("dbe_ssim_y"))
            bdr_dbnet_ssim_u = bd_rate(anc_rates, anc_sdb_u, db_rates, ssim_db("dbe_ssim_u"))
            bdr_dbnet_ssim_v = bd_rate(anc_rates, anc_sdb_v, db_rates, ssim_db("dbe_ssim_v"))
            row.update({
                "dbnet_bdr_y": bdr_dbnet_y, "dbnet_bdr_u": bdr_dbnet_u, "dbnet_bdr_v": bdr_dbnet_v,
                "dbnet_bdr_ssim_y": bdr_dbnet_ssim_y, "dbnet_bdr_ssim_u": bdr_dbnet_ssim_u, "dbnet_bdr_ssim_v": bdr_dbnet_ssim_v,
            })
            all_bd["dbnet_bdr_y"].append(bdr_dbnet_y)
            all_bd["dbnet_bdr_u"].append(bdr_dbnet_u)
            all_bd["dbnet_bdr_v"].append(bdr_dbnet_v)
            all_bd["dbnet_bdr_ssim_y"].append(bdr_dbnet_ssim_y)
            all_bd["dbnet_bdr_ssim_u"].append(bdr_dbnet_ssim_u)
            all_bd["dbnet_bdr_ssim_v"].append(bdr_dbnet_ssim_v)

        rows.append(row)

        for k in ["other_bdr_y", "other_bdr_u", "other_bdr_v",
                   "nfnet_bdr_y", "nfnet_bdr_u", "nfnet_bdr_v",
                   "other_bdr_ssim_y", "other_bdr_ssim_u", "other_bdr_ssim_v",
                   "nfnet_bdr_ssim_y", "nfnet_bdr_ssim_u", "nfnet_bdr_ssim_v"]:
            all_bd[k].append(row[k])

        s = seq_id[:38]
        line = f"{s:<40}"
        if has_filter_model and has_dbonly_model:
            line += f" {bdr_fnet_y:>+11.2f}% {bdr_dbnet_y:>+11.2f}% {bdr_other_y:>+11.2f}% {bdr_nfnet_y:>+11.2f}%"
        elif has_filter_model:
            line += f" {bdr_fnet_y:>+11.2f}% {bdr_other_y:>+11.2f}% {bdr_nfnet_y:>+11.2f}%"
        elif has_dbonly_model:
            line += f" {bdr_dbnet_y:>+11.2f}% {bdr_other_y:>+11.2f}% {bdr_nfnet_y:>+11.2f}%"
        else:
            line += f" {bdr_other_y:>+11.2f}% {bdr_nfnet_y:>+11.2f}%"
        print(line)

    print("=" * len(hdr))
    def m(k):
        return np.mean(all_bd[k])

    print(f"\n{'PODSUMOWANIE (BD-Rate PSNR wzgl. ' + anchor_label + ')':>60}")
    print("-" * 70)
    print(f"{'':>30} {'Y':>12} {'U':>12} {'V':>12}")
    other_label = "Z filtrami" if rev else "Bez filtrow (degradacja)"
    print(f"{other_label:<30} {m('other_bdr_y'):>+11.2f}% {m('other_bdr_u'):>+11.2f}% {m('other_bdr_v'):>+11.2f}%")
    print(f"{'Bez filtrow + siec':<30} {m('nfnet_bdr_y'):>+11.2f}% {m('nfnet_bdr_u'):>+11.2f}% {m('nfnet_bdr_v'):>+11.2f}%")
    if has_filter_model:
        print(f"{'Z filtrami + siec':<30} {m('fnet_bdr_y'):>+11.2f}% {m('fnet_bdr_u'):>+11.2f}% {m('fnet_bdr_v'):>+11.2f}%")
    if has_dbonly_model:
        print(f"{'DB-only + siec':<30} {m('dbnet_bdr_y'):>+11.2f}% {m('dbnet_bdr_u'):>+11.2f}% {m('dbnet_bdr_v'):>+11.2f}%")

    print(f"\n{'PODSUMOWANIE (BD-Rate SSIM wzgl. ' + anchor_label + ')':>60}")
    print("-" * 70)
    print(f"{'':>30} {'Y':>12} {'U':>12} {'V':>12}")
    print(f"{other_label:<30} {m('other_bdr_ssim_y'):>+11.2f}% {m('other_bdr_ssim_u'):>+11.2f}% {m('other_bdr_ssim_v'):>+11.2f}%")
    print(f"{'Bez filtrow + siec':<30} {m('nfnet_bdr_ssim_y'):>+11.2f}% {m('nfnet_bdr_ssim_u'):>+11.2f}% {m('nfnet_bdr_ssim_v'):>+11.2f}%")
    if has_filter_model:
        print(f"{'Z filtrami + siec':<30} {m('fnet_bdr_ssim_y'):>+11.2f}% {m('fnet_bdr_ssim_u'):>+11.2f}% {m('fnet_bdr_ssim_v'):>+11.2f}%")
    if has_dbonly_model:
        print(f"{'DB-only + siec':<30} {m('dbnet_bdr_ssim_y'):>+11.2f}% {m('dbnet_bdr_ssim_u'):>+11.2f}% {m('dbnet_bdr_ssim_v'):>+11.2f}%")

    fieldnames = ["sequence",
                  "other_bdr_y", "other_bdr_u", "other_bdr_v",
                  "nfnet_bdr_y", "nfnet_bdr_u", "nfnet_bdr_v",
                  "other_bdr_ssim_y", "other_bdr_ssim_u", "other_bdr_ssim_v",
                  "nfnet_bdr_ssim_y", "nfnet_bdr_ssim_u", "nfnet_bdr_ssim_v",
                  "bdp_other_y", "bdp_nfnet_y"]
    if has_filter_model:
        fieldnames.extend(["fnet_bdr_y", "fnet_bdr_u", "fnet_bdr_v",
                           "fnet_bdr_ssim_y", "fnet_bdr_ssim_u", "fnet_bdr_ssim_v"])
    if has_dbonly_model:
        fieldnames.extend(["dbnet_bdr_y", "dbnet_bdr_u", "dbnet_bdr_v",
                           "dbnet_bdr_ssim_y", "dbnet_bdr_ssim_u", "dbnet_bdr_ssim_v"])

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        avg_row = {"sequence": "AVERAGE"}
        for k in fieldnames[1:]:
            if k in all_bd:
                avg_row[k] = np.mean(all_bd[k])
        w.writerow(avg_row)
    print(f"\nZapisano: {args.output}")


if __name__ == "__main__":
    main()
