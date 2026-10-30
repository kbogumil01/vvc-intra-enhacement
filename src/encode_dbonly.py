#!/usr/bin/env python3
"""
Re-encode existing single-frame YUV files with VTM All Intra mode
but with only Deblocking Filter enabled (SAO, ALF, CCALF disabled).

Reuses already-extracted frames from splitted_CTC / splitted_DVI_BVC / splitted_DIV2K.
Output goes to decoded_*_dbonly/ directories.

Usage:
    python encode_dbonly.py --dataset ctc
    python encode_dbonly.py --dataset bvi
    python encode_dbonly.py --dataset div2k
    python encode_dbonly.py --dataset all
"""

import argparse
import multiprocessing
import os
import subprocess
from itertools import product
from pathlib import Path

DATA = Path("/net/tscratch/people/plgkarolbogumil/data")
VTM_ROOT = Path("/net/people/plgrid/plgkarolbogumil/mgr/dense/VVCSoftware_VTM")
ENCODER = VTM_ROOT / "bin" / "umake" / "gcc-13.3" / "x86_64" / "release" / "EncoderApp"
CFG = VTM_ROOT / "cfg" / "encoder_intra_vtm_dbonly.cfg"

QP_VALUES = [22, 27, 32, 37]

DATASETS = {
    "ctc": {
        "src": DATA / "splitted_CTC",
        "dst": DATA / "decoded_CTC_YUV420_10bit_dbonly",
    },
    "bvi": {
        "src": DATA / "splitted_DVI_BVC",
        "dst": DATA / "decoded_BVI_frames_YUV420_10bit_dbonly",
    },
    "div2k": {
        "src": DATA / "splitted_DIV2K",
        "dst": DATA / "decoded_DIV2K_YUV420_10bit_dbonly",
    },
}


def parse_info(info_path: Path) -> dict:
    info = {}
    with open(info_path) as f:
        for line in f:
            key, val = line.strip().split("=", 1)
            info[key] = val
    return info


def encode_one(args: tuple) -> str:
    """Encode a single frame at a given QP with only DB enabled."""
    yuv_path, qp, dst_dir = args
    stem = yuv_path.stem

    info_path = yuv_path.with_suffix(".yuv.info")
    if not info_path.exists():
        return f"SKIP (no info): {stem} QP{qp}"

    info = parse_info(info_path)
    width = int(info["width"])
    height = int(info["height"])

    out_dir = dst_dir / f"QP{qp}" / stem
    rec_path = out_dir / "rec.yuv"
    bin_path = out_dir / "bitstream.bin"
    info_out = out_dir / "rec.yuv.info"

    if rec_path.exists() and bin_path.exists():
        return f"EXISTS: {stem} QP{qp}"

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ENCODER),
        "-c", str(CFG),
        "-i", str(yuv_path),
        "-b", str(bin_path),
        "-o", str(rec_path),
        "-wdt", str(width),
        "-hgt", str(height),
        "-q", str(qp),
        "-f", "1",
        "-fr", "1",
        "--InputBitDepth=10",
        "--InputChromaFormat=420",
        "--TemporalSubsampleRatio=1",
        "--CCALF=0",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        rec_path.unlink(missing_ok=True)
        bin_path.unlink(missing_ok=True)
        stderr_tail = result.stderr[-300:] if result.stderr else result.stdout[-300:]
        return f"FAIL: {stem} QP{qp} — {stderr_tail}"

    with open(info_out, "w") as f:
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"num_frames=1\n")
        f.write(f"pix_fmt=yuv420p10le\n")
        f.write(f"qp={qp}\n")

    bin_size = bin_path.stat().st_size if bin_path.exists() else 0
    return f"OK: {stem} QP{qp} (bin={bin_size} bytes)"


def main():
    parser = argparse.ArgumentParser(
        description="Encode frames with VTM All Intra, only DB enabled (SAO/ALF/CCALF off)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ctc", "bvi", "div2k", "all"],
                        help="Which dataset to encode")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--qp", type=int, nargs="+", default=QP_VALUES,
                        help="QP values (default: 22 27 32 37)")
    args = parser.parse_args()

    datasets_to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds_name in datasets_to_process:
        ds = DATASETS[ds_name]
        src_dir = ds["src"]
        dst_dir = ds["dst"]

        if not src_dir.exists():
            print(f"SKIP: {src_dir} does not exist")
            continue

        yuv_files = sorted(src_dir.glob("*.yuv"))
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"Source: {src_dir} ({len(yuv_files)} files)")
        print(f"Output: {dst_dir}")
        print(f"Config: {CFG}")
        print(f"QP: {args.qp}")
        print(f"Encoder: {ENCODER}")
        print(f"{'='*60}")

        dst_dir.mkdir(parents=True, exist_ok=True)

        jobs = [(yuv, qp, dst_dir) for yuv, qp in product(yuv_files, args.qp)]
        total = len(jobs)
        print(f"Total encode jobs: {total}")

        n_workers = args.workers or multiprocessing.cpu_count()
        print(f"Workers: {n_workers}\n")

        ok = fail = skip = 0
        with multiprocessing.Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(encode_one, jobs), 1):
                if result.startswith("OK"):
                    ok += 1
                elif result.startswith("FAIL"):
                    fail += 1
                    print(f"[{i}/{total}] {result}")
                else:
                    skip += 1
                if i % 50 == 0 or i == total:
                    print(f"[{i}/{total}] ok={ok} fail={fail} skip={skip}")

        print(f"\nDone: {ds_name} — ok={ok}, fail={fail}, skip={skip}")


if __name__ == "__main__":
    main()
