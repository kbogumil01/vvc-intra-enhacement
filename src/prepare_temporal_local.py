#!/usr/bin/env python3
"""
Prepare temporal data for training the model with neighbors.

Phase 1: BVI non-4K (1920x1088, 960x544, 480x272) + XIPH (train+test)
Phase 2: BVI 4K (3840x2176)

Does:
  --phase split:  Extract frames from sequences (curr + prev/next)
  --phase encode: VTM nofilter encoding (curr with trace → maps, neighbors without trace)
  --phase maps:   Generate partition maps from trace CSV (curr XIPH only)
  --phase verify: Verify data correctness
  --phase all:    All of the above

Output structure:
  /path/to/dograna_data/
    splitted_XIPH/
      {seq}_frame{N}.yuv + .yuv.info
    decoded_XIPH_nofilter/
      QP{22,27,32,37}/{seq}_frame{N}/
        rec.yuv, bitstream.bin, rec.yuv.info [, bitstream_partition.csv, *.raw]
    splitted_BVI_neighbors/
      {seq}_frame{N}.yuv + .yuv.info
    decoded_BVI_neighbors_nofilter/
      QP{22,27,32,37}/{seq}_frame{N}/
        rec.yuv, bitstream.bin, rec.yuv.info
    splitted_CTC_neighbors/
      {seq}_frame{N}.yuv + .yuv.info
    decoded_CTC_neighbors_nofilter/
      QP{22,27,32,37}/{seq}_frame{N}/
        rec.yuv, bitstream.bin, rec.yuv.info

Usage:
    # Dry run — verify what exists:
    python prepare_temporal_local.py --phase verify

    # Phase 1 only (without 4K):
    python prepare_temporal_local.py --phase all --no-4k --workers 8

    # Phase 2 (4K only):
    python prepare_temporal_local.py --phase all --only-4k --workers 4

    # CTC neighbors (for evaluation):
    python prepare_temporal_local.py --phase all --datasets ctc --workers 8
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/path/to/data")
OUTPUT_ROOT = Path("/path/to/dograna_data")

XIPH_TRAIN_DIR = DATA_ROOT / "XIPH_train"
XIPH_TEST_DIR = DATA_ROOT / "XIPH_test"
BVI_SRC_DIR = DATA_ROOT / "BVI_DVC_YUV420_10bit"
CTC_ROOT = DATA_ROOT / "ctc"

CTC_SPLITTED = DATA_ROOT / "splitted_CTC"
CTC_DECODED_NOFILTER = DATA_ROOT / "decoded_CTC_YUV420_10bit_nofilter"

XIPH_SPLITTED = OUTPUT_ROOT / "splitted_XIPH"
XIPH_DECODED = OUTPUT_ROOT / "decoded_XIPH_nofilter"
BVI_NB_SPLITTED = OUTPUT_ROOT / "splitted_BVI_neighbors"
BVI_NB_DECODED = OUTPUT_ROOT / "decoded_BVI_neighbors_nofilter"
CTC_NB_SPLITTED = OUTPUT_ROOT / "splitted_CTC_neighbors"
CTC_NB_DECODED = OUTPUT_ROOT / "decoded_CTC_neighbors_nofilter"

ENCODER_TRACE = Path("/path/to/VVCSoftware_VTM/build_trace/source/App/EncoderApp/EncoderApp")
ENCODER_NOTRACE = Path("/path/to/VVCSoftware_VTM/build_notrace/source/App/EncoderApp/EncoderApp")
CFG_NOFILTER = Path("/path/to/VVCSoftware_VTM/cfg/encoder_intra_vtm_nofilter.cfg")

PARTITION_MAP_SCRIPT = Path("/path/to/csv_to_partition_maps.py")

QP_VALUES = [22, 27, 32, 37]

TARGET_FRAMES = [4, 19, 34, 49, 60]
NEIGHBOR_OFFSETS = [-1, +1]

RES_4K = "3840x2176"

CTC_SEQUENCES = [
    ("A1", "Tango2", 3840, 2160, 10),
    ("A1", "FoodMarket4", 3840, 2160, 10),
    ("A1", "Campfire", 3840, 2160, 10),
    ("A2", "CatRobot", 3840, 2160, 10),
    ("A2", "DaylightRoad2", 3840, 2160, 10),
    ("A2", "ParkRunning3", 3840, 2160, 10),
    ("B", "MarketPlace", 1920, 1080, 10),
    ("B", "RitualDance", 1920, 1080, 10),
    ("B", "Cactus", 1920, 1080, 8),
    ("B", "BasketballDrive", 1920, 1080, 8),
    ("B", "BQTerrace", 1920, 1080, 8),
    ("C", "BasketballDrill", 832, 480, 8),
    ("C", "BQMall", 832, 480, 8),
    ("C", "PartyScene", 832, 480, 8),
    ("C", "RaceHorses", 832, 480, 8),
    ("D", "BasketballPass", 416, 240, 8),
    ("D", "BQSquare", 416, 240, 8),
    ("D", "BlowingBubbles", 416, 240, 8),
    ("D", "RaceHorsesLow", 416, 240, 8),
    ("E", "FourPeople", 1280, 720, 8),
    ("E", "Johnny", 1280, 720, 8),
    ("E", "KristenAndSara", 1280, 720, 8),
]


def frame_size_bytes(w: int, h: int, bit_depth: int) -> int:
    bps = 2 if bit_depth == 10 else 1
    return int(w * h * 1.5) * bps


def convert_8bit_to_10bit(data_8bit: bytes, w: int, h: int) -> bytes:
    arr_8 = np.frombuffer(data_8bit, dtype=np.uint8)
    arr_10 = arr_8.astype(np.uint16) << 2
    return arr_10.tobytes()


def extract_frame(yuv_path: Path, w: int, h: int, bit_depth: int,
                  frame_idx: int) -> bytes | None:
    fsize = frame_size_bytes(w, h, bit_depth)
    offset = frame_idx * fsize
    file_size = yuv_path.stat().st_size
    if offset + fsize > file_size:
        return None
    with open(yuv_path, "rb") as f:
        f.seek(offset)
        data = f.read(fsize)
    return data if len(data) == fsize else None


def get_num_frames(yuv_path: Path, w: int, h: int, bit_depth: int) -> int:
    return yuv_path.stat().st_size // frame_size_bytes(w, h, bit_depth)


def write_frame(data: bytes, out_path: Path, w: int, h: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(data)
    with open(out_path.with_suffix(".yuv.info"), "w") as f:
        f.write(f"width={w}\nheight={h}\nnum_frames=1\npix_fmt=yuv420p10le\n")


def parse_xiph_resolution(yuv_path: Path) -> tuple[int, int] | None:
    """Get resolution from .y4m.info or filename heuristics."""
    info_path = yuv_path.with_suffix(".y4m.info")
    if info_path.exists():
        w = h = None
        with open(info_path) as f:
            for line in f:
                if "Width" in line and "pixels" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val = parts[-1].strip().replace(" ", "").replace("pixels", "")
                        if val.isdigit():
                            w = int(val)
                elif "Height" in line and "pixels" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val = parts[-1].strip().replace(" ", "").replace("pixels", "")
                        if val.isdigit():
                            h = int(val)
        if w and h:
            return w, h

    name = yuv_path.stem.lower()
    known = [
        ("qcif", 176, 144), ("cif", 352, 288), ("4cif", 704, 576),
        ("720p", 1280, 720), ("1080p", 1920, 1080),
        ("1280x720", 1280, 720), ("1920x1080", 1920, 1080),
    ]
    for key, w, h in known:
        if key in name:
            expected = frame_size_bytes(w, h, 8)
            if yuv_path.stat().st_size % expected == 0:
                return w, h

    m = re.search(r"(\d{3,4})x(\d{3,4})", yuv_path.stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_bvi_info(info_path: Path) -> tuple[int, int] | None:
    info = {}
    with open(info_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                info[k.strip()] = v.strip()
    w = int(info.get("width", 0))
    h = int(info.get("height", 0))
    return (w, h) if w and h else None


def split_xiph():
    """Split XIPH train+test: curr (TARGET_FRAMES) + neighbors, convert 8→10bit."""
    print("\n=== Splitting XIPH (train + test) ===")
    XIPH_SPLITTED.mkdir(parents=True, exist_ok=True)

    all_dirs = []
    if XIPH_TRAIN_DIR.exists():
        all_dirs.append(("train", XIPH_TRAIN_DIR))
    if XIPH_TEST_DIR.exists():
        all_dirs.append(("test", XIPH_TEST_DIR))

    total_written = 0
    total_skipped = 0

    for split_name, src_dir in all_dirs:
        yuv_files = sorted(src_dir.glob("*.yuv"))
        print(f"  {split_name}: {len(yuv_files)} sequences")

        for yuv_path in yuv_files:
            res = parse_xiph_resolution(yuv_path)
            if res is None:
                print(f"    SKIP (no resolution): {yuv_path.name}")
                continue
            w, h = res
            num_frames = get_num_frames(yuv_path, w, h, bit_depth=8)
            seq_name = yuv_path.stem

            all_indices = set()
            for t in TARGET_FRAMES:
                if t < num_frames:
                    all_indices.add(t)
                    for offset in NEIGHBOR_OFFSETS:
                        nb = t + offset
                        if 0 <= nb < num_frames:
                            all_indices.add(nb)

            for idx in sorted(all_indices):
                out_path = XIPH_SPLITTED / f"{seq_name}_frame{idx}.yuv"
                if out_path.exists():
                    total_skipped += 1
                    continue

                data = extract_frame(yuv_path, w, h, bit_depth=8, frame_idx=idx)
                if data is None:
                    print(f"    WARN: cannot read frame {idx} from {yuv_path.name}")
                    continue

                data_10 = convert_8bit_to_10bit(data, w, h)
                write_frame(data_10, out_path, w, h)
                total_written += 1

    print(f"  Done: {total_written} written, {total_skipped} skipped")


def split_bvi_neighbors(only_4k: bool = False, no_4k: bool = False):
    """Split BVI-DVC neighbor frames (frame±1 around TARGET_FRAMES)."""
    label = "4K only" if only_4k else "no-4K" if no_4k else "all"
    print(f"\n=== Splitting BVI-DVC neighbors ({label}) ===")
    BVI_NB_SPLITTED.mkdir(parents=True, exist_ok=True)

    if not BVI_SRC_DIR.exists():
        print(f"  ERROR: {BVI_SRC_DIR} not found")
        return

    yuv_files = sorted(BVI_SRC_DIR.glob("*.yuv"))
    if only_4k:
        yuv_files = [f for f in yuv_files if RES_4K in f.name]
    elif no_4k:
        yuv_files = [f for f in yuv_files if RES_4K not in f.name]

    print(f"  {len(yuv_files)} source sequences")

    total_written = 0
    total_skipped = 0

    for yuv_path in yuv_files:
        info_path = yuv_path.with_suffix(".yuv.info")
        if not info_path.exists():
            continue
        res = parse_bvi_info(info_path)
        if res is None:
            continue
        w, h = res
        num_frames = get_num_frames(yuv_path, w, h, bit_depth=10)

        neighbor_indices = set()
        for t in TARGET_FRAMES:
            if t < num_frames:
                for offset in NEIGHBOR_OFFSETS:
                    nb = t + offset
                    if 0 <= nb < num_frames and nb not in TARGET_FRAMES:
                        neighbor_indices.add(nb)

        for idx in sorted(neighbor_indices):
            out_path = BVI_NB_SPLITTED / f"{yuv_path.stem}_frame{idx}.yuv"
            if out_path.exists():
                total_skipped += 1
                continue

            data = extract_frame(yuv_path, w, h, bit_depth=10, frame_idx=idx)
            if data is None:
                print(f"    WARN: cannot read frame {idx} from {yuv_path.name}")
                continue

            write_frame(data, out_path, w, h)
            total_written += 1

    print(f"  Done: {total_written} written, {total_skipped} skipped")


def split_ctc_neighbors():
    """Split CTC neighbor frames."""
    print("\n=== Splitting CTC neighbors ===")
    CTC_NB_SPLITTED.mkdir(parents=True, exist_ok=True)

    if not CTC_ROOT.exists():
        print(f"  ERROR: {CTC_ROOT} not found")
        return

    existing_frames = defaultdict(list)
    if CTC_SPLITTED.exists():
        for f in CTC_SPLITTED.glob("*.yuv"):
            m = re.match(r"^(.+)_frame(\d+)\.yuv$", f.name)
            if m:
                existing_frames[m.group(1)].append(int(m.group(2)))

    total_written = 0
    total_skipped = 0

    for cls, name, w, h, bit_depth in CTC_SEQUENCES:
        cls_dir = CTC_ROOT / cls
        src_yuv_list = list(cls_dir.glob(f"{name}*"))
        src_yuv = [c for c in src_yuv_list if c.suffix == ".yuv"]
        if not src_yuv:
            print(f"    SKIP: no .yuv for {name} in {cls_dir}")
            continue
        src_yuv = src_yuv[0]
        num_frames = get_num_frames(src_yuv, w, h, bit_depth)

        seq_key = f"{name}_{w}x{h}_10bit_420"
        curr_frames = sorted(existing_frames.get(seq_key, []))
        if not curr_frames:
            for key in existing_frames:
                if name in key:
                    seq_key = key
                    curr_frames = sorted(existing_frames[key])
                    break
        if not curr_frames:
            print(f"    WARN: no existing CTC frames for {name}")
            continue

        for t in curr_frames:
            for offset in NEIGHBOR_OFFSETS:
                nb = t + offset
                if nb < 0 or nb >= num_frames or nb in curr_frames:
                    continue
                out_path = CTC_NB_SPLITTED / f"{seq_key}_frame{nb}.yuv"
                if out_path.exists():
                    total_skipped += 1
                    continue

                data = extract_frame(src_yuv, w, h, bit_depth=bit_depth, frame_idx=nb)
                if data is None:
                    continue

                if bit_depth == 8:
                    data = convert_8bit_to_10bit(data, w, h)
                write_frame(data, out_path, w, h)
                total_written += 1

    print(f"  Done: {total_written} written, {total_skipped} skipped")


def encode_one(args: tuple) -> str:
    yuv_path, qp, out_dir, use_trace = args
    stem = Path(yuv_path).stem

    info_path = Path(yuv_path).with_suffix(".yuv.info")
    if not info_path.exists():
        return f"SKIP (no info): {stem} QP{qp}"

    info = {}
    with open(info_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                info[k.strip()] = v.strip()

    width = int(info["width"])
    height = int(info["height"])

    frame_dir = Path(out_dir) / f"QP{qp}" / stem
    rec_path = frame_dir / "rec.yuv"
    bin_path = frame_dir / "bitstream.bin"

    if rec_path.exists() and bin_path.exists():
        return f"SKIP (exists): {stem} QP{qp}"

    frame_dir.mkdir(parents=True, exist_ok=True)
    encoder = ENCODER_TRACE if use_trace else ENCODER_NOTRACE

    cmd = [
        str(encoder),
        "-c", str(CFG_NOFILTER),
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
    ]

    if use_trace:
        csv_path = frame_dir / "bitstream_partition.csv"
        cmd.extend([
            f"--TraceFile={csv_path}",
            "--TraceRule=D_BLOCK_STATISTICS_ALL:poc==0",
        ])

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")[-300:]
            return f"FAIL: {stem} QP{qp}: {err}"
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {stem} QP{qp}"
    except Exception as e:
        return f"ERROR: {stem} QP{qp}: {e}"

    with open(frame_dir / "rec.yuv.info", "w") as f:
        f.write(f"width={width}\nheight={height}\nnum_frames=1\npix_fmt=yuv420p10le\nqp={qp}\n")

    return f"OK: {stem} QP{qp}"


def run_encode_tasks(tasks: list[tuple], workers: int, desc: str):
    if not tasks:
        print(f"  {desc}: nothing to encode")
        return
    print(f"  {desc}: {len(tasks)} tasks, {workers} workers")
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(encode_one, tasks)
    ok = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    fail = [r for r in results if not r.startswith("OK") and not r.startswith("SKIP")]
    print(f"  Results: {ok} encoded, {skip} skipped, {len(fail)} failed")
    for f in fail[:5]:
        print(f"    {f}")


def encode_xiph(workers: int):
    print("\n=== Encoding XIPH ===")
    if not XIPH_SPLITTED.exists():
        print("  ERROR: Run split first")
        return
    tasks_trace = []
    tasks_notrace = []
    for yuv_path in sorted(XIPH_SPLITTED.glob("*.yuv")):
        m = re.match(r"^(.+)_frame(\d+)$", yuv_path.stem)
        if not m:
            continue
        frame_idx = int(m.group(2))
        is_curr = frame_idx in TARGET_FRAMES
        for qp in QP_VALUES:
            if is_curr:
                tasks_trace.append((str(yuv_path), qp, str(XIPH_DECODED), True))
            else:
                tasks_notrace.append((str(yuv_path), qp, str(XIPH_DECODED), False))
    run_encode_tasks(tasks_trace, workers, "XIPH curr (with trace)")
    run_encode_tasks(tasks_notrace, workers, "XIPH neighbors (no trace)")


def encode_bvi_neighbors(workers: int, only_4k: bool = False, no_4k: bool = False):
    label = "4K only" if only_4k else "no-4K" if no_4k else "all"
    print(f"\n=== Encoding BVI neighbors ({label}) ===")
    if not BVI_NB_SPLITTED.exists():
        print("  ERROR: Run split first")
        return
    tasks = []
    for yuv_path in sorted(BVI_NB_SPLITTED.glob("*.yuv")):
        if only_4k and RES_4K not in yuv_path.stem:
            continue
        if no_4k and RES_4K in yuv_path.stem:
            continue
        for qp in QP_VALUES:
            tasks.append((str(yuv_path), qp, str(BVI_NB_DECODED), False))
    run_encode_tasks(tasks, workers, f"BVI neighbors ({label})")


def encode_ctc_neighbors(workers: int):
    print("\n=== Encoding CTC neighbors ===")
    if not CTC_NB_SPLITTED.exists():
        print("  ERROR: Run split first")
        return
    tasks = []
    for yuv_path in sorted(CTC_NB_SPLITTED.glob("*.yuv")):
        for qp in QP_VALUES:
            tasks.append((str(yuv_path), qp, str(CTC_NB_DECODED), False))
    run_encode_tasks(tasks, workers, "CTC neighbors")


def generate_partition_maps(workers: int):
    print("\n=== Generating partition maps (XIPH curr) ===")
    if not PARTITION_MAP_SCRIPT.exists():
        print(f"  ERROR: {PARTITION_MAP_SCRIPT} not found")
        return
    cmd = [sys.executable, str(PARTITION_MAP_SCRIPT),
           "--root", str(XIPH_DECODED), "--workers", str(workers)]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def verify_all():
    print("\n=== VERIFICATION ===\n")

    def check_dir(decoded_dir: Path, label: str, expect_trace_for_curr: bool = False):
        if not decoded_dir.exists():
            print(f"  {label}: not yet processed")
            return
        n_ok = n_bad = 0
        for qp_dir in sorted(decoded_dir.iterdir()):
            if not qp_dir.is_dir():
                continue
            for fd in sorted(qp_dir.iterdir()):
                if not fd.is_dir():
                    continue
                rec = fd / "rec.yuv"
                info = fd / "rec.yuv.info"
                if not rec.exists():
                    n_bad += 1
                    if n_bad <= 3:
                        print(f"    MISSING rec.yuv: {fd}")
                    continue
                if not info.exists():
                    n_bad += 1
                    continue
                meta = {}
                with open(info) as f:
                    for line in f:
                        if "=" in line:
                            k, v = line.strip().split("=", 1)
                            meta[k] = v
                w = int(meta.get("width", 0))
                h = int(meta.get("height", 0))
                if w and h:
                    expected = frame_size_bytes(w, h, 10)
                    if rec.stat().st_size != expected:
                        n_bad += 1
                        if n_bad <= 3:
                            print(f"    SIZE MISMATCH: {fd.name} ({rec.stat().st_size} vs {expected})")
                        continue
                if expect_trace_for_curr:
                    m = re.match(r"^(.+)_frame(\d+)$", fd.name)
                    if m and int(m.group(2)) in TARGET_FRAMES:
                        csv = fd / "bitstream_partition.csv"
                        if not csv.exists():
                            n_bad += 1
                            if n_bad <= 3:
                                print(f"    MISSING trace CSV: {fd.name}")
                            continue
                n_ok += 1
        print(f"  {label}: {n_ok} OK, {n_bad} issues")

    check_dir(XIPH_DECODED, "XIPH", expect_trace_for_curr=True)
    check_dir(BVI_NB_DECODED, "BVI neighbors")
    check_dir(CTC_NB_DECODED, "CTC neighbors")

    print()
    xiph_n = len(list(XIPH_SPLITTED.glob("*.yuv"))) if XIPH_SPLITTED.exists() else 0
    bvi_n = len(list(BVI_NB_SPLITTED.glob("*.yuv"))) if BVI_NB_SPLITTED.exists() else 0
    ctc_n = len(list(CTC_NB_SPLITTED.glob("*.yuv"))) if CTC_NB_SPLITTED.exists() else 0
    print(f"  Splitted frames: XIPH={xiph_n} (expect ~{59*15}), "
          f"BVI_nb={bvi_n} (expect ~{772*10}), CTC_nb={ctc_n} (expect ~{154*2})")


def main():
    parser = argparse.ArgumentParser(description="Prepare temporal data (local)")
    parser.add_argument("--phase", required=True,
                        choices=["split", "encode", "maps", "verify", "all"])
    parser.add_argument("--datasets", nargs="+", default=["xiph", "bvi", "ctc"],
                        choices=["xiph", "bvi", "ctc"])
    parser.add_argument("--workers", type=int,
                        default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--no-4k", action="store_true",
                        help="Skip 4K BVI (Faza 1)")
    parser.add_argument("--only-4k", action="store_true",
                        help="Only 4K BVI (Faza 2)")
    args = parser.parse_args()

    if args.no_4k and args.only_4k:
        print("ERROR: --no-4k and --only-4k are mutually exclusive")
        sys.exit(1)

    if args.phase in ("encode", "all"):
        missing = []
        if not ENCODER_TRACE.exists():
            missing.append(f"  Encoder (trace): {ENCODER_TRACE}")
        if not ENCODER_NOTRACE.exists():
            missing.append(f"  Encoder (notrace): {ENCODER_NOTRACE}")
        if not CFG_NOFILTER.exists():
            missing.append(f"  Config: {CFG_NOFILTER}")
        if missing:
            print("ERROR: Missing files:")
            for m in missing:
                print(m)
            sys.exit(1)

    if args.phase == "verify":
        verify_all()
        return

    if args.phase in ("split", "all"):
        if "xiph" in args.datasets:
            split_xiph()
        if "bvi" in args.datasets:
            split_bvi_neighbors(only_4k=args.only_4k, no_4k=args.no_4k)
        if "ctc" in args.datasets:
            split_ctc_neighbors()

    if args.phase in ("encode", "all"):
        if "xiph" in args.datasets:
            encode_xiph(args.workers)
        if "bvi" in args.datasets:
            encode_bvi_neighbors(args.workers, only_4k=args.only_4k, no_4k=args.no_4k)
        if "ctc" in args.datasets:
            encode_ctc_neighbors(args.workers)

    if args.phase in ("maps", "all"):
        if "xiph" in args.datasets:
            generate_partition_maps(args.workers)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
