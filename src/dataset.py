from __future__ import annotations

import os
import random
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_info_file(path: Path) -> Dict[str, str]:
    """Parse an .info file (one key=value per line)."""
    info: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
    return info


def build_records(
    decoded_root: str,
    original_root: str,
    qp_list: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan the directory tree and build a list of records (rec+ref pairs).

    decoded_root/
      QP{22,27,32,37}/
        {seq_name}_frame{N}/
          rec.yuv, rec.yuv.info

    original_root/
      {seq_name}_frame{N}.yuv
      {seq_name}_frame{N}.yuv.info
    """
    decoded = Path(decoded_root)
    original = Path(original_root)

    if qp_list is None:
        qp_list = [22, 27, 32, 37]

    records: List[Dict[str, Any]] = []

    for qp in qp_list:
        qp_dir = decoded / f"QP{qp}"
        if not qp_dir.is_dir():
            continue

        for frame_dir in sorted(qp_dir.iterdir()):
            if not frame_dir.is_dir():
                continue

            rec_yuv = frame_dir / "rec.yuv"
            rec_info = frame_dir / "rec.yuv.info"
            if not rec_yuv.exists() or not rec_info.exists():
                continue

            frame_name = frame_dir.name
            ref_yuv = original / f"{frame_name}.yuv"
            if not ref_yuv.exists():
                continue

            info = parse_info_file(rec_info)
            width = int(info["width"])
            height = int(info["height"])

            m = re.match(r"^(.+)_frame(\d+)$", frame_name)
            if m:
                seq_id = m.group(1)
                frame_id = int(m.group(2))
            else:
                seq_id = frame_name
                frame_id = 0

            records.append({
                "rec_path": str(rec_yuv),
                "ref_path": str(ref_yuv),
                "width": width,
                "height": height,
                "qp": qp,
                "sequence_id": seq_id,
                "frame_id": frame_id,
            })

    return records


def train_val_split_by_sequence(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split into train/val by sequence — all frames
    and all QPs of a given sequence go to the same subset.
    """
    seq_ids = sorted(set(r["sequence_id"] for r in records))
    rng = random.Random(seed)
    rng.shuffle(seq_ids)

    n_val = max(1, int(len(seq_ids) * val_ratio))
    val_seqs = set(seq_ids[:n_val])

    train_records = [r for r in records if r["sequence_id"] not in val_seqs]
    val_records = [r for r in records if r["sequence_id"] in val_seqs]

    return train_records, val_records


def train_val_test_split_by_sequence(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split into train/val/test by sequence.
    All frames and QPs of a given sequence in the same subset.
    """
    seq_ids = sorted(set(r["sequence_id"] for r in records))
    rng = random.Random(seed)
    rng.shuffle(seq_ids)

    n_test = max(1, int(len(seq_ids) * test_ratio))
    n_val = max(1, int(len(seq_ids) * val_ratio))

    test_seqs = set(seq_ids[:n_test])
    val_seqs = set(seq_ids[n_test: n_test + n_val])

    train_records = [r for r in records if r["sequence_id"] not in test_seqs and r["sequence_id"] not in val_seqs]
    val_records = [r for r in records if r["sequence_id"] in val_seqs]
    test_records = [r for r in records if r["sequence_id"] in test_seqs]

    return train_records, val_records, test_records


class VVCFrameDataset(Dataset):
    """
    Dataset for VVC INTRA post-processing.

    On disk: YUV420 10-bit, one frame per file (planar, uint16 LE).

    In __getitem__:
      - reads ref and rec
      - samples a patch in 420 space
      - upsamples U/V to 444
      - returns:
          input  -> [4, patch, patch]  = YUV444 + QP map
          target -> [3, patch, patch]  = YUV444 reference
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        patch_size: int = 132,
        patches_per_frame: int = 1,
        bit_depth: int = 10,
        qp_max: int = 63,
        training: bool = True,
        augment: bool = True,
    ) -> None:
        super().__init__()

        if patch_size % 2 != 0:
            raise ValueError(f"patch_size musi być parzysty, dostałem {patch_size}")

        self.records = records
        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.bit_depth = bit_depth
        self.max_value = float((1 << bit_depth) - 1)
        self.qp_max = qp_max
        self.training = training
        self.augment = augment and training

        self.valid_records = [
            r for r in records
            if r["width"] >= patch_size and r["height"] >= patch_size
        ]

    def __len__(self) -> int:
        return len(self.valid_records) * self.patches_per_frame

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec_idx = index // self.patches_per_frame
        rec = self.valid_records[rec_idx]

        w, h = rec["width"], rec["height"]
        ps = self.patch_size

        if self.training:
            x, y = self._random_crop_xy(w, h, ps)
        else:
            x, y = self._center_crop_xy(w, h, ps)

        y_ref_p, u_ref_p, v_ref_p = self._read_yuv420_patch(rec["ref_path"], h, w, x, y, ps)
        y_rec_p, u_rec_p, v_rec_p = self._read_yuv420_patch(rec["rec_path"], h, w, x, y, ps)

        ref_444 = self._upsample_to_444(y_ref_p, u_ref_p, v_ref_p)
        rec_444 = self._upsample_to_444(y_rec_p, u_rec_p, v_rec_p)

        if self.augment:
            ref_444, rec_444 = self._augment(ref_444, rec_444)

        ref_t = torch.from_numpy(ref_444.astype(np.float32) / self.max_value)
        rec_t = torch.from_numpy(rec_444.astype(np.float32) / self.max_value)

        qp_val = float(rec["qp"]) / float(self.qp_max)
        qp_map = torch.full((1, ps, ps), fill_value=qp_val, dtype=torch.float32)

        inp = torch.cat([rec_t, qp_map], dim=0)

        return {"input": inp, "target": ref_t}


    @staticmethod
    def _read_yuv420(
        path: str, height: int, width: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_size = width * height
        uv_w, uv_h = width // 2, height // 2
        uv_size = uv_w * uv_h
        expected = y_size + 2 * uv_size

        data = np.fromfile(path, dtype=np.uint16)
        if data.size != expected:
            raise ValueError(
                f"{path}: oczekiwano {expected} uint16, jest {data.size}"
            )

        y = data[:y_size].reshape(height, width)
        u = data[y_size: y_size + uv_size].reshape(uv_h, uv_w)
        v = data[y_size + uv_size:].reshape(uv_h, uv_w)
        return y, u, v

    @staticmethod
    def _read_yuv420_patch(
        path: str, height: int, width: int,
        cx: int, cy: int, ps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read from disk ONLY the rows needed to extract the patch.

        For a 4K frame (3840x2160) and ps=132:
          full frame = ~18MB,
          patch rows = ~1.3MB (1 seek + sequential read per plane).
        """
        bps = 2
        y_row_bytes = width * bps
        uv_w = width // 2
        uv_row_bytes = uv_w * bps
        y_plane_bytes = y_row_bytes * height
        uv_plane_bytes = uv_row_bytes * (height // 2)

        xu, yu, psu = cx // 2, cy // 2, ps // 2

        with open(path, "rb") as f:
            f.seek(cy * y_row_bytes)
            y_block = np.frombuffer(f.read(ps * y_row_bytes), dtype=np.uint16)
            y_patch = y_block.reshape(ps, width)[:, cx : cx + ps].copy()

            f.seek(y_plane_bytes + yu * uv_row_bytes)
            u_block = np.frombuffer(f.read(psu * uv_row_bytes), dtype=np.uint16)
            u_patch = u_block.reshape(psu, uv_w)[:, xu : xu + psu].copy()

            f.seek(y_plane_bytes + uv_plane_bytes + yu * uv_row_bytes)
            v_block = np.frombuffer(f.read(psu * uv_row_bytes), dtype=np.uint16)
            v_patch = v_block.reshape(psu, uv_w)[:, xu : xu + psu].copy()

        return y_patch, u_patch, v_patch


    @staticmethod
    def _random_crop_xy(w: int, h: int, ps: int) -> Tuple[int, int]:
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        x -= x % 2
        y -= y % 2
        return x, y

    @staticmethod
    def _center_crop_xy(w: int, h: int, ps: int) -> Tuple[int, int]:
        x = (w - ps) // 2
        y = (h - ps) // 2
        x -= x % 2
        y -= y % 2
        return x, y

    @staticmethod
    def _crop_420(
        yp: np.ndarray, up: np.ndarray, vp: np.ndarray,
        x: int, y: int, ps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xu, yu, psu = x // 2, y // 2, ps // 2
        return (
            yp[y: y + ps, x: x + ps],
            up[yu: yu + psu, xu: xu + psu],
            vp[yu: yu + psu, xu: xu + psu],
        )


    @staticmethod
    def _upsample_to_444(
        yp: np.ndarray, up: np.ndarray, vp: np.ndarray
    ) -> np.ndarray:
        h, w = yp.shape
        u_up = cv2.resize(up.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        v_up = cv2.resize(vp.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        np.clip(u_up, 0, 1023, out=u_up)
        np.clip(v_up, 0, 1023, out=v_up)
        yp = yp.astype(np.float32)
        return np.ascontiguousarray(np.stack([yp, u_up, v_up], axis=0))


    @staticmethod
    def _augment(
        ref: np.ndarray, rec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            ref = ref[:, :, ::-1]
            rec = rec[:, :, ::-1]
        if random.random() < 0.5:
            ref = ref[:, ::-1, :]
            rec = rec[:, ::-1, :]
        k = random.randint(0, 3)
        if k > 0:
            ref = np.rot90(ref, k=k, axes=(1, 2))
            rec = np.rot90(rec, k=k, axes=(1, 2))
        return np.ascontiguousarray(ref), np.ascontiguousarray(rec)


class VVCCachedDataset(Dataset):
    """
    Same interface as VVCFrameDataset but reads ALL frames into RAM
    once during __init__. Stores raw YUV420 uint16 — identical data
    pipeline as VVCFrameDataset (same cv2.resize on patches).

    Memory usage: ~198GB for full BVI+DIV2K dataset at all QPs.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        patch_size: int = 132,
        patches_per_frame: int = 1,
        bit_depth: int = 10,
        qp_max: int = 63,
        training: bool = True,
        augment: bool = True,
    ) -> None:
        super().__init__()

        if patch_size % 2 != 0:
            raise ValueError(f"patch_size musi być parzysty, dostałem {patch_size}")

        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.bit_depth = bit_depth
        self.max_value = float((1 << bit_depth) - 1)
        self.qp_max = qp_max
        self.training = training
        self.augment = augment and training

        self.valid_records = [
            r for r in records
            if r["width"] >= patch_size and r["height"] >= patch_size
        ]

        print(f"  Caching {len(self.valid_records)} records into RAM (YUV420 uint16) ...")
        self._frame_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        all_paths = set()
        for r in self.valid_records:
            all_paths.add((r["rec_path"], r["height"], r["width"]))
            all_paths.add((r["ref_path"], r["height"], r["width"]))

        for i, (path, h, w) in enumerate(sorted(all_paths)):
            if path not in self._frame_cache:
                self._frame_cache[path] = VVCFrameDataset._read_yuv420(path, h, w)
            if (i + 1) % 500 == 0:
                print(f"    cached {i + 1}/{len(all_paths)} files")
        print(f"  Cached {len(self._frame_cache)} unique files into RAM.")

    def __len__(self) -> int:
        return len(self.valid_records) * self.patches_per_frame

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec_idx = index // self.patches_per_frame
        rec = self.valid_records[rec_idx]

        w, h = rec["width"], rec["height"]
        ps = self.patch_size

        if self.training:
            cx, cy = VVCFrameDataset._random_crop_xy(w, h, ps)
        else:
            cx, cy = VVCFrameDataset._center_crop_xy(w, h, ps)

        y_ref, u_ref, v_ref = self._frame_cache[rec["ref_path"]]
        y_rec, u_rec, v_rec = self._frame_cache[rec["rec_path"]]

        y_ref_p, u_ref_p, v_ref_p = VVCFrameDataset._crop_420(
            y_ref, u_ref, v_ref, cx, cy, ps)
        y_rec_p, u_rec_p, v_rec_p = VVCFrameDataset._crop_420(
            y_rec, u_rec, v_rec, cx, cy, ps)

        ref_444 = VVCFrameDataset._upsample_to_444(y_ref_p, u_ref_p, v_ref_p)
        rec_444 = VVCFrameDataset._upsample_to_444(y_rec_p, u_rec_p, v_rec_p)

        if self.augment:
            ref_444, rec_444 = VVCFrameDataset._augment(ref_444, rec_444)

        ref_t = torch.from_numpy(ref_444.astype(np.float32) / self.max_value)
        rec_t = torch.from_numpy(rec_444.astype(np.float32) / self.max_value)

        qp_val = float(rec["qp"]) / float(self.qp_max)
        qp_map = torch.full((1, ps, ps), fill_value=qp_val, dtype=torch.float32)

        inp = torch.cat([rec_t, qp_map], dim=0)
        return {"input": inp, "target": ref_t}


class VVCMmapDataset(Dataset):
    """
    Reads from a pre-processed dataset (created by preprocess_memmap.py).

    All frames are pre-upsampled to YUV444, normalized to [0,1], stored as
    float16 packed sequentially. Data must be pre-loaded into RAM (passed
    as data_array) to avoid Lustre random-read latency.

    Optionally loads partition maps (luma + chroma depth/boundary) from a
    separate memmap (created by preprocess_partition_memmap.py). When partition
    data is provided, input has 8 channels [Y, U, V, QP, l_depth, l_boundary,
    c_depth, c_boundary]; otherwise the original 4 channels [Y, U, V, QP] are
    returned — fully backward compatible.

    __getitem__: numpy slice → crop → augment → tensor.
    No file opens, no cv2.resize, no I/O.
    """

    def __init__(
        self,
        memmap_dir: str,
        records: List[Dict[str, Any]],
        patch_size: int = 256,
        patches_per_frame: int = 1,
        qp_max: int = 63,
        training: bool = True,
        augment: bool = True,
        data_array: Optional[np.ndarray] = None,
        partition_memmap_dir: Optional[str] = None,
        partition_data_array: Optional[np.ndarray] = None,
        partition_channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        import json

        if patch_size % 2 != 0:
            raise ValueError(f"patch_size musi być parzysty, dostałem {patch_size}")

        idx_path = os.path.join(memmap_dir, "index.json")

        with open(idx_path) as f:
            index = json.load(f)

        self.file_index = index["file_index"]

        if data_array is not None:
            self.data = data_array
        else:
            dat_path = os.path.join(memmap_dir, "frames.dat")
            total_elems = index["total_elems"]
            self.data = np.memmap(dat_path, dtype=np.float16, mode="r", shape=(total_elems,))

        self.partition_index = None
        self.partition_data = None
        if partition_memmap_dir is not None:
            pidx_path = os.path.join(partition_memmap_dir, "partition_index.json")
            with open(pidx_path) as f:
                pindex = json.load(f)
            self.partition_index = pindex["file_index"]
            if partition_data_array is not None:
                self.partition_data = partition_data_array
            else:
                pdat_path = os.path.join(partition_memmap_dir, "partition_frames.dat")
                ptotal = pindex["total_elems"]
                self.partition_data = np.memmap(pdat_path, dtype=np.float16, mode="r", shape=(ptotal,))

        filter_conds = [
            r for r in records
            if r["width"] >= patch_size
            and r["height"] >= patch_size
            and r["rec_path"] in self.file_index
            and r["ref_path"] in self.file_index
        ]
        if self.partition_index is not None:
            filter_conds = [
                r for r in filter_conds
                if r["rec_path"] in self.partition_index
            ]

        self.valid_records = filter_conds
        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.qp_max = qp_max
        self.training = training
        self.augment = augment and training
        self.partition_channels = partition_channels

    def _read_patch(self, path: str, cx: int, cy: int, ps: int) -> np.ndarray:
        """Read a [3, ps, ps] float32 patch from in-RAM array."""
        fi = self.file_index[path]
        offset = fi["offset"]
        h, w = fi["height"], fi["width"]

        patches = np.empty((3, ps, ps), dtype=np.float32)
        for c in range(3):
            ch_offset = offset + c * h * w + cy * w
            block = self.data[ch_offset : ch_offset + ps * w]
            patches[c] = block.reshape(ps, w)[:, cx:cx+ps].astype(np.float32)
        return patches

    def _read_partition_patch(self, rec_path: str, cx: int, cy: int, ps: int) -> np.ndarray:
        """Read partition patch, filtered by self.partition_channels."""
        fi = self.partition_index[rec_path]
        offset = fi["offset"]
        h, w = fi["height"], fi["width"]

        n_ch_total = 4
        channels = self.partition_channels if self.partition_channels is not None else list(range(n_ch_total))
        patches = np.empty((len(channels), ps, ps), dtype=np.float32)
        for i, c in enumerate(channels):
            ch_offset = offset + c * h * w + cy * w
            block = self.partition_data[ch_offset : ch_offset + ps * w]
            patches[i] = block.reshape(ps, w)[:, cx:cx+ps].astype(np.float32)
        return patches

    def __len__(self) -> int:
        return len(self.valid_records) * self.patches_per_frame

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec_idx = index // self.patches_per_frame
        rec = self.valid_records[rec_idx]

        w, h = rec["width"], rec["height"]
        ps = self.patch_size

        if self.training:
            cx, cy = VVCFrameDataset._random_crop_xy(w, h, ps)
        else:
            cx, cy = VVCFrameDataset._center_crop_xy(w, h, ps)

        rec_patch = self._read_patch(rec["rec_path"], cx, cy, ps)
        ref_patch = self._read_patch(rec["ref_path"], cx, cy, ps)

        part_patch = None
        if self.partition_data is not None:
            part_patch = self._read_partition_patch(rec["rec_path"], cx, cy, ps)

        if self.augment:
            ref_patch, rec_patch, part_patch = self._augment_with_partition(
                ref_patch, rec_patch, part_patch
            )

        ref_t = torch.from_numpy(np.ascontiguousarray(ref_patch))
        rec_t = torch.from_numpy(np.ascontiguousarray(rec_patch))

        qp_val = float(rec["qp"]) / float(self.qp_max)
        qp_map = torch.full((1, ps, ps), fill_value=qp_val, dtype=torch.float32)

        inp = torch.cat([rec_t, qp_map], dim=0)

        if part_patch is not None:
            part_t = torch.from_numpy(np.ascontiguousarray(part_patch))
            inp = torch.cat([inp, part_t], dim=0)

        return {"input": inp, "target": ref_t}

    @staticmethod
    def _augment_with_partition(
        ref: np.ndarray, rec: np.ndarray, part: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Same augmentation as VVCFrameDataset._augment but also transforms partition maps."""
        if random.random() < 0.5:
            ref = ref[:, :, ::-1]
            rec = rec[:, :, ::-1]
            if part is not None:
                part = part[:, :, ::-1]
        if random.random() < 0.5:
            ref = ref[:, ::-1, :]
            rec = rec[:, ::-1, :]
            if part is not None:
                part = part[:, ::-1, :]
        k = random.randint(0, 3)
        if k > 0:
            ref = np.rot90(ref, k=k, axes=(1, 2))
            rec = np.rot90(rec, k=k, axes=(1, 2))
            if part is not None:
                part = np.rot90(part, k=k, axes=(1, 2))
        return (
            np.ascontiguousarray(ref),
            np.ascontiguousarray(rec),
            np.ascontiguousarray(part) if part is not None else None,
        )
