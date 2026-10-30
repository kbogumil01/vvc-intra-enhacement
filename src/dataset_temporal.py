"""
Temporal Memmap Dataset for 12-channel VVC post-processing model.

Input: [YUV_curr(3), QP(1), boundary_luma(1), boundary_chroma(1), YUV_prev(3), YUV_next(3)] = 12ch
Target: [YUV_ref(3)]

Uses the temporal memmap built by preprocess_temporal_memmap.py — records contain
prev_path and next_path pointing to neighbor frames in the same memmap.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import VVCFrameDataset


class VVCTemporalMmapDataset(Dataset):
    """
    12-channel temporal dataset from preprocessed memmap.

    All frames (curr rec, ref, prev, next) and partition maps are in the same
    memmap file (created by preprocess_temporal_memmap.py).

    Channels:
      [0:3]  YUV curr rec (from frames memmap)
      [3]    QP map (scalar, normalized)
      [4]    boundary_luma (from partition memmap)
      [5]    boundary_chroma (from partition memmap)
      [6:9]  YUV prev rec (from frames memmap)
      [9:12] YUV next rec (from frames memmap)
    """

    def __init__(
        self,
        memmap_dir: str,
        records: List[Dict[str, Any]],
        patch_size: int = 240,
        patches_per_frame: int = 1,
        qp_max: int = 63,
        training: bool = True,
        augment: bool = True,
        data_array: Optional[np.ndarray] = None,
        partition_data_array: Optional[np.ndarray] = None,
        temporal_mode: str = "both",
    ) -> None:
        super().__init__()

        if patch_size % 2 != 0:
            raise ValueError(f"patch_size must be even, got {patch_size}")

        if temporal_mode not in ("both", "prev_only", "next_only"):
            raise ValueError(f"temporal_mode must be both/prev_only/next_only, got {temporal_mode}")
        self.temporal_mode = temporal_mode

        idx_path = os.path.join(memmap_dir, "index.json")
        with open(idx_path) as f:
            index = json.load(f)
        self.file_index = index["file_index"]

        pidx_path = os.path.join(memmap_dir, "partition_index.json")
        with open(pidx_path) as f:
            pindex = json.load(f)
        self.partition_index = pindex["file_index"]

        if data_array is not None:
            self.data = data_array
        else:
            dat_path = os.path.join(memmap_dir, "frames.dat")
            total_elems = index["total_elems"]
            self.data = np.memmap(dat_path, dtype=np.float16, mode="r", shape=(total_elems,))

        if partition_data_array is not None:
            self.partition_data = partition_data_array
        else:
            pdat_path = os.path.join(memmap_dir, "partition_frames.dat")
            ptotal = pindex["total_elems"]
            self.partition_data = np.memmap(pdat_path, dtype=np.float16, mode="r", shape=(ptotal,))

        self.valid_records = [
            r for r in records
            if (r["width"] >= patch_size
                and r["height"] >= patch_size
                and r["rec_path"] in self.file_index
                and r["ref_path"] in self.file_index
                and r["prev_path"] in self.file_index
                and r["next_path"] in self.file_index
                and r["rec_path"] in self.partition_index)
        ]

        self.patch_size = patch_size
        self.patches_per_frame = patches_per_frame
        self.qp_max = qp_max
        self.training = training
        self.augment = augment and training

    def __len__(self) -> int:
        return len(self.valid_records) * self.patches_per_frame

    def _read_patch(self, path: str, cx: int, cy: int, ps: int) -> np.ndarray:
        """Read [3, ps, ps] patch from frames memmap."""
        fi = self.file_index[path]
        offset = fi["offset"]
        h, w = fi["height"], fi["width"]

        patches = np.empty((3, ps, ps), dtype=np.float32)
        for c in range(3):
            ch_offset = offset + c * h * w + cy * w
            block = self.data[ch_offset:ch_offset + ps * w]
            patches[c] = block.reshape(ps, w)[:, cx:cx + ps].astype(np.float32)
        return patches

    def _read_partition_patch(self, rec_path: str, cx: int, cy: int, ps: int) -> np.ndarray:
        """Read [2, ps, ps] partition patch (boundary_luma, boundary_chroma)."""
        fi = self.partition_index[rec_path]
        offset = fi["offset"]
        h, w = fi["height"], fi["width"]

        patches = np.empty((2, ps, ps), dtype=np.float32)
        for c in range(2):
            ch_offset = offset + c * h * w + cy * w
            block = self.partition_data[ch_offset:ch_offset + ps * w]
            patches[c] = block.reshape(ps, w)[:, cx:cx + ps].astype(np.float32)
        return patches

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
        prev_patch = self._read_patch(rec["prev_path"], cx, cy, ps)
        next_patch = self._read_patch(rec["next_path"], cx, cy, ps)
        part_patch = self._read_partition_patch(rec["rec_path"], cx, cy, ps)

        if self.augment:
            ref_patch, rec_patch, prev_patch, next_patch, part_patch = self._augment_temporal(
                ref_patch, rec_patch, prev_patch, next_patch, part_patch
            )

        rec_t = torch.from_numpy(np.ascontiguousarray(rec_patch))
        ref_t = torch.from_numpy(np.ascontiguousarray(ref_patch))
        prev_t = torch.from_numpy(np.ascontiguousarray(prev_patch))
        next_t = torch.from_numpy(np.ascontiguousarray(next_patch))
        part_t = torch.from_numpy(np.ascontiguousarray(part_patch))

        qp_val = float(rec["qp"]) / float(self.qp_max)
        qp_map = torch.full((1, ps, ps), fill_value=qp_val, dtype=torch.float32)

        if self.temporal_mode == "both":
            inp = torch.cat([rec_t, qp_map, part_t, prev_t, next_t], dim=0)
        elif self.temporal_mode == "prev_only":
            inp = torch.cat([rec_t, qp_map, part_t, prev_t], dim=0)
        else:
            inp = torch.cat([rec_t, qp_map, part_t, next_t], dim=0)

        return {"input": inp, "target": ref_t}

    @staticmethod
    def _augment_temporal(
        ref: np.ndarray, rec: np.ndarray,
        prev: np.ndarray, nxt: np.ndarray,
        part: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply same random augmentation to all patches."""
        if random.random() < 0.5:
            ref = ref[:, :, ::-1]
            rec = rec[:, :, ::-1]
            prev = prev[:, :, ::-1]
            nxt = nxt[:, :, ::-1]
            part = part[:, :, ::-1]
        if random.random() < 0.5:
            ref = ref[:, ::-1, :]
            rec = rec[:, ::-1, :]
            prev = prev[:, ::-1, :]
            nxt = nxt[:, ::-1, :]
            part = part[:, ::-1, :]
        k = random.randint(0, 3)
        if k > 0:
            ref = np.rot90(ref, k=k, axes=(1, 2))
            rec = np.rot90(rec, k=k, axes=(1, 2))
            prev = np.rot90(prev, k=k, axes=(1, 2))
            nxt = np.rot90(nxt, k=k, axes=(1, 2))
            part = np.rot90(part, k=k, axes=(1, 2))

        if random.random() < 0.5:
            prev, nxt = nxt, prev

        return (
            np.ascontiguousarray(ref),
            np.ascontiguousarray(rec),
            np.ascontiguousarray(prev),
            np.ascontiguousarray(nxt),
            np.ascontiguousarray(part),
        )
