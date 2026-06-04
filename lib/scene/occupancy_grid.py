"""World-frame occupancy grid built from training LiDAR returns.

For every training (sensor, frame) we take the r1 range image, project to
world coordinates, remove points falling inside any dynamic-object bounding
box (the same per-frame bbox transform used by the bg/object split in
gs_loader.py), and voxelise the survivors. The resulting set of "occupied"
voxel keys is the support of the static scene — voxels no LiDAR return
ever landed in are treated as free space.

The opacity of a background Gaussian sitting in a free voxel is then a
structural phantom signal independent of the sky mask and free-space rays:
no observation supports geometry there. The training loop penalises those
opacities directly (see train.py).

Voxel keys are packed as 21 bits per axis into int64 (range ±1M voxel,
i.e. ±500km at voxel_size=0.5m) so membership tests reduce to a single
``torch.isin`` against a sorted-unique 1D tensor.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from tqdm import tqdm

from lib.utils.general_utils import build_rotation


_AXIS_BITS = 21
_AXIS_MASK = (1 << _AXIS_BITS) - 1
_AXIS_OFFSET = 1 << (_AXIS_BITS - 1)  # shift signed → unsigned before packing


def _pack_keys(voxel_idx: torch.Tensor) -> torch.Tensor:
    # voxel_idx: (N, 3) int64. Returns (N,) int64.
    shifted = voxel_idx + _AXIS_OFFSET
    # Drop any point whose voxel index escapes the 21-bit range — caller
    # decides what to do with the in-range subset. We mask here so the
    # bit-shift never overflows int64.
    in_range = ((shifted >= 0) & (shifted <= _AXIS_MASK)).all(dim=-1)
    safe = shifted[in_range]
    keys = (safe[:, 0] << (2 * _AXIS_BITS)) | (safe[:, 1] << _AXIS_BITS) | safe[:, 2]
    return keys, in_range


class WorldOccupancyGrid:
    def __init__(self, voxel_size: float):
        self.voxel_size = float(voxel_size)
        self.occupied_keys: Optional[torch.Tensor] = None  # (M,) int64 sorted-unique
        self.n_points_used: int = 0

    def _xyz_to_keys(self, xyz: torch.Tensor):
        voxel_idx = torch.floor(xyz / self.voxel_size).to(torch.int64)
        return _pack_keys(voxel_idx)

    @torch.no_grad()
    def build(self,
              train_lidars: Dict,
              object_gaussians,
              progress_desc: str = "Build occupancy grid") -> dict:
        """Sweep every training (sensor, frame), strip dynamic-bbox interior
        points, and accumulate occupied voxel keys.

        object_gaussians: iterable of GaussianModel with .bounding_box set
        (i.e. gaussians_assets[1:]); used only to read bbox frame transforms.
        """
        bbox_assets = [g for g in object_gaussians if g.bounding_box is not None]

        pairs = [
            (sensor_name, frame_id)
            for sensor_name, lidar in train_lidars.items()
            for frame_id in lidar.train_frames
        ]

        key_chunks = []
        n_points_total = 0
        for sensor_name, frame_id in tqdm(pairs, desc=progress_desc):
            lidar = train_lidars[sensor_name]
            depth = lidar.get_depth(frame_id)
            hit_mask = lidar.get_mask(frame_id)
            pts_world = lidar.range2point(frame_id, depth)[hit_mask].to("cuda")
            if pts_world.numel() == 0:
                continue

            # Mirror gs_loader.py bg/object split: subtract bbox-frame
            # interior points so the occupancy grid records only the
            # static-scene support.
            for gs in bbox_assets:
                bbox = gs.bounding_box
                if frame_id not in bbox.frame:
                    continue
                T = bbox.frame[frame_id][0].to(pts_world.device)
                R = build_rotation(bbox.frame[frame_id][1])[0].to(pts_world.device)
                pts_local = (pts_world - T) @ R.inverse().T
                inside = (pts_local.abs() < bbox.size / 2).all(dim=1)
                if inside.any():
                    pts_world = pts_world[~inside]
                    if pts_world.numel() == 0:
                        break

            if pts_world.numel() == 0:
                continue

            keys, in_range = self._xyz_to_keys(pts_world)
            n_points_total += int(in_range.sum().item())
            if keys.numel() > 0:
                key_chunks.append(torch.unique(keys))

        if key_chunks:
            all_keys = torch.cat(key_chunks)
            self.occupied_keys = torch.unique(all_keys)
        else:
            self.occupied_keys = torch.empty(0, dtype=torch.int64, device="cuda")
        self.n_points_used = n_points_total

        return {
            "n_voxels": int(self.occupied_keys.numel()),
            "n_points": n_points_total,
            "n_frames": len(pairs),
            "voxel_size": self.voxel_size,
        }

    @torch.no_grad()
    def free_mask(self, world_xyz: torch.Tensor) -> torch.Tensor:
        """Per-Gaussian boolean mask: True where the Gaussian sits in a voxel
        no training point ever fell into. Points whose voxel index escapes
        the packed range are conservatively treated as occupied (mask=False)
        so we never penalise something we cannot judge.
        """
        if self.occupied_keys is None or self.occupied_keys.numel() == 0:
            return torch.zeros(world_xyz.shape[0], dtype=torch.bool,
                               device=world_xyz.device)
        keys, in_range = self._xyz_to_keys(world_xyz)
        free_in_range = ~torch.isin(keys, self.occupied_keys)
        free = torch.zeros(world_xyz.shape[0], dtype=torch.bool,
                           device=world_xyz.device)
        free[in_range] = free_in_range
        return free
