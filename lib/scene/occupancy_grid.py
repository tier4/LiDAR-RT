"""World-frame occupancy grid built from training LiDAR returns.

SLAM-style three-state classification per voxel:
  - OCCUPIED: at least one BG (non-dynamic) LiDAR return landed in the voxel.
  - FREE:     at least one training ray traversed the voxel without
              terminating inside it (= LiDAR actively observed empty space).
  - UNKNOWN:  no ray ever entered this voxel (out of LiDAR FOV, behind an
              occluder, beyond the trace range, etc.).

Distinguishing free from unknown matters for the anti-phantom losses /
hard-prune: an unknown-region high-opacity Gaussian shouldn't necessarily
be penalised (there's no observation evidence either way), while a
free-region one is a clear phantom.

Ray traversal uses uniform sampling along each ray at voxel_size
increments (cheap, GPU-vectorised; equivalent to 3D-DDA for our voxel
sizes). For each ray:

  * Hit on BG (gt_mask=True and hit point not inside any dynamic bbox):
    free range = [0, gt_depth - voxel_size/2]; hit voxel → occupied.
  * Hit on dynamic object: skip the ray entirely (we don't claim any
    voxel free or occupied — the object's own motion makes the visible
    space ambiguous).
  * No return (sky/dropout): free range = [0, max_depth].

Voxel keys are packed as 21 bits per axis into int64 (range ±1M voxel,
i.e. ±500km at voxel_size=0.5m) so membership tests reduce to a single
``torch.isin`` against a sorted-unique 1D tensor.

Build is expensive (1-3 min for T4-scale), so the result is hashed
against its inputs and cached as a .pt file. Subsequent runs with the
same data + voxel_size load in <1 s.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from lib.utils.general_utils import build_rotation


_AXIS_BITS = 21
_AXIS_MASK = (1 << _AXIS_BITS) - 1
_AXIS_OFFSET = 1 << (_AXIS_BITS - 1)  # shift signed → unsigned before packing


def _pack_keys(voxel_idx: torch.Tensor):
    """voxel_idx: (N, 3) int64. Returns ((K,) int64 keys, (N,) bool in_range mask)."""
    shifted = voxel_idx + _AXIS_OFFSET
    in_range = ((shifted >= 0) & (shifted <= _AXIS_MASK)).all(dim=-1)
    safe = shifted[in_range]
    keys = (safe[:, 0] << (2 * _AXIS_BITS)) | (safe[:, 1] << _AXIS_BITS) | safe[:, 2]
    return keys, in_range


@torch.no_grad()
def _trace_ray_voxels(rays_o: torch.Tensor,
                      rays_d: torch.Tensor,
                      t_max: torch.Tensor,
                      voxel_size: float) -> torch.Tensor:
    """For each ray, sample points along [0, t_max] at voxel_size spacing,
    voxelise, and return the unique packed keys touched.

    rays_o, rays_d: (B, 3). t_max: (B,) per-ray maximum trace distance.
    Rays with t_max <= 0 are skipped (no contribution).

    Memory bound: at each step we allocate one (B_active, 3) tensor, so
    the peak is one frame's worth of rays — manageable on GPU.
    """
    if rays_o.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=rays_o.device)
    max_t = float(t_max.max().item())
    if max_t <= 0:
        return torch.empty(0, dtype=torch.int64, device=rays_o.device)
    n_steps = int(max_t / voxel_size) + 1
    key_buckets: List[torch.Tensor] = []
    for s in range(n_steps):
        t = s * voxel_size
        active = t_max > t
        if not active.any():
            break
        pts = rays_o[active] + t * rays_d[active]
        voxel_idx = torch.floor(pts / voxel_size).to(torch.int64)
        keys, _ = _pack_keys(voxel_idx)
        if keys.numel() > 0:
            key_buckets.append(keys)
    if not key_buckets:
        return torch.empty(0, dtype=torch.int64, device=rays_o.device)
    return torch.unique(torch.cat(key_buckets))


class WorldOccupancyGrid:
    def __init__(self, voxel_size: float):
        self.voxel_size = float(voxel_size)
        self.occupied_keys: Optional[torch.Tensor] = None  # (M,) int64 sorted-unique
        self.free_keys: Optional[torch.Tensor] = None      # (M,) int64 sorted-unique
        self.n_points_used: int = 0
        self.n_rays_used: int = 0

    # ---------- key helpers ----------

    def _xyz_to_keys(self, xyz: torch.Tensor):
        voxel_idx = torch.floor(xyz / self.voxel_size).to(torch.int64)
        return _pack_keys(voxel_idx)

    # ---------- caching ----------

    def _cache_key(self,
                   train_lidars: Dict,
                   object_gaussians,
                   max_depth: float,
                   extra: Optional[dict] = None) -> str:
        """Deterministic SHA1 prefix from build inputs. Different voxel_size,
        max_depth, frame set, dynamic-bbox configuration → different key.
        Any change in any of these auto-invalidates the cache.
        """
        spec = {
            "version": 2,  # 1 was occupied-only; 2 has free_keys
            "voxel_size": self.voxel_size,
            "max_depth": max_depth,
            "sensors": sorted(
                (sname, list(map(int, sorted(lidar.train_frames))))
                for sname, lidar in train_lidars.items()
            ),
            "bboxes": [],
        }
        for gs in object_gaussians:
            if getattr(gs, "bounding_box", None) is None:
                continue
            bbox = gs.bounding_box
            spec["bboxes"].append({
                "id": getattr(bbox, "object_id", None),
                "size": bbox.size.tolist(),
                "frames": [
                    [int(f), bbox.frame[f][0].tolist(),
                     bbox.frame[f][1].squeeze(0).tolist()]
                    for f in sorted(bbox.frame.keys())
                ],
            })
        if extra:
            spec["extra"] = extra
        blob = json.dumps(spec, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def _save_cache(self, path: str, max_depth: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "version": 2,
                "voxel_size": self.voxel_size,
                "max_depth": max_depth,
                "occupied_keys": self.occupied_keys.cpu(),
                "free_keys": self.free_keys.cpu(),
                "n_points_used": self.n_points_used,
                "n_rays_used": self.n_rays_used,
            },
            path,
        )

    def _load_cache(self, path: str, device: str = "cuda") -> bool:
        try:
            d = torch.load(path, map_location="cpu")
        except Exception:
            return False
        if d.get("version") != 2:
            return False
        self.occupied_keys = d["occupied_keys"].to(device)
        self.free_keys = d["free_keys"].to(device)
        self.n_points_used = d.get("n_points_used", 0)
        self.n_rays_used = d.get("n_rays_used", 0)
        return True

    # ---------- build ----------

    @torch.no_grad()
    def build_or_load_cache(self,
                            train_lidars: Dict,
                            object_gaussians,
                            max_depth: float = 100.0,
                            cache_dir: Optional[str] = None,
                            cache_extra: Optional[dict] = None,
                            progress_desc: str = "Build occupancy grid") -> dict:
        """Try to load from cache first; fall back to full build + save.
        cache_dir=None disables caching.
        """
        cached = False
        if cache_dir is not None:
            key = self._cache_key(train_lidars, object_gaussians, max_depth, cache_extra)
            path = os.path.join(cache_dir, f"{key}.pt")
            if os.path.exists(path) and self._load_cache(path):
                cached = True
                stats = {
                    "cached": True,
                    "cache_path": path,
                    "n_occupied": int(self.occupied_keys.numel()),
                    "n_free": int(self.free_keys.numel()),
                    "n_points_used": self.n_points_used,
                    "n_rays_used": self.n_rays_used,
                    "voxel_size": self.voxel_size,
                }
                return stats

        stats = self.build(train_lidars, object_gaussians,
                           max_depth=max_depth, progress_desc=progress_desc)

        if cache_dir is not None:
            key = self._cache_key(train_lidars, object_gaussians, max_depth, cache_extra)
            path = os.path.join(cache_dir, f"{key}.pt")
            self._save_cache(path, max_depth)
            stats["cache_path"] = path
            stats["cached"] = False
        return stats

    @torch.no_grad()
    def build(self,
              train_lidars: Dict,
              object_gaussians,
              max_depth: float = 100.0,
              progress_desc: str = "Build occupancy grid") -> dict:
        """Sweep every training (sensor, frame), classify rays, accumulate
        per-frame free / occupied key sets, then unique-merge across frames.

        object_gaussians: iterable of GaussianModel with .bounding_box set
        (i.e. gaussians_assets[1:]); used only to read bbox frame transforms
        for the dynamic-hit filter.
        """
        bbox_assets = [g for g in object_gaussians
                       if getattr(g, "bounding_box", None) is not None]

        pairs = [
            (sensor_name, frame_id)
            for sensor_name, lidar in train_lidars.items()
            for frame_id in lidar.train_frames
        ]

        occupied_chunks: List[torch.Tensor] = []
        free_chunks: List[torch.Tensor] = []
        n_points_total = 0
        n_rays_total = 0
        n_free_unique = 0
        n_occ_unique = 0

        pbar = tqdm(pairs, desc=progress_desc)
        for sensor_name, frame_id in pbar:
            lidar = train_lidars[sensor_name]
            rays_o, rays_d = lidar.get_range_rays(frame_id)        # (H, W, 3) each
            depth = lidar.get_depth(frame_id)                       # (H, W)
            hit_mask = lidar.get_mask(frame_id)                     # (H, W) bool

            H, W = depth.shape[0], depth.shape[1]
            rays_o = rays_o.to("cuda").reshape(-1, 3)               # (HW, 3)
            rays_d = rays_d.to("cuda").reshape(-1, 3)               # (HW, 3)
            depth = depth.to("cuda").reshape(-1)                    # (HW,)
            hit_mask = hit_mask.to("cuda").reshape(-1)              # (HW,)

            # World-frame hit points for the hit pixels (for dynamic-bbox check
            # and for occupied-voxel registration).
            hit_pts_world = rays_o + depth.unsqueeze(-1) * rays_d   # (HW, 3)

            # Dynamic-hit detection: a ray's hit lies inside at least one
            # dynamic bbox at this frame. Such rays are excluded entirely —
            # we don't trust either their traversed-free or hit-occupied
            # claim because the object isn't part of the static scene.
            dyn_hit = torch.zeros_like(hit_mask)
            if bbox_assets:
                for gs in bbox_assets:
                    bbox = gs.bounding_box
                    if frame_id not in bbox.frame:
                        continue
                    T = bbox.frame[frame_id][0].to("cuda")
                    R = build_rotation(bbox.frame[frame_id][1])[0].to("cuda")
                    pts_local = (hit_pts_world - T) @ R.inverse().T
                    inside = (pts_local.abs() < bbox.size / 2).all(dim=-1)
                    dyn_hit = dyn_hit | (hit_mask & inside)

            # Per-ray trace length:
            #   * BG hit:    [0, depth - voxel_size/2]
            #   * Sky/drop:  [0, max_depth]
            #   * Dyn hit:   0 (skip)
            t_max = torch.zeros_like(depth)
            bg_hit = hit_mask & ~dyn_hit
            t_max[bg_hit] = (depth[bg_hit] - self.voxel_size / 2).clamp_min(0.0)
            sky_mask = ~hit_mask
            t_max[sky_mask] = max_depth
            # dyn_hit rays keep t_max=0 → no samples generated for them

            n_rays_total += int((t_max > 0).sum().item())

            # Trace traversed voxels (free). Move to CPU immediately so
            # GPU memory doesn't accumulate across frames — at voxel_size
            # 0.1m a single frame already yields ~5e7 unique keys, and
            # holding all 46 on GPU exceeds 1 GB easily and competes with
            # concurrent training jobs.
            free_keys = _trace_ray_voxels(rays_o, rays_d, t_max, self.voxel_size)
            if free_keys.numel() > 0:
                free_chunks.append(free_keys.cpu())
                n_free_unique += free_keys.numel()
            del free_keys

            # Register occupied voxels (BG hits only). Same CPU offload.
            if bg_hit.any():
                occ_pts = hit_pts_world[bg_hit]
                occ_idx = torch.floor(occ_pts / self.voxel_size).to(torch.int64)
                occ_keys, occ_in_range = _pack_keys(occ_idx)
                n_points_total += int(occ_in_range.sum().item())
                if occ_keys.numel() > 0:
                    unique_keys = torch.unique(occ_keys).cpu()
                    occupied_chunks.append(unique_keys)
                    n_occ_unique += unique_keys.numel()

            # Periodic compression: every 5 frames, run a global unique on
            # the accumulated CPU chunks to drop duplicates that span
            # multiple frames (rays from the same area on consecutive
            # frames share many voxels). Keeps the CPU memory footprint
            # at O(scene volume / voxel_size^3) rather than O(frames).
            if len(free_chunks) >= 5:
                free_chunks = [torch.unique(torch.cat(free_chunks))]
            if len(occupied_chunks) >= 10:
                occupied_chunks = [torch.unique(torch.cat(occupied_chunks))]

            # Free transient GPU buffers before next frame.
            del rays_o, rays_d, depth, hit_mask, hit_pts_world, t_max, bg_hit, dyn_hit
            torch.cuda.empty_cache()

            pbar.set_postfix({
                "occ~": n_occ_unique,
                "free~": n_free_unique,
                "rays": n_rays_total,
            })

        # Merge across frames (chunks are already on CPU). Final tensor lives
        # on GPU for fast torch.isin during free_mask() queries.
        if occupied_chunks:
            occupied_cpu = torch.unique(torch.cat(occupied_chunks))
            self.occupied_keys = occupied_cpu.to("cuda")
            del occupied_cpu
        else:
            self.occupied_keys = torch.empty(0, dtype=torch.int64, device="cuda")
        if free_chunks:
            free_cpu = torch.unique(torch.cat(free_chunks))
            self.free_keys = free_cpu.to("cuda")
            del free_cpu
        else:
            self.free_keys = torch.empty(0, dtype=torch.int64, device="cuda")
        self.n_points_used = n_points_total
        self.n_rays_used = n_rays_total

        return {
            "cached": False,
            "n_occupied": int(self.occupied_keys.numel()),
            "n_free": int(self.free_keys.numel()),
            "n_points_used": n_points_total,
            "n_rays_used": n_rays_total,
            "n_frames": len(pairs),
            "voxel_size": self.voxel_size,
            "max_depth": max_depth,
        }

    # ---------- query ----------

    @torch.no_grad()
    def free_mask(self, world_xyz: torch.Tensor) -> torch.Tensor:
        """Per-Gaussian boolean: True iff the voxel containing the Gaussian's
        centre was observed as free (some training ray traversed it) AND
        is NOT registered as occupied (no BG hit landed in it). Points
        whose voxel falls in the 'unknown' set (no ray ever entered) return
        False — we don't punish or prune them with no evidence either way.
        """
        if self.free_keys is None or self.free_keys.numel() == 0:
            return torch.zeros(world_xyz.shape[0], dtype=torch.bool,
                               device=world_xyz.device)
        keys, in_range = self._xyz_to_keys(world_xyz)
        in_free = torch.isin(keys, self.free_keys)
        if self.occupied_keys is not None and self.occupied_keys.numel() > 0:
            in_occ = torch.isin(keys, self.occupied_keys)
        else:
            in_occ = torch.zeros_like(in_free)
        free_in_range = in_free & ~in_occ
        result = torch.zeros(world_xyz.shape[0], dtype=torch.bool,
                             device=world_xyz.device)
        result[in_range] = free_in_range
        return result

    @torch.no_grad()
    def voxel_centers(self, keys: torch.Tensor) -> torch.Tensor:
        """Inverse of _pack_keys + xyz_to_keys: given (N,) packed int64 keys,
        return the (N, 3) world-frame centres of those voxels. Used by the
        rerun visualiser to plot the occupied / free voxel clouds."""
        if keys.numel() == 0:
            return torch.empty(0, 3, device=keys.device, dtype=torch.float32)
        ix = ((keys >> (2 * _AXIS_BITS)) & _AXIS_MASK) - _AXIS_OFFSET
        iy = ((keys >> _AXIS_BITS) & _AXIS_MASK) - _AXIS_OFFSET
        iz = (keys & _AXIS_MASK) - _AXIS_OFFSET
        idx = torch.stack([ix, iy, iz], dim=-1).to(torch.float32)
        return idx * self.voxel_size + (self.voxel_size / 2)

    @torch.no_grad()
    def inspect_classification(self, world_xyz: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns three boolean masks of shape (N,): is_occupied, is_free,
        is_unknown. Together they partition the input. For visualization /
        debugging only — production code should prefer free_mask().
        """
        n = world_xyz.shape[0]
        device = world_xyz.device
        is_occupied = torch.zeros(n, dtype=torch.bool, device=device)
        is_free = torch.zeros(n, dtype=torch.bool, device=device)
        keys, in_range = self._xyz_to_keys(world_xyz)
        if self.occupied_keys is not None and self.occupied_keys.numel() > 0:
            in_occ = torch.isin(keys, self.occupied_keys)
            is_occupied[in_range] = in_occ
        if self.free_keys is not None and self.free_keys.numel() > 0:
            in_free = torch.isin(keys, self.free_keys)
            # free supersedes occupied? No — occupied wins (it's the stronger
            # evidence: an actual hit). So mask out occupied from free.
            is_free[in_range] = in_free & ~is_occupied[in_range]
        is_unknown = ~(is_occupied | is_free)
        return is_occupied, is_free, is_unknown
