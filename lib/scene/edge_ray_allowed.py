"""Per-edge-ray allowed-depth intervals: a "where the model is allowed to
have geometry on this ray" map, precomputed once per scene.

For each edge pixel (sobel_edge | sky_boundary) in each train frame, we
sample N depths along the ray and query distance to the union of all
train frames' bg GT points (in world coords). Samples within
`dist_threshold` are marked "allowed". Allowed samples are compressed
into intervals [d_lo, d_hi] for fast loss evaluation.

Training loss: penalise rendered_depth at edge pixels for landing outside
any allowed interval. This catches midair phantoms on edge rays that
existing losses miss:
  - depth_l1 only at the on-ray GT depth (one specific position)
  - lambda_cd (chamfer) ray-agnostic — phantoms near a NEIGHBOUR ray's GT
                                       are not penalised
  - lambda_front_acc only catches α in front of the on-ray GT
  - lambda_occupancy per-Gaussian, voxel-quantised, ray-agnostic

The novelty here is RAY-LOCAL: a phantom on edge ray r at depth d_p is
penalised iff no GT point along r at depth near d_p exists, regardless of
whether other rays' GT points happen to be near (origin + d_p * dir_r) in
3D space.

Currently bg-only (does not handle dynamic objects' bbox interiors —
their GT positions move between frames, so unioning produces ghost
returns; mitigation is a future task). Static-scene assumption.
"""

import hashlib
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm


def _compute_edge_mask(gt_depth, gt_mask, edge_depth_grad_thresh):
    """Mirror of the edge_mask construction in train.py:309-340.

    Edge_mask = (Sobel-edge with valid 3x3 neighbourhood) ∪ (sky boundary).
    Operates on the same device as the input tensors.
    """
    gt_d = gt_depth.unsqueeze(0).unsqueeze(0).float()
    sx = torch.tensor(
        [[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]],
        dtype=gt_d.dtype, device=gt_d.device,
    )
    sy = sx.transpose(-1, -2).contiguous()
    gx = F.conv2d(gt_d, sx, padding=1).squeeze()
    gy = F.conv2d(gt_d, sy, padding=1).squeeze()
    edge_mag = torch.sqrt(gx * gx + gy * gy)
    invalid = (~gt_mask).float().unsqueeze(0).unsqueeze(0)
    any_invalid = F.max_pool2d(invalid, 3, stride=1, padding=1).squeeze() > 0.5
    sobel_edge = (edge_mag > edge_depth_grad_thresh) & ~any_invalid & gt_mask
    sky_boundary = any_invalid & gt_mask
    return sobel_edge | sky_boundary


def _intervals_from_mask(allowed_1d, depths_1d):
    """Compress a 1D boolean allowed mask into list of (d_lo, d_hi).

    allowed_1d: bool (K,)
    depths_1d:  float (K,) — sorted ascending
    Returns: list of (lo, hi) tuples; each contiguous True run is one
    interval. Allowed = depth in [depths_1d[run_start], depths_1d[run_end]].
    """
    if not allowed_1d.any():
        return []
    # Pad with 0 on both ends so np.diff catches edge runs.
    pad = np.concatenate([[0], allowed_1d.astype(np.int8), [0]])
    diff = np.diff(pad)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return [(float(depths_1d[s]), float(depths_1d[e]))
            for s, e in zip(starts, ends)]


class EdgeRayAllowed:
    """Per-edge-ray allowed-depth intervals, with disk cache.

    Workflow:
        era = EdgeRayAllowed(dist_threshold=0.5, num_samples=100)
        era.build_or_load_cache(lidars, cache_dir, cache_extra={...})
        era.to_cuda()
        # At training time:
        data = era.get_frame_data(sensor_name, frame_id)
        loss = edge_ray_allowed_loss(depth_render, data)
    """

    def __init__(self,
                 dist_threshold=0.5,
                 num_samples=100,
                 min_depth=0.5,
                 max_depth=100.0,
                 edge_depth_grad_thresh=15.0,
                 max_intervals_per_pixel=5):
        self.dist_threshold = float(dist_threshold)
        self.num_samples = int(num_samples)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.edge_depth_grad_thresh = float(edge_depth_grad_thresh)
        self.max_intervals_per_pixel = int(max_intervals_per_pixel)
        # dict[(sensor_name, frame_id)] -> dict of tensors
        self._data = {}
        self._stats = {}

    def _cache_hash(self, lidars, cache_extra):
        h = hashlib.sha1()
        for sname in sorted(lidars.keys()):
            lidar = lidars[sname]
            h.update(sname.encode())
            h.update(str(sorted(lidar.train_frames)).encode())
        h.update(f"{self.dist_threshold}".encode())
        h.update(f"{self.num_samples}".encode())
        h.update(f"{self.min_depth}".encode())
        h.update(f"{self.max_depth}".encode())
        h.update(f"{self.edge_depth_grad_thresh}".encode())
        if cache_extra:
            for k in sorted(cache_extra.keys()):
                h.update(f"{k}={cache_extra[k]}".encode())
        return h.hexdigest()[:16]

    def build_or_load_cache(self, lidars, cache_dir, cache_extra=None):
        os.makedirs(cache_dir, exist_ok=True)
        cache_hash = self._cache_hash(lidars, cache_extra)
        cache_file = os.path.join(cache_dir, f"{cache_hash}.pt")

        if os.path.exists(cache_file):
            t0 = time.time()
            saved = torch.load(cache_file, weights_only=False)
            self._data = saved["data"]
            self._stats = saved["stats"]
            self._stats["cached"] = True
            print(f"[EdgeRayAllowed] loaded cache from {cache_file} "
                  f"({time.time() - t0:.1f}s)")
            return self._stats

        t0 = time.time()
        self._stats = self._build(lidars)
        self._stats["cached"] = False
        torch.save({"data": self._data, "stats": self._stats}, cache_file)
        print(f"[EdgeRayAllowed] built cache to {cache_file} "
              f"({time.time() - t0:.1f}s)")
        return self._stats

    def _build(self, lidars):
        # Step 1: union all train frames' bg GT points in world coords.
        # Dynamic-object GT inside bbox is NOT excluded here (limitation noted
        # in module docstring). Static scene assumption.
        all_pts = []
        n_frames_seen = 0
        for sname, lidar in lidars.items():
            for fid in lidar.train_frames:
                if fid not in lidar.range_image_return1:
                    continue
                gt_depth = lidar.get_depth(fid)
                gt_mask = lidar.get_mask(fid)
                # range2point returns (H, W, 3) in world coords
                pts_world = lidar.range2point(fid, gt_depth)
                pts_flat = pts_world.reshape(-1, 3)
                mask_flat = gt_mask.reshape(-1).cuda() if not gt_mask.is_cuda else gt_mask.reshape(-1)
                pts_valid = pts_flat[mask_flat].cpu().numpy()
                if len(pts_valid) > 0:
                    all_pts.append(pts_valid)
                n_frames_seen += 1
        if not all_pts:
            raise RuntimeError("No GT points to build EdgeRayAllowed cache")
        gt_cloud = np.concatenate(all_pts, axis=0).astype(np.float64)
        print(f"[EdgeRayAllowed] building KDTree over {len(gt_cloud):,} GT "
              f"points (from {n_frames_seen} train frames)")
        tree = cKDTree(gt_cloud)

        # Step 2: log-spaced depth samples (more resolution near sensor).
        sample_depths = np.exp(np.linspace(
            np.log(self.min_depth), np.log(self.max_depth),
            self.num_samples)).astype(np.float64)

        # Step 3: per (sensor, frame) per edge pixel
        n_total_pixels = 0
        n_total_intervals = 0
        for sname, lidar in lidars.items():
            for fid in tqdm(lidar.train_frames,
                            desc=f"EdgeRayAllowed[{sname}]", leave=False):
                if fid not in lidar.range_image_return1:
                    continue
                gt_depth = lidar.get_depth(fid).cuda()
                gt_mask = lidar.get_mask(fid).cuda()
                edge_mask = _compute_edge_mask(
                    gt_depth, gt_mask, self.edge_depth_grad_thresh)
                if not edge_mask.any():
                    continue
                H, W = gt_depth.shape
                edge_flat = edge_mask.reshape(-1)
                pixel_indices = torch.nonzero(edge_flat).squeeze(-1).cpu().numpy()
                n_pixels = len(pixel_indices)

                # Compute ray directions via range2point at unit range:
                #   range2point returns origin + range * unit_dir, so
                #   range=1 gives origin + unit_dir.
                unit_pts = lidar.range2point(fid, torch.ones_like(gt_depth))
                unit_pts_flat = unit_pts.reshape(-1, 3).cpu().numpy().astype(np.float64)
                origin = lidar.sensor_center[fid].cpu().numpy().astype(np.float64)
                directions = unit_pts_flat[pixel_indices] - origin  # (N, 3) unit

                # Build (N * num_samples, 3) sample points along each ray
                sample_pts = (
                    origin[None, None, :]
                    + sample_depths[None, :, None] * directions[:, None, :]
                )  # (N, S, 3)
                sample_pts_flat = sample_pts.reshape(-1, 3)

                # Single batched KDTree query — much faster than per-point.
                # workers=-1 uses all CPU cores; without it scipy runs
                # single-threaded and a single frame takes ~7 min for the
                # ~4.8M points × 8M-point tree scale. With 32 cores the
                # per-frame cost drops to ~15-30 s.
                dists, _ = tree.query(sample_pts_flat, k=1, workers=-1)
                dists = dists.reshape(n_pixels, self.num_samples)
                allowed = dists < self.dist_threshold  # (N, S)

                K = self.max_intervals_per_pixel
                intervals_lo = np.full((n_pixels, K), np.inf, dtype=np.float32)
                intervals_hi = np.full((n_pixels, K), np.inf, dtype=np.float32)
                n_intervals = np.zeros(n_pixels, dtype=np.int64)
                for i in range(n_pixels):
                    ivs = _intervals_from_mask(allowed[i], sample_depths)
                    n_use = min(len(ivs), K)
                    for k in range(n_use):
                        intervals_lo[i, k] = ivs[k][0]
                        intervals_hi[i, k] = ivs[k][1]
                    n_intervals[i] = n_use
                    n_total_intervals += n_use

                key = (sname, int(fid))
                self._data[key] = {
                    "pixel_indices": torch.from_numpy(
                        pixel_indices.astype(np.int64)),
                    "intervals_lo": torch.from_numpy(intervals_lo),
                    "intervals_hi": torch.from_numpy(intervals_hi),
                    "n_intervals": torch.from_numpy(n_intervals),
                    "H": int(H), "W": int(W),
                }
                n_total_pixels += n_pixels

        return {
            "n_gt_pts": int(len(gt_cloud)),
            "n_edge_pixels": int(n_total_pixels),
            "n_intervals_total": int(n_total_intervals),
            "n_frames": int(n_frames_seen),
            "dist_threshold": self.dist_threshold,
            "num_samples": self.num_samples,
            "edge_depth_grad_thresh": self.edge_depth_grad_thresh,
        }

    def get_frame_data(self, sensor_name, frame_id):
        return self._data.get((sensor_name, int(frame_id)))

    def to_cuda(self):
        for key in self._data:
            for tk in self._data[key]:
                v = self._data[key][tk]
                if torch.is_tensor(v):
                    self._data[key][tk] = v.cuda()


def edge_ray_allowed_loss(depth_rendered, frame_data, max_dist=10.0):
    """Distance from rendered_depth to nearest allowed interval, averaged
    over edge pixels of this frame.

    depth_rendered: (H, W) tensor; can have a trailing singleton dim
                    (squeezed inside).
    frame_data: dict from EdgeRayAllowed.get_frame_data(), or None.
    max_dist: clamp on per-pixel distance so an extreme outlier doesn't
              dominate the gradient (similar to a Huber-style cap).

    Returns: scalar tensor. 0 if no edge data for this frame.
    """
    if frame_data is None:
        return torch.tensor(0.0, device=depth_rendered.device)

    pixel_indices = frame_data["pixel_indices"]
    intervals_lo = frame_data["intervals_lo"]
    intervals_hi = frame_data["intervals_hi"]
    n_intervals = frame_data["n_intervals"]

    if depth_rendered.dim() > 2:
        depth_rendered = depth_rendered.squeeze(-1)
    depth_flat = depth_rendered.reshape(-1)
    d_at_edge = depth_flat[pixel_indices]  # (N,)
    d = d_at_edge.unsqueeze(-1)            # (N, 1)

    # Distance to each interval: 0 if inside [lo, hi], else min(|d-lo|, |d-hi|).
    lo = intervals_lo  # (N, K), padded with +inf
    hi = intervals_hi
    inside = (d >= lo) & (d <= hi)         # (N, K)
    dist_iv = torch.where(
        inside,
        torch.zeros_like(d.expand_as(lo)),
        torch.minimum((d - lo).abs(), (d - hi).abs()),
    )
    # Padded slots have lo=+inf so their dist becomes inf — they lose the min.
    valid_iv = torch.isfinite(lo)
    dist_iv = torch.where(
        valid_iv, dist_iv, torch.full_like(dist_iv, float("inf")))

    min_dist, _ = dist_iv.min(dim=-1)      # (N,)

    has_iv = n_intervals > 0
    if not has_iv.any():
        return torch.tensor(0.0, device=depth_rendered.device)
    return min_dist[has_iv].clamp(0.0, max_dist).mean()
