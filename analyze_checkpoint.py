"""Analyze a checkpoint: where are the Gaussians relative to LiDAR sensor centers?

Usage:
    .venv/bin/python analyze_checkpoint.py \
        -ec configs/t4/exp_t4.yaml \
        -dc configs/t4/dynamic/example.yaml \
        -s ~/.webauto/data/data/annotation_dataset/<UUID>/<VER> \
        -m output/.../models/ckpt_it_13000_good.pth
"""

import argparse
import sys

import numpy as np
import torch

from lib.arguments import parse
from vis_rerun import load_scene_fast, load_gaussians_fast


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("-ec", "--exp_config_path", required=True)
    p.add_argument("-dc", "--data_config_path", required=True)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-s", "--source_dir", default="")
    launch = p.parse_args()
    args = parse(launch.exp_config_path)
    args = parse(launch.data_config_path, args)
    args.model_path = launch.model
    args.unet = ""
    args.eval_type = "all"
    args.rerun_save = ""
    if launch.source_dir:
        args.source_dir = launch.source_dir
    return args, launch


def main():
    args, launch = build_args()

    lidars, _bboxes = load_scene_fast(args)
    num_g = len(torch.load(args.model_path, map_location="cpu", weights_only=False)[0])
    gaussians, first_iter = load_gaussians_fast(args.model_path, num_g, args)

    # Gather all sensor centers across all frames (world coords)
    centers = []
    for sensor_name, lidar in lidars.items():
        for fid, c in lidar.sensor_center.items():
            centers.append(c.cpu().numpy())
    centers = np.stack(centers)
    print(f"\n[Sensors] {centers.shape[0]} sensor positions "
          f"(across {len(lidars)} sensor(s) x frames)")
    print(f"  x: [{centers[:, 0].min():.2f}, {centers[:, 0].max():.2f}]")
    print(f"  y: [{centers[:, 1].min():.2f}, {centers[:, 1].max():.2f}]")
    print(f"  z: [{centers[:, 2].min():.2f}, {centers[:, 2].max():.2f}]")

    centers_t = torch.tensor(centers, device="cuda", dtype=torch.float32)

    bg = gaussians[0]  # asset 0 is bg
    if bg.bounding_box is not None:
        print("ERROR: asset 0 has bbox, expected background.", file=sys.stderr)
        sys.exit(1)

    xyz = bg._xyz.detach().cuda()   # world coords for bg
    opa = torch.sigmoid(bg._opacity.detach()).squeeze(-1).cuda()  # activated opacity
    print(f"\n[Background Gaussians] {xyz.shape[0]:,} total at iter {first_iter}")
    print(f"  opacity   min={opa.min():.4f}  med={opa.median():.4f}  "
          f"max={opa.max():.4f}  mean={opa.mean():.4f}")

    # Distance from each Gaussian to the NEAREST sensor across all frames
    # (chunked to avoid OOM on 1.5M+ Gaussians x N frames).
    # Sensor centers are in UTM-like world coords (~1e5), so we shift to the
    # sensor centroid before cdist to avoid catastrophic-cancellation noise
    # in the |a|^2 + |b|^2 - 2*a*b expansion. See note in
    # gaussian_model.densify_and_prune for the full explanation.
    origin = centers_t.mean(dim=0, keepdim=True)
    xyz_shift = (xyz - origin).float()
    cents_shift = (centers_t - origin).float()
    dist_min = torch.empty(xyz.shape[0], device="cuda")
    chunk = 65536
    for s in range(0, xyz.shape[0], chunk):
        e = min(s + chunk, xyz.shape[0])
        d = torch.cdist(xyz_shift[s:e], cents_shift)
        dist_min[s:e] = d.min(dim=1).values

    # Histogram by distance bands
    print("\n[Distance to nearest sensor center]")
    bands = [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 5.0),
        (5.0, 10.0),
        (10.0, 30.0),
        (30.0, 100.0),
        (100.0, float("inf")),
    ]
    total = xyz.shape[0]
    print(f"  {'range (m)':<14} {'count':>10}  {'frac':>7}  {'avg opa':>8}  {'min opa':>8}")
    for lo, hi in bands:
        mask = (dist_min >= lo) & (dist_min < hi)
        n = int(mask.sum().item())
        if n == 0:
            print(f"  [{lo:>4.1f}, {hi:>5.1f})    {n:>10,}  {0:>6.2%}  {'-':>8}  {'-':>8}")
            continue
        sub_opa = opa[mask]
        print(f"  [{lo:>4.1f}, {hi:>5.1f})    {n:>10,}  {n/total:>6.2%}  "
              f"{sub_opa.mean().item():>8.4f}  {sub_opa.min().item():>8.4f}")

    # Specifically the "phantom danger zone": within hardware min range
    for cap in [1.0, 3.0]:
        mask = dist_min < cap
        n = int(mask.sum().item())
        if n > 0:
            sub_opa = opa[mask]
            above_prune = (sub_opa >= 0.003).sum().item()
            print(f"\n[< {cap}m of any sensor]")
            print(f"  count: {n:,} ({n/total:.2%})")
            print(f"  opacity stats: min={sub_opa.min():.4f} "
                  f"med={sub_opa.median():.4f} max={sub_opa.max():.4f}")
            print(f"  above prune threshold (0.003): {above_prune:,} / {n:,} "
                  f"({above_prune/n:.1%})")

    # Object Gaussians overview (xyz is in local bbox coord -> skip distance)
    if num_g > 1:
        total_obj = 0
        for i in range(1, num_g):
            total_obj += gaussians[i]._xyz.shape[0]
        print(f"\n[Object Gaussians] {num_g-1} bboxes, {total_obj:,} points total "
              f"(local coords, distance check skipped)")

    # Per-object bbox extent and world position over the trajectory
    print("\n[Per-object bbox] (size_xyz = bbox extent in local frame)")
    obj_ids = sorted(_bboxes.keys())
    sensor_xy_mean = centers_t[:, :2].mean(dim=0).cpu().numpy()
    sensor_z_avg = float(centers_t[:, 2].mean().item())
    for idx, oid in enumerate(obj_ids, start=1):
        if idx >= num_g:
            break
        bbox = _bboxes[oid]
        # bbox.min_xyz/max_xyz are the local frame extents (around 0)
        size = bbox.size.cpu().numpy()
        # bbox.frame: dict[frame] -> (pos, quat, dT, dR). Use mean center across frames
        centers_world = []
        for fid, tpl in bbox.frame.items():
            centers_world.append(tpl[0].cpu().numpy())
        if not centers_world:
            continue
        cw = np.stack(centers_world)
        c_mean = cw.mean(axis=0)
        horiz_to_ego = np.sqrt(((c_mean[:2] - sensor_xy_mean) ** 2).sum())
        bottom_z = c_mean[2] - size[2] / 2
        bottom_rel = bottom_z - sensor_z_avg
        n_g = gaussians[idx]._xyz.shape[0]
        print(f"  obj[{idx:>2}] id={oid[:8]}... size=({size[0]:5.2f}, {size[1]:5.2f}, {size[2]:5.2f})  "
              f"center≈({c_mean[0]:.1f},{c_mean[1]:.1f},{c_mean[2]:.2f})  "
              f"horiz_to_ego={horiz_to_ego:6.1f}m  "
              f"bottom_z_rel={bottom_rel:+5.2f}m  "
              f"N_g={n_g:>10,}")

    # bg xyz extents
    print("\n[Background Gaussian xyz extents]")
    print(f"  x: [{xyz[:,0].min().item():.2f}, {xyz[:,0].max().item():.2f}]")
    print(f"  y: [{xyz[:,1].min().item():.2f}, {xyz[:,1].max().item():.2f}]")
    print(f"  z: [{xyz[:,2].min().item():.2f}, {xyz[:,2].max().item():.2f}]")

    sensor_z_mean = float(centers_t[:, 2].mean().item())
    print(f"  sensor z (mean): {sensor_z_mean:.2f}")

    # z histogram (relative to sensor z)
    z_rel = (xyz[:, 2] - sensor_z_mean).cpu().numpy()
    print("\n[bg Gaussian z relative to sensor mean]")
    z_bands = [
        (-5.0, -3.0), (-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0),
        (0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 50.0),
    ]
    for lo, hi in z_bands:
        mask = (z_rel >= lo) & (z_rel < hi)
        n = int(mask.sum())
        print(f"  z_rel [{lo:>5.1f}, {hi:>5.1f}): {n:>10,}  ({n/total:.2%})")

    # Decompose horizontal vs vertical distance to nearest sensor
    # using a coarse "for each Gaussian, find nearest sensor index"
    dist_idx = torch.empty(xyz.shape[0], dtype=torch.long, device="cuda")
    for s in range(0, xyz.shape[0], chunk):
        e = min(s + chunk, xyz.shape[0])
        d = torch.cdist(xyz_shift[s:e], cents_shift)
        dist_idx[s:e] = d.argmin(dim=1)
    nearest = centers_t[dist_idx]
    horiz = torch.norm(xyz[:, :2] - nearest[:, :2], dim=1)
    vert = xyz[:, 2] - nearest[:, 2]

    # Highlight Gaussians on the road plane (z below sensor by ~ground-mount height)
    # i.e. vert in [-3, -1], horiz < 30 m
    road_band = (vert > -3.0) & (vert < -1.0) & (horiz < 30.0)
    n_road = int(road_band.sum().item())
    print(f"\n[Road-plane band: vert∈(-3,-1), horiz<30m] {n_road:,} Gaussians "
          f"({n_road/total:.2%})")

    # Same but only horiz < 5 m (the "definitely under the ego" zone)
    near_road = (vert > -3.0) & (vert < -1.0) & (horiz < 5.0)
    n_near = int(near_road.sum().item())
    print(f"[Right under ego: vert∈(-3,-1), horiz<5m]   {n_near:,} Gaussians "
          f"({n_near/total:.2%})")


if __name__ == "__main__":
    main()
