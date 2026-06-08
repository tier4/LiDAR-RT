"""Rerun visualization for LiDAR-RT evaluation results.

Fast visualization that skips heavy scene initialization (normal estimation,
voxel downsampling, optimizer restore) and loads only what's needed.

Usage:
    python vis_rerun.py -dc configs/t4/dynamic/example.yaml -ec configs/t4/exp_t4.yaml \
        -m output/t4_test/test/scene_t4d1/models/ckpt_it_24000_good.pth \
        -un output/t4_test/test/scene_t4d1/models/unet.pth \
        -t all
"""

import argparse
import os
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from lib.arguments import parse
from lib.gaussian_renderer import raytracing
from lib.scene import BoundingBox, GaussianModel, LiDARSensor
from lib.scene.unet import UNet
from lib.utils.image_utils import color_mapping, colorize_depth, colorize_intensity
from tqdm import tqdm

RAYDROP_RATIO = 0.5
COLORMAP = 20  # JET


def build_args():
    parser = argparse.ArgumentParser(description="Rerun visualization for LiDAR-RT")
    parser.add_argument("-dc", "--data_config_path", type=str, required=True)
    parser.add_argument("-ec", "--exp_config_path", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-un", "--unet", type=str, default="")
    parser.add_argument(
        "-t", "--type", type=str, default="all", help="train/test/all"
    )
    parser.add_argument("--save", type=str, default="", help="Save .rrd file path")
    parser.add_argument(
        "-s",
        "--source_dir",
        type=str,
        default="",
        help="override source_dir from data config",
    )
    launch = parser.parse_args()

    args = parse(launch.exp_config_path)
    args = parse(launch.data_config_path, args)
    args.model_path = launch.model
    args.unet = launch.unet
    args.eval_type = launch.type
    args.rerun_save = launch.save
    if launch.source_dir:
        args.source_dir = launch.source_dir
    return args


def load_scene_fast(args):
    """Load LiDAR data and bounding boxes WITHOUT heavy Gaussian initialization.

    Skips: normal estimation, voxel downsampling, optimizer setup.
    Only loads: range images, transforms, bounding boxes.

    Returns:
        lidars: dict[str, LiDARSensor]
        bboxes: dict[str, BoundingBox]
    """
    t0 = time.time()

    # Load raw LiDAR data (range images + transforms + bboxes)
    if getattr(args, "data_type", None) == "T4":
        from lib.dataloader import t4_loader
        lidars, bboxes = t4_loader.load_t4_raw(args.source_dir, args)
    elif "waymo" in args.source_dir:
        from lib.dataloader import waymo_loader
        lidar, bboxes = waymo_loader.load_waymo_raw(args.source_dir, args)
        lidars = {"waymo_top": lidar}
    elif "kitti" in args.source_dir:
        from lib.dataloader import kitti_loader
        lidar, bboxes = kitti_loader.load_kitti_raw(args.source_dir, args)
        lidars = {"kitti": lidar}
    else:
        raise ValueError("Unknown dataset type")

    # Setup frame lists on all sensors
    frame_range = args.frame_length
    eval_frames = list(args.eval_frames)
    train_frames = [
        fid for fid in range(frame_range[0], frame_range[1] + 1)
        if fid not in eval_frames
    ]

    # Mirror the loader's drop filter so visualization skips broken frames too
    skip_drops = getattr(args, "skip_dropped_frames", False)
    weak_col_thr = float(getattr(args, "dropped_frame_threshold", 0.01))
    weak_hit_rate = float(getattr(args, "dropped_frame_hit_rate", 0.5))
    if skip_drops:
        dropped = set()
        for sensor_name, lidar in lidars.items():
            bad = lidar.detect_dropped_frames(
                weak_col_threshold=weak_col_thr,
                weak_hit_rate=weak_hit_rate,
            )
            if bad:
                print(f"[{sensor_name}] {len(bad)} dropped frame(s) "
                      f"(weak_col > {weak_col_thr:.0%}): "
                      + ", ".join(f"{fid}({frac*100:.1f}%)" for fid, frac in bad))
                dropped.update(fid for fid, _ in bad)
        if dropped:
            before_t, before_e = len(train_frames), len(eval_frames)
            train_frames = [f for f in train_frames if f not in dropped]
            eval_frames = [f for f in eval_frames if f not in dropped]
            print(f"[skip_dropped_frames] train: {before_t} -> {len(train_frames)}, "
                  f"eval: {before_e} -> {len(eval_frames)}")

    for sensor_name, lidar in lidars.items():
        lidar.set_frames(train_frames, eval_frames)

    print(f"LiDAR data loaded in {time.time() - t0:.1f}s ({len(lidars)} sensor(s))", flush=True)
    return lidars, bboxes


def load_gaussians_fast(model_path, num_gaussians, args):
    """Load trained Gaussian models from checkpoint, skipping optimizer state."""
    t0 = time.time()

    # Create empty Gaussian models
    gaussians = []
    for _ in range(num_gaussians):
        gs = GaussianModel(args.model.dimension, args.model.sh_degree)
        gaussians.append(gs)

    # Load checkpoint
    model_params, first_iter = torch.load(model_path, map_location="cuda")
    assert len(model_params) == num_gaussians, \
        f"Checkpoint has {len(model_params)} gaussians, expected {num_gaussians}"

    for i, gs in enumerate(gaussians):
        (gs.active_sh_degree,
         gs._xyz, gs._features_dc, gs._features_rest,
         gs._scaling, gs._rotation, gs._opacity,
         gs.max_radii2D,
         _, _, _,  # skip: xyz_gradient_accum, denom, optimizer_state
         gs.spatial_lr_scale) = model_params[i]
        print(f"  Gaussian[{i}]: {gs._xyz.shape[0]:,} points", flush=True)

    print(f"Model loaded in {time.time() - t0:.1f}s (iter {first_iter})", flush=True)
    return gaussians, first_iter


def depth_to_colormap(depth, vmin, vmax):
    norm = (depth - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)
    colored = (color_mapping(norm, COLORMAP)[..., :3] * 255).astype(np.uint8)
    return colored


def intensity_to_colormap(intensity):
    norm = np.clip(intensity, 0, 1)
    colored = (color_mapping(norm, COLORMAP)[..., :3] * 255).astype(np.uint8)
    return colored


def depth_to_colors(depth, vmin=0.0, vmax=120.0):
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colors = cv2.applyColorMap(
        (norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET
    )
    return colors.reshape(-1, 3)[:, ::-1].copy()  # BGR -> RGB


def intensity_to_point_colors(intensity_1d, fade=0.0,
                              vmin=0.0, vmax=64.0, log_scale=True):
    """Map (N,) intensity in raw LiDAR scale to (N, 3) RGB uint8 via the JET
    colormap.

    T4 LiDAR intensity is uint8 in raw 0-255 with a heavy long-tail
    distribution (p50 ≈ 12, p90 ≈ 30, p99 ≈ 64, retroreflectors at 255).
    A linear [0, 255] ramp wastes most of the colormap on the empty
    high-intensity tail, while a linear [0, 1] mapping clips ~99% of points
    to the top of the ramp. Defaulting to log-scale [0, 64] spreads the
    visible spectrum across the meaningful band where 99% of points live;
    anything above 64 saturates to the top color (deep red) instead of
    pulling the rest of the distribution into blue.

    fade in [0, 1] blends toward white for the GT "pale wash" look so a
    saturated rendered cloud can be visually distinguished from a faded GT
    one when overlaid in the 3D view.
    """
    v = np.asarray(intensity_1d, dtype=np.float32)
    if log_scale:
        v = np.log1p(np.maximum(v, 0.0))
        lo = np.log1p(max(vmin, 0.0))
        hi = np.log1p(max(vmax, vmin + 1e-3))
    else:
        lo, hi = float(vmin), float(vmax)
    norm = np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    norm_u8 = (norm * 255).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET).reshape(-1, 3)
    rgb = bgr[:, ::-1].astype(np.float32)
    if fade > 0:
        rgb = rgb * (1.0 - fade) + 255.0 * fade
    return rgb.astype(np.uint8)


def main():
    args = build_args()
    data_type = getattr(args, "data_type", "unknown")
    save_path = args.rerun_save or f"output/vis_rerun_{data_type.lower()}.rrd"

    # --- Init Rerun ---
    rr.init(f"LiDAR-RT {data_type} Visualization")
    rr.save(save_path)
    print(f"Saving to {save_path}", flush=True)

    # --- Load data (fast path) ---
    lidars, bboxes = load_scene_fast(args)
    sensor_names = list(lidars.keys())
    multi_sensor = len(sensor_names) > 1

    # Determine number of gaussian models from checkpoint
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    num_gaussians = len(checkpoint[0])
    del checkpoint  # free memory

    gaussians, first_iter = load_gaussians_fast(args.model_path, num_gaussians, args)

    # Load per-sensor UNets
    unets = {}
    if args.unet:
        in_channels = 9 if args.refine.use_spatial else 3
        if os.path.isfile(args.unet):
            # Single unet file: apply to first sensor
            unet = UNet(in_channels=in_channels, out_channels=1).cuda()
            unet.load_state_dict(torch.load(args.unet))
            unets[sensor_names[0]] = unet
            print(f"Loaded UNet for {sensor_names[0]}", flush=True)
        else:
            # Look for per-sensor unet files
            unet_dir = args.unet if os.path.isdir(args.unet) else os.path.dirname(args.model_path)
            for sname in sensor_names:
                unet_path = os.path.join(unet_dir, f"unet_{sname}.pth")
                if os.path.exists(unet_path):
                    unet = UNet(in_channels=in_channels, out_channels=1).cuda()
                    unet.load_state_dict(torch.load(unet_path))
                    unets[sname] = unet
                    print(f"Loaded UNet for {sname}", flush=True)

    background = torch.tensor([0, 0, 1], device="cuda").float()

    # --- Determine frames ---
    # Use the per-sensor frame lists set up by load_lidar_data(), which have
    # already been filtered for dropped frames if skip_dropped_frames=True.
    first_lidar_obj = next(iter(lidars.values()))
    train_frames = list(first_lidar_obj.train_frames)
    eval_frames = list(first_lidar_obj.eval_frames)
    if args.eval_type == "train":
        all_frames = train_frames
    elif args.eval_type == "test":
        all_frames = eval_frames
    else:
        all_frames = sorted(set(train_frames) | set(eval_frames))

    print(f"Rendering {len(all_frames)} frames x {len(sensor_names)} sensor(s)...", flush=True)

    # --- Static logging ---
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # Flag post-prune checkpoints (produced by sky_prune_checkpoint.py) so the
    # viewer header makes the variant unambiguous when flipping between runs.
    is_post_prune = "_skyprune" in os.path.basename(args.model_path)
    rr.log("metadata", rr.TextDocument(
        f"Model: {args.model_path}\nIteration: {first_iter}"
        + ("  [POST-PRUNE]" if is_post_prune else "")
        + f"\nFrames: {len(all_frames)} ({args.eval_type})\n"
        f"Sensors: {', '.join(sensor_names)}"
    ))

    # --- Blueprint: 3D view + per-sensor range image panels ---
    range_image_views = []
    for sname in sensor_names:
        prefix = f"range_image/{sname}" if multi_sensor else "range_image"
        range_image_views.extend([
            rrb.Spatial2DView(name=f"Depth GT ({sname})" if multi_sensor else "Depth GT",
                              contents=f"{prefix}/depth/gt/**"),
            rrb.Spatial2DView(name=f"Depth Rendered ({sname})" if multi_sensor else "Depth Rendered",
                              contents=f"{prefix}/depth/rendered/**"),
        ])
    range_image_views.append(
        rrb.TimeSeriesView(name="Depth Error (m/point)", contents="metrics/**"),
    )

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(
                    name="3D World (GT + Rendered)",
                    origin="world",
                    background=rrb.Background(color=[30, 30, 30]),
                    eye_controls=rrb.EyeControls3D(
                        kind="Orbital",
                        position=[0, 0, 80],
                        look_target=[0, 0, 0],
                        eye_up=[1, 0, 0],
                        tracking_entity="world/ego",
                    ),
                ),
                row_shares=[1],
            ),
            rrb.Vertical(
                *range_image_views,
                row_shares=[1] * len(range_image_views),
            ),
            column_shares=[3, 1],
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
        rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
        rrb.TimePanel(state=rrb.PanelState.Expanded),
    )
    rr.send_blueprint(blueprint)

    # --- Compute origin offset from first frame's ego position (first sensor) ---
    first_lidar = lidars[sensor_names[0]]
    first_ego = first_lidar.ego2world[all_frames[0]]
    if torch.is_tensor(first_ego):
        first_ego = first_ego.cpu().numpy()
    origin_offset = first_ego[:3, 3].copy()
    print(f"Origin offset (first frame ego): {origin_offset}", flush=True)

    # --- Occupancy grid visualisation (disabled) ---
    # The translucent red occupied-voxel boxes were turned off in favour of
    # direct per-Gaussian ellipsoid rendering below. Re-enable by flipping
    # the flag if you need the voxel-grid overlay back (and keep voxel_size
    # ≥ 0.3 m or rerun starts struggling at >1M cubes).
    SHOW_OCCUPANCY_BOXES = False
    if SHOW_OCCUPANCY_BOXES:
        try:
            from lib.scene.occupancy_grid import WorldOccupancyGrid
            class _BBoxRef:
                __slots__ = ("bounding_box",)
                def __init__(self, b):
                    self.bounding_box = b
            bbox_refs = [_BBoxRef(b) for b in bboxes.values()]
            occ_voxel_size = float(getattr(args.opt, "occupancy_voxel_size", 0.5))
            occ_max_depth = float(getattr(args.opt, "occupancy_free_max_depth", 100.0))
            occupancy_grid = WorldOccupancyGrid(voxel_size=occ_voxel_size)
            cache_dir = os.path.join(args.model_dir, ".occupancy_grid_cache")
            stats = occupancy_grid.build_or_load_cache(
                lidars, bbox_refs, max_depth=occ_max_depth, cache_dir=cache_dir,
                cache_extra={"source_dir": str(getattr(args, "source_dir", "")),
                             "frame_length": list(args.frame_length)},
            )
            OCC_VIZ_CAP = 500_000
            occ_keys = occupancy_grid.occupied_keys
            if occ_keys.numel() > OCC_VIZ_CAP:
                idx = torch.randint(0, occ_keys.numel(), (OCC_VIZ_CAP,), device=occ_keys.device)
                occ_keys = occ_keys[idx]
            occ_np = (occupancy_grid.voxel_centers(occ_keys).cpu().numpy().astype(np.float64) - origin_offset)
            if occ_np.shape[0] > 0:
                sizes = np.full((occ_np.shape[0], 3), float(occ_voxel_size), dtype=np.float32)
                colors = np.tile(np.array([255, 0, 0, 60], dtype=np.uint8), (occ_np.shape[0], 1))
                rr.log("world/occupancy/occupied",
                       rr.Boxes3D(centers=occ_np, sizes=sizes, colors=colors),
                       static=True)
        except Exception as e:
            print(f"[Occupancy] viz skipped: {type(e).__name__}: {e}", flush=True)

    # --- bg Gaussian ellipsoid visualisation ---
    # Renders each (subsampled) bg surfel as an oriented 3D ellipsoid in
    # rerun so the shape distribution is directly inspectable: needle-like
    # phantoms appear as thin red ellipsoids, healthy surfels as more
    # isotropic green ones. The third axis is set tiny (2D surfel → thin
    # disc) to preserve geometric meaning rather than inflate to a sphere.
    # Subsampled to keep the viewer responsive (2M+ ellipsoids would tank
    # it). Colour encodes aniso (green→red over σmax/σmin = 1→20+), alpha
    # encodes opacity (transparent → opaque).
    try:
        bg = gaussians[0]
        n_bg = bg._xyz.shape[0]
        BG_VIZ_CAP = 150_000
        if n_bg > BG_VIZ_CAP:
            sel = torch.randint(0, n_bg, (BG_VIZ_CAP,), device=bg._xyz.device)
        else:
            sel = torch.arange(n_bg, device=bg._xyz.device)

        xyz = bg._xyz.detach()[sel]
        rot_wxyz = bg._rotation.detach()[sel]
        sc_2d = torch.exp(bg._scaling.detach()[sel])           # (S, 2)
        opa = torch.sigmoid(bg._opacity.detach()[sel]).view(-1)  # (S,)

        # Half-sizes: σ on the two surfel axes, tiny on the normal axis so
        # the ellipsoid stays disc-shaped (matches the 2D-Gaussian model).
        thickness = 0.005
        half_sizes = torch.cat([
            sc_2d,
            torch.full((sc_2d.shape[0], 1), thickness, device=sc_2d.device),
        ], dim=-1).cpu().numpy()

        # Quaternion: model stores (w, x, y, z); rerun expects (x, y, z, w).
        rot_norm = rot_wxyz / rot_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        quats_xyzw = rot_norm[:, [1, 2, 3, 0]].cpu().numpy()

        # Colour: aniso → hue (green at 1, red at ≥20), opacity → alpha
        # (min alpha=30 so even near-zero α stays faintly visible).
        aniso = (sc_2d.max(-1).values / sc_2d.min(-1).values.clamp_min(1e-12)).clamp_max(50.0)
        aniso_norm = ((aniso - 1.0) / 19.0).clamp(0.0, 1.0)
        r = (aniso_norm * 255).to(torch.uint8)
        g = ((1.0 - aniso_norm) * 255).to(torch.uint8)
        b = torch.zeros_like(r)
        a = (opa * 180 + 30).clamp(0, 255).to(torch.uint8)
        colors = torch.stack([r, g, b, a], dim=-1).cpu().numpy()

        centers = xyz.cpu().numpy().astype(np.float64) - origin_offset

        # FillMode.Solid renders filled ellipsoids; falls back to wireframe
        # if the rerun version doesn't expose it.
        try:
            fill_mode = rr.components.FillMode.Solid
        except Exception:
            fill_mode = None

        if centers.shape[0] > 0:
            rr.log(
                "world/gaussians/bg",
                rr.Ellipsoids3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    quaternions=quats_xyzw,
                    colors=colors,
                    fill_mode=fill_mode,
                ),
                static=True,
            )
            n_needle = int(((aniso > 10) & (opa > 0.3)).sum())
            print(f"[Gaussians] bg: {n_bg:,} → viz {centers.shape[0]:,} ellipsoids "
                  f"(green→red by aniso, α by opacity); "
                  f"~{n_needle:,} sampled as needles (aniso>10 & α>0.3)",
                  flush=True)
    except Exception as e:
        print(f"[Gaussians] bg viz skipped: {type(e).__name__}: {e}", flush=True)

    # --- Ego-prune swept-volume visualisation ---
    # Renders the ego OBB(s) at every cached pose (base + interpolated) as
    # translucent red boxes under world/ego_prune. This is the volume where
    # bg Gaussians should not exist (and are hard-pruned every densify
    # cycle by the ego-prune path in densify_and_prune). Useful for
    # eyeballing whether a remaining bg ellipsoid sits inside the swept
    # volume — should be 0 after iter 1600.
    try:
        ego_prune_enabled_cfg = bool(getattr(args.opt, "ego_prune_enabled", False))
        if ego_prune_enabled_cfg:
            # Mirror the (base + 5 interp/segment) cache from gs_loader.optimize.
            base_poses = []
            for sname, lidar in lidars.items():
                for fid in sorted(set(lidar.train_frames) | set(lidar.eval_frames)):
                    if fid in lidar.ego2world:
                        pose = lidar.ego2world[fid]
                        if not torch.is_tensor(pose):
                            pose = torch.from_numpy(np.asarray(pose))
                        base_poses.append(pose.float())
            if base_poses:
                base = torch.stack(base_poses, dim=0)  # (M0, 4, 4) on CPU
                interp_n = 5
                if base.shape[0] >= 2:
                    a = base[:-1]
                    b = base[1:]
                    ts = torch.linspace(0.0, 1.0, interp_n + 2)[1:-1]
                    interps = (a[None] + ts[:, None, None, None]
                                          * (b[None] - a[None]))
                    interps = interps.permute(1, 0, 2, 3).reshape(-1, 4, 4)
                    all_poses = torch.cat([base, interps], dim=0)
                else:
                    all_poses = base
                # Bboxes from config: list of [min_x..max_z] in ego frame
                ego_bboxes_cfg = getattr(args.opt, "ego_bboxes", None)
                if not ego_bboxes_cfg:
                    ego_bboxes_cfg = [[-0.85, -0.8475, 0.0, 3.55, 0.8475, 2.5]]
                bb_arr = np.asarray(ego_bboxes_cfg, dtype=np.float32).reshape(
                    -1, 2, 3)  # (K, 2, 3)
                # Per-bbox ego-frame center / full-size (rerun's Boxes3D uses
                # full sizes, not half-sizes — matches how world/bboxes logs).
                ego_centers = (bb_arr[:, 0, :] + bb_arr[:, 1, :]) / 2.0   # (K, 3)
                ego_sizes = bb_arr[:, 1, :] - bb_arr[:, 0, :]              # (K, 3)
                # Transform each (pose, bbox) into world.
                R = all_poses[:, :3, :3].numpy()   # (M, 3, 3)
                t = all_poses[:, :3, 3].numpy()    # (M, 3)
                M = R.shape[0]
                K = ego_centers.shape[0]
                # world_center[m, k] = R[m] @ ego_centers[k] + t[m]
                world_centers = (R[:, None, :, :] @
                                  ego_centers[None, :, :, None]).squeeze(-1) \
                                + t[:, None, :]                            # (M, K, 3)
                world_centers = (world_centers.reshape(-1, 3).astype(np.float64)
                                  - origin_offset)                          # (M*K, 3)
                sizes = np.tile(ego_sizes[None, :, :], (M, 1, 1)).reshape(-1, 3)
                # Quaternion (xyzw) from each rotation matrix.
                rot_t = torch.from_numpy(R).float()
                # Manual matrix → quaternion (wxyz) then reorder.
                tr = rot_t[..., 0, 0] + rot_t[..., 1, 1] + rot_t[..., 2, 2]
                qw = torch.sqrt(torch.clamp(1 + tr, min=0)) / 2
                qx = (rot_t[..., 2, 1] - rot_t[..., 1, 2]) / (4 * qw.clamp_min(1e-8))
                qy = (rot_t[..., 0, 2] - rot_t[..., 2, 0]) / (4 * qw.clamp_min(1e-8))
                qz = (rot_t[..., 1, 0] - rot_t[..., 0, 1]) / (4 * qw.clamp_min(1e-8))
                quats_xyzw_pose = torch.stack([qx, qy, qz, qw], dim=-1).numpy()  # (M, 4)
                quats = np.tile(quats_xyzw_pose[:, None, :],
                                 (1, K, 1)).reshape(-1, 4)
                # Translucent red — RGBA alpha 40/255 so dense overlaps stay
                # readable without occluding the GT/rendered point clouds.
                colors = np.tile(np.array([255, 0, 0, 40], dtype=np.uint8),
                                  (world_centers.shape[0], 1))
                rr.log("world/ego_prune/swept_volume",
                       rr.Boxes3D(centers=world_centers, sizes=sizes,
                                  quaternions=quats, colors=colors),
                       static=True)
                print(f"[ego_prune viz] logged {M} poses × {K} bbox(es) "
                      f"= {M*K} boxes", flush=True)
    except Exception as e:
        print(f"[ego_prune viz] skipped: {type(e).__name__}: {e}", flush=True)

    # Per-sensor colors for 3D points
    SENSOR_COLORS = [
        [255, 255, 255],  # white
        [0, 200, 255],    # cyan
        [255, 200, 0],    # yellow
        [200, 100, 255],  # purple
        [255, 100, 100],  # light red
        [100, 255, 100],  # light green
    ]

    # --- Render loop ---
    t_render = time.time()
    for frame_id in tqdm(all_frames, desc="Rendering"):
        rr.set_time_sequence("frame", frame_id)

        all_gt_pts_frame = []
        all_gt_colors_frame = []
        all_rd_pts_frame = []
        all_rd_colors_frame = []
        frame_mae_values = []

        for si, sensor_name in enumerate(sensor_names):
            lidar = lidars[sensor_name]
            unet = unets.get(sensor_name)
            ri_prefix = f"range_image/{sensor_name}" if multi_sensor else "range_image"
            sensor_color = SENSOR_COLORS[si % len(SENSOR_COLORS)]

            # Ray tracing
            rendered_pkg = raytracing(frame_id, gaussians, lidar, background, args)
            rendered_depth = rendered_pkg["depth"].detach()
            rendered_intensity = rendered_pkg["intensity"].detach()
            rendered_raydrop = rendered_pkg["raydrop"].detach()

            # UNet refinement
            if unet:
                H, W = rendered_depth.shape[0], rendered_depth.shape[1]
                inp = torch.cat([
                    rendered_raydrop.reshape(1, H, W),
                    rendered_intensity.reshape(1, H, W),
                    rendered_depth.reshape(1, H, W),
                ], dim=0)
                if args.refine.use_spatial:
                    ray_o, ray_d = lidar.get_range_rays(frame_id)
                    inp = torch.cat([inp, ray_o.permute(2, 0, 1), ray_d.permute(2, 0, 1)], dim=0)
                rendered_raydrop = unet(inp.unsqueeze(0)).detach().reshape(H, W, 1)

            # Ground truth
            gt_rayhit = lidar.get_mask(frame_id).unsqueeze(-1)
            gt_depth = lidar.get_depth(frame_id)
            # Raw intensity (T4: uint8 0-255). Kept un-clamped for the 3D
            # point-cloud colormap which now handles the raw scale natively
            # with log-range mapping. The clamped [0, 1] version stays for
            # the 2D range-image viz which still expects normalised input.
            gt_intensity_raw = lidar.get_intensity(frame_id)
            gt_intensity = gt_intensity_raw.clamp(0, 1)

            gt_rayhit_np = gt_rayhit.cpu().numpy()
            gt_depth_np = gt_depth.unsqueeze(-1).cpu().numpy()
            gt_intensity_np = gt_intensity.unsqueeze(-1).cpu().numpy()
            gt_intensity_raw_np = gt_intensity_raw.cpu().numpy()
            rendered_rayhit_np = (rendered_raydrop < RAYDROP_RATIO).cpu().numpy()
            rendered_depth_np = rendered_depth.cpu().numpy()
            rendered_intensity_np = rendered_intensity.clamp(0, 1).cpu().numpy()
            rendered_intensity_raw_np = rendered_intensity.cpu().numpy()
            mask = rendered_rayhit_np

            # 3D point clouds. Filter out unrealistic close-range hits so the
            # rerun view does not show a cloud of points right next to the
            # sensor. Default 0 = no filter; set to the LiDAR hardware-min
            # range (e.g. 1.0 m) in the data config to enable.
            viz_min_depth = float(getattr(args, "viz_min_depth", 0.0))
            gt_all_pts = lidar.range2point(frame_id, gt_depth_np).cpu().numpy().astype(np.float64)
            rd_all_pts = lidar.range2point(frame_id, rendered_depth_np).cpu().numpy().astype(np.float64)

            gt_mask_2d = gt_rayhit_np.squeeze(-1).astype(bool)
            rd_mask_2d = (mask & (rendered_depth_np > viz_min_depth)).squeeze(-1).astype(bool)

            gt_pts = gt_all_pts[gt_mask_2d] - origin_offset
            rendered_pts = rd_all_pts[rd_mask_2d] - origin_offset

            # Per-point intensity colors. Both clouds run through the same
            # log-ramp colormap on raw intensity (default [0, 64] which
            # covers p99 of T4 LiDAR), but GT is heavily faded toward white
            # while rendered stays saturated — same colormap, different
            # visual weight, so the two clouds remain distinguishable when
            # overlaid in the 3D view. `sensor_color` is intentionally not
            # used: the per-sensor entity paths still keep them toggleable.
            gt_int_vals = gt_intensity_raw_np[gt_mask_2d]
            rd_int_vals = rendered_intensity_raw_np.squeeze(-1)[rd_mask_2d]
            gt_colors = intensity_to_point_colors(gt_int_vals, fade=0.65)
            rd_colors = intensity_to_point_colors(rd_int_vals, fade=0.0)

            # Per-sensor 3D points
            rr.log(f"world/pointcloud/gt/{sensor_name}",
                    rr.Points3D(gt_pts, colors=gt_colors, radii=0.05))
            rr.log(f"world/pointcloud/rendered/{sensor_name}",
                    rr.Points3D(rendered_pts, colors=rd_colors, radii=0.05))

            # Range images. Same masks/colormap as the wandb visualization in
            # train.py: GT uses gt_mask, pred uses depth>0 (any Gaussian hit).
            # Note this intentionally ignores the UNet-refined raydrop here so
            # both viewers show the same picture; the raydrop-filtered point
            # cloud is still logged separately above.
            dmin = 0.0
            dmax = max(float(gt_depth_np.max()), float(rendered_depth_np.max()))

            gt_depth_2d = gt_depth_np.squeeze(-1)
            rd_depth_2d = rendered_depth_np.squeeze(-1)
            pred_hit_any = rd_depth_2d > 0
            pred_phantom = pred_hit_any & (rd_depth_2d <= viz_min_depth)

            # colorize_depth returns BGR; rerun expects RGB. log scale so
            # close-range is not buried at the dark end. Phantom pixels
            # (0 < depth <= viz_min_depth) are painted magenta so they stand
            # out from genuine "no Gaussian hit" (black).
            gt_bgr = colorize_depth(gt_depth_2d, dmin, dmax, mask=gt_mask_2d, log_scale=True)
            rd_bgr = colorize_depth(rd_depth_2d, dmin, dmax, mask=pred_hit_any, log_scale=True)
            if pred_phantom.any():
                rd_bgr[pred_phantom] = (255, 0, 255)  # BGR magenta
            gt_dvis = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
            rd_vis = cv2.cvtColor(rd_bgr, cv2.COLOR_BGR2RGB)
            rr.log(f"{ri_prefix}/depth/gt", rr.Image(gt_dvis))
            rr.log(f"{ri_prefix}/depth/rendered", rr.Image(rd_vis))

            gt_ivis = cv2.cvtColor(
                colorize_intensity(gt_intensity_np.squeeze(-1), mask=gt_mask_2d), cv2.COLOR_BGR2RGB
            )
            ri_vis = cv2.cvtColor(
                colorize_intensity(rendered_intensity_np.squeeze(-1), mask=pred_hit_any), cv2.COLOR_BGR2RGB
            )
            rr.log(f"{ri_prefix}/intensity/gt", rr.Image(gt_ivis))
            rr.log(f"{ri_prefix}/intensity/rendered", rr.Image(ri_vis))

            # Per-sensor depth error. Keep using the raydrop-filtered pred
            # depth here (= simulated LiDAR output) so MAE reflects post-UNet
            # quality, not raw Gaussian rendering.
            rd_filtered = rd_depth_2d * mask.squeeze(-1)
            valid = (gt_rayhit_np.squeeze(-1) > 0) & (rd_filtered > 0)
            if valid.any():
                err_map = np.zeros_like(gt_depth_np.squeeze(-1))
                err_map[valid] = np.abs(gt_depth_np.squeeze(-1)[valid] - rd_filtered[valid])
                err = err_map[valid]
                frame_mae_values.append(float(err.mean()))
                if multi_sensor:
                    rr.log(f"metrics/MAE_{sensor_name}", rr.Scalars(float(err.mean())))

        # Ego position (from first sensor)
        ego2world = first_lidar.ego2world[frame_id]
        if torch.is_tensor(ego2world):
            ego2world = ego2world.cpu().numpy()
        rr.log("world/ego", rr.Points3D([ego2world[:3, 3] - origin_offset], colors=[[255, 0, 0]], radii=[0.5]))

        # Bounding boxes
        centers, sizes, quats, labels = [], [], [], []
        for obj_id, bbox in bboxes.items():
            if frame_id in bbox.frame:
                pos, quat, _, _ = bbox.frame[frame_id]
                centers.append(pos.cpu().numpy() - origin_offset)
                s = bbox.size.cpu().numpy()
                if data_type == "T4":
                    sizes.append(s[[1, 0, 2]])
                else:
                    sizes.append(s)
                q_wxyz = quat.squeeze(0).cpu().numpy()
                quats.append(q_wxyz[[1, 2, 3, 0]])
                labels.append(str(obj_id)[:8])
        if centers:
            rr.log("world/bboxes", rr.Boxes3D(
                centers=np.array(centers), sizes=np.array(sizes),
                quaternions=np.array(quats), labels=labels,
                colors=[[0, 255, 0]] * len(centers),
            ))

        # Aggregate metrics
        if frame_mae_values:
            rr.log("metrics/MAE (m/point)", rr.Scalars(float(np.mean(frame_mae_values))))

        frame_label = "EVAL" if frame_id in eval_frames else "TRAIN"
        rr.log("metadata/frame_type", rr.TextDocument(f"Frame {frame_id} ({frame_label})"))

    dt = time.time() - t_render
    print(f"Rendering done in {dt:.1f}s ({len(all_frames)/dt:.1f} fps)", flush=True)
    print(f"Open with: rerun {save_path}", flush=True)


if __name__ == "__main__":
    main()
