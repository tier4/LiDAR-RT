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
from lib.utils.image_utils import color_mapping
from tqdm import tqdm

RAYDROP_RATIO = 0.4
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
    launch = parser.parse_args()

    args = parse(launch.exp_config_path)
    args = parse(launch.data_config_path, args)
    args.model_path = launch.model
    args.unet = launch.unet
    args.eval_type = launch.type
    args.rerun_save = launch.save
    return args


def load_scene_fast(args):
    """Load LiDAR data and bounding boxes WITHOUT heavy Gaussian initialization.

    Skips: normal estimation, voxel downsampling, optimizer setup.
    Only loads: range images, transforms, bounding boxes.
    """
    t0 = time.time()

    # Load raw LiDAR data (range images + transforms + bboxes)
    if getattr(args, "data_type", None) == "T4":
        from lib.dataloader import t4_loader
        lidar, bboxes = t4_loader.load_t4_raw(args.source_dir, args)
    elif "waymo" in args.source_dir:
        from lib.dataloader import waymo_loader
        lidar, bboxes = waymo_loader.load_waymo_raw(args.source_dir, args)
    elif "kitti" in args.source_dir:
        from lib.dataloader import kitti_loader
        lidar, bboxes = kitti_loader.load_kitti_raw(args.source_dir, args)
    else:
        raise ValueError("Unknown dataset type")

    # Setup frame lists on lidar sensor
    frame_range = args.frame_length
    eval_frames = args.eval_frames
    train_frames = [
        fid for fid in range(frame_range[0], frame_range[1] + 1)
        if fid not in eval_frames
    ]
    lidar.set_frames(train_frames, eval_frames)

    print(f"LiDAR data loaded in {time.time() - t0:.1f}s", flush=True)
    return lidar, bboxes


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


def main():
    args = build_args()
    save_path = args.rerun_save or "output/t4_test/vis_rerun.rrd"

    # --- Init Rerun ---
    rr.init("LiDAR-RT T4 Visualization")
    rr.save(save_path)
    print(f"Saving to {save_path}", flush=True)

    # --- Load data (fast path) ---
    lidar, bboxes = load_scene_fast(args)

    # Determine number of gaussian models from checkpoint
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    num_gaussians = len(checkpoint[0])
    del checkpoint  # free memory

    gaussians, first_iter = load_gaussians_fast(args.model_path, num_gaussians, args)

    # Load UNet
    unet = None
    if args.unet and os.path.exists(args.unet):
        in_channels = 9 if args.refine.use_spatial else 3
        unet = UNet(in_channels=in_channels, out_channels=1).cuda()
        unet.load_state_dict(torch.load(args.unet))
        print("Loaded UNet", flush=True)

    background = torch.tensor([0, 0, 1], device="cuda").float()

    # --- Determine frames ---
    eval_frames = args.eval_frames
    if args.eval_type == "train":
        all_frames = [f for f in range(args.frame_length[0], args.frame_length[1] + 1)
                      if f not in eval_frames]
    elif args.eval_type == "test":
        all_frames = eval_frames
    else:
        all_frames = list(range(args.frame_length[0], args.frame_length[1] + 1))

    print(f"Rendering {len(all_frames)} frames...", flush=True)

    # --- Static logging ---
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("metadata", rr.TextDocument(
        f"Model: {args.model_path}\nIteration: {first_iter}\n"
        f"Frames: {len(all_frames)} ({args.eval_type})"
    ))

    # --- Blueprint: 3D view large, range images on the right ---
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(
                    name="3D World (GT + Rendered)",
                    origin="world",
                    background=rrb.Background(color=[30, 30, 30]),
                ),
                row_shares=[1],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="Depth GT", contents="range_image/depth/gt/**"),
                rrb.Spatial2DView(name="Depth Rendered", contents="range_image/depth/rendered/**"),
                rrb.Spatial2DView(name="Intensity GT", contents="range_image/intensity/gt/**"),
                rrb.Spatial2DView(name="Intensity Rendered", contents="range_image/intensity/rendered/**"),
                rrb.TimeSeriesView(name="Depth Error", contents="metrics/**"),
                row_shares=[1, 1, 1, 1, 1],
            ),
            column_shares=[3, 1],
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
        rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
        rrb.TimePanel(state=rrb.PanelState.Expanded),
    )
    rr.send_blueprint(blueprint)

    # --- Compute origin offset from first frame's ego position ---
    first_ego = lidar.ego2world[all_frames[0]]
    if torch.is_tensor(first_ego):
        first_ego = first_ego.cpu().numpy()
    origin_offset = first_ego[:3, 3].copy()
    print(f"Origin offset (first frame ego): {origin_offset}", flush=True)

    # --- Render loop ---
    t_render = time.time()
    for frame_id in tqdm(all_frames, desc="Rendering"):
        rr.set_time_sequence("frame", frame_id)

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
        gt_intensity = lidar.get_intensity(frame_id).clamp(0, 1)

        gt_rayhit_np = gt_rayhit.cpu().numpy()
        gt_depth_np = gt_depth.unsqueeze(-1).cpu().numpy()
        gt_intensity_np = gt_intensity.unsqueeze(-1).cpu().numpy()
        rendered_rayhit_np = (rendered_raydrop < RAYDROP_RATIO).cpu().numpy()
        rendered_depth_np = rendered_depth.cpu().numpy()
        rendered_intensity_np = rendered_intensity.clamp(0, 1).cpu().numpy()
        mask = rendered_rayhit_np

        # 3D point clouds
        gt_pts = lidar.inverse_projection_with_range(
            frame_id, gt_depth_np, gt_rayhit_np
        ).cpu().numpy().astype(np.float64)
        rendered_pts = lidar.inverse_projection_with_range(
            frame_id, rendered_depth_np, mask
        ).cpu().numpy().astype(np.float64)

        gt_colors = depth_to_colors(np.linalg.norm(gt_pts, axis=1), 0.0, args.max_depth)
        rendered_colors = depth_to_colors(np.linalg.norm(rendered_pts, axis=1), 0.0, args.max_depth)

        gt_pts -= origin_offset
        rendered_pts -= origin_offset
        rr.log("world/pointcloud/gt", rr.Points3D(gt_pts, colors=gt_colors, radii=0.05))
        rr.log("world/pointcloud/rendered", rr.Points3D(rendered_pts, colors=rendered_colors, radii=0.05))

        # Ego position
        ego2world = lidar.ego2world[frame_id]
        if torch.is_tensor(ego2world):
            ego2world = ego2world.cpu().numpy()
        rr.log("world/ego", rr.Points3D([ego2world[:3, 3] - origin_offset], colors=[[255, 0, 0]], radii=[0.5]))

        # Bounding boxes
        centers, sizes, quats, labels = [], [], [], []
        for obj_id, bbox in bboxes.items():
            if frame_id in bbox.frame:
                pos, quat, _, _ = bbox.frame[frame_id]
                centers.append(pos.cpu().numpy() - origin_offset)
                sizes.append(bbox.size.cpu().numpy())
                quats.append(quat.squeeze(0).cpu().numpy())
                labels.append(str(obj_id)[:8])
        if centers:
            rr.log("world/bboxes", rr.Boxes3D(
                centers=np.array(centers), sizes=np.array(sizes),
                quaternions=np.array(quats), labels=labels,
                colors=[[0, 255, 0]] * len(centers),
            ))

        # Range images
        dmin = float(gt_depth_np[gt_rayhit_np.squeeze(-1) > 0].min()) if gt_rayhit_np.any() else 0.0
        dmax = float(gt_depth_np.max())

        gt_dvis = depth_to_colormap(gt_depth_np.squeeze(-1), dmin, dmax) * gt_rayhit_np.astype(np.uint8)
        rd_masked = rendered_depth_np * mask
        rd_vis = depth_to_colormap(rd_masked.squeeze(-1), dmin, dmax) * (mask & (rendered_depth_np > 0)).astype(np.uint8)
        rr.log("range_image/depth/gt", rr.Image(gt_dvis))
        rr.log("range_image/depth/rendered", rr.Image(rd_vis))

        gt_ivis = intensity_to_colormap(gt_intensity_np.squeeze(-1)) * gt_rayhit_np.astype(np.uint8)
        ri_vis = intensity_to_colormap(rendered_intensity_np.squeeze(-1)) * mask.astype(np.uint8)
        rr.log("range_image/intensity/gt", rr.Image(gt_ivis))
        rr.log("range_image/intensity/rendered", rr.Image(ri_vis))

        gt_mvis = (gt_rayhit_np.squeeze(-1) * 255).astype(np.uint8)
        rd_mvis = (rendered_rayhit_np.squeeze(-1) * 255).astype(np.uint8)
        rr.log("range_image/rayhit/gt", rr.Image(np.stack([gt_mvis]*3, axis=-1)))
        rr.log("range_image/rayhit/rendered", rr.Image(np.stack([rd_mvis]*3, axis=-1)))

        # Depth error as raw float for hover
        rr.log("range_image/depth_raw/gt", rr.DepthImage(gt_depth_np.squeeze(-1), meter=1.0))
        rr.log("range_image/depth_raw/rendered", rr.DepthImage(rd_masked.squeeze(-1), meter=1.0))

        # Per-frame metrics
        valid = (gt_rayhit_np.squeeze(-1) > 0) & (rd_masked.squeeze(-1) > 0)
        if valid.any():
            err = np.abs(gt_depth_np.squeeze(-1)[valid] - rd_masked.squeeze(-1)[valid])
            rr.log("metrics/depth_mae_m", rr.Scalars(float(err.mean())))
            rr.log("metrics/depth_rmse_m", rr.Scalars(float(np.sqrt((err**2).mean()))))

        frame_label = "EVAL" if frame_id in eval_frames else "TRAIN"
        rr.log("metadata/frame_type", rr.TextDocument(f"Frame {frame_id} ({frame_label})"))

    dt = time.time() - t_render
    print(f"Rendering done in {dt:.1f}s ({len(all_frames)/dt:.1f} fps)", flush=True)
    print(f"Open with: rerun {save_path}", flush=True)


if __name__ == "__main__":
    main()
