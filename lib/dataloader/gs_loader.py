import math
import os
import random
from typing import Dict

import numpy as np
import open3d as o3d
import torch
from lib.scene import BoundingBox, GaussianModel, LiDARSensor, Scene, dataset_readers
from lib.utils import general_utils
from lib.utils.camera_utils import cameraList_from_camInfos
from lib.utils.general_utils import build_rotation
from lib.utils.graphics_utils import BasicPointCloud
from PIL import Image


class SceneLidar(Scene):
    def __init__(self, args, waymo_raw_pkg, shuffle=True, resize_ratio=1, test=False):
        scene_id = (
            str(args.scene_id) if isinstance(args.scene_id, int) else args.scene_id
        )
        self.output_dir = os.path.join(
            args.model_dir, args.task_name, args.exp_name, "scene_" + scene_id
        )
        self.model_save_dir = os.path.join(self.output_dir, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.loaded_iter = None
        self.camera_extent = 0
        self.gaussians_assets = [
            GaussianModel(
                args.model.dimension, args.model.sh_degree, extent=self.camera_extent
            )
        ]

        # Multi-sensor support: lidars_dict is dict[str, LiDARSensor]
        lidars_dict: Dict[str, LiDARSensor] = waymo_raw_pkg[0]
        bboxes: Dict[str, BoundingBox] = waymo_raw_pkg[1]
        frame_range = args.frame_length
        eval_frames = args.eval_frames
        train_frames = [
            frame_id
            for frame_id in range(frame_range[0], frame_range[1] + 1)
            if frame_id not in eval_frames
        ]

        # Skip frames whose range images have missing azimuth bands (rosbag
        # packet drops, off-by-one chunk reads, etc). Detection is per-sensor:
        # a frame is dropped if ANY sensor flagged it. Opt-in via config
        # (skip_dropped_frames, default off for non-T4 datasets).
        skip_drops = getattr(args, "skip_dropped_frames", False)
        weak_col_thr = float(getattr(args, "dropped_frame_threshold", 0.01))
        weak_hit_rate = float(getattr(args, "dropped_frame_hit_rate", 0.5))
        if skip_drops:
            dropped = set()
            for sensor_name, lidar in lidars_dict.items():
                bad = lidar.detect_dropped_frames(
                    weak_col_threshold=weak_col_thr,
                    weak_hit_rate=weak_hit_rate,
                )
                if bad:
                    print(f"[{sensor_name}] detected {len(bad)} dropped frame(s) "
                          f"(weak_col > {weak_col_thr:.0%}, "
                          f"col hit rate < {weak_hit_rate:.0%}):")
                    for fid, frac in bad:
                        print(f"    frame {fid:>3}: weak_col={frac*100:.1f}%")
                    dropped.update(fid for fid, _ in bad)
            if dropped:
                before_train, before_eval = len(train_frames), len(eval_frames)
                train_frames = [f for f in train_frames if f not in dropped]
                eval_frames = [f for f in eval_frames if f not in dropped]
                print(f"[skip_dropped_frames] train: {before_train} -> {len(train_frames)}, "
                      f"eval: {before_eval} -> {len(eval_frames)}")

        self.train_lidars = lidars_dict
        for sensor_name, lidar in self.train_lidars.items():
            lidar.set_frames(train_frames, eval_frames)

        # Backward compat: train_lidar returns first sensor
        self._sensor_names = list(self.train_lidars.keys())
        print(f"[Loaded] {len(self._sensor_names)} LiDAR sensor(s): {self._sensor_names}")

        # initialize objects with bounding boxes
        if args.dynamic:
            obj_ids = list(bboxes.keys())
            for obj_id in obj_ids:
                bbox = bboxes[obj_id]
                general_utils.fill_zeros_with_previous_nonzero(
                    range(frame_range[0], frame_range[1] + 1), bbox.frame
                )

                abs_velocities = []
                for frame in range(frame_range[0], frame_range[1]):
                    velocity = bbox.frame[frame + 1][0] - bbox.frame[frame][0]
                    abs_velocities.append(torch.norm(velocity).item())
                avg_velocity = torch.tensor(abs_velocities).mean().item()

                if avg_velocity > 0.01 and bbox.object_type == 1:
                    extent = (
                        torch.norm(bbox.size, keepdim=False).item()
                        * args.model.object_extent_factor
                    )
                    gaussian_model = GaussianModel(
                        args.model.dimension,
                        args.model.sh_degree,
                        extent=extent,
                        bounding_box=bbox,
                    )
                    gaussian_model.tmp_points_intensities_list = []
                    self.gaussians_assets.append(gaussian_model)

            if not bboxes:
                print("No dynamic objects in the scene")
                args.dynamic = False

        # initialize bkgd points — aggregate from ALL sensors
        all_points = []
        all_intensity = []
        all_normals = []
        for frame in range(frame_range[0], frame_range[1] + 1):
            # Collect points and normals per-sensor from r1 range image (GPU, fast)
            frame_pts_list = []
            frame_int_list = []
            frame_normals_list = []
            for sensor_name, lidar in self.train_lidars.items():
                depth = lidar.get_depth(frame)  # (H, W)
                intensity_map = lidar.get_intensity(frame)  # (H, W)
                hit_mask = lidar.get_mask(frame)  # (H, W)
                # Points from r1 range image
                pts_3d = lidar.range2point(frame, depth)  # (H, W, 3) on GPU
                pts = pts_3d[hit_mask].cpu()
                intensity = intensity_map[hit_mask].cpu()
                # Normals from range image finite differences (GPU)
                normal_map = lidar.get_normal(frame)[0]  # (H, W, 3)
                sensor_normals = normal_map[hit_mask].cpu()
                # Replace zero normals (boundary pixels) with random unit vectors
                zero_mask = (sensor_normals.norm(dim=-1) < 1e-6)
                if zero_mask.any():
                    n_zero = zero_mask.sum().item()
                    rand_normals = torch.randn(n_zero, 3)
                    rand_normals = rand_normals / rand_normals.norm(dim=-1, keepdim=True)
                    sensor_normals[zero_mask] = rand_normals
                frame_pts_list.append(pts)
                frame_int_list.append(intensity)
                frame_normals_list.append(sensor_normals)

            lidar_pts = torch.cat(frame_pts_list, dim=0)
            lidar_intensity = torch.cat(frame_int_list, dim=0)
            normals = torch.cat(frame_normals_list, dim=0)
            for gaussian_model in self.gaussians_assets[1:]:
                bbox = gaussian_model.bounding_box
                T = bbox.frame[frame][0].cpu()
                R = build_rotation(bbox.frame[frame][1])[0].cpu()
                points_in_local = (lidar_pts - T) @ R.inverse().T
                normals_in_local = normals @ R.inverse().T
                mask = (torch.abs(points_in_local) < bbox.size.cpu() / 2).all(dim=1)
                gaussian_model.tmp_points_intensities_list.append(
                    (
                        points_in_local[mask],
                        lidar_intensity[mask],
                        normals_in_local[mask],
                    )
                )
                lidar_pts, lidar_intensity = lidar_pts[~mask], lidar_intensity[~mask]
                normals = normals[~mask]

            all_points.append(lidar_pts)
            all_intensity.append(lidar_intensity)
            all_normals.append(normals)

        all_points = torch.cat(all_points, dim=0)
        all_intensity = torch.cat(all_intensity, dim=0)
        all_normals = torch.cat(all_normals, dim=0)
        hit_probs = torch.ones(all_points.shape[0])
        drop_probs = torch.zeros(all_points.shape[0])
        ip = torch.stack([all_intensity, hit_probs, drop_probs], dim=1)

        if args.opt.use_voxel_init:
            points_lidar = o3d.geometry.PointCloud()
            points_lidar.points = o3d.utility.Vector3dVector(
                all_points.cpu().numpy().astype(np.float64)
            )
            points_lidar.colors = o3d.utility.Vector3dVector(
                ip.cpu().numpy().astype(np.float64)
            )
            points_lidar.normals = o3d.utility.Vector3dVector(
                all_normals.cpu().numpy().astype(np.float64)
            )
            downsample_points_lidar = points_lidar.voxel_down_sample(
                voxel_size=args.model.voxel_size
            )
            all_points = torch.tensor(
                np.asarray(downsample_points_lidar.points)
            ).float()
            ip = torch.tensor(np.asarray(downsample_points_lidar.colors)).float()
            normals = torch.tensor(np.asarray(downsample_points_lidar.normals)).float()
        else:
            mask = torch.randperm(all_points.shape[0])[
                : all_points.shape[0] // (frame_range[1] - frame_range[0]) * 5
            ]
            all_points = all_points[mask]
            ip = ip[mask]
            normals = all_normals[mask]
        # calcualte extent
        scene_center = all_points.mean(dim=0)
        point_extent = 2 * torch.norm(all_points - scene_center, dim=1)
        self.camera_extent = (
            args.model.bkgd_extent_factor
            * torch.quantile(point_extent, 0.90).int().item()
        )
        self.gaussians_assets[0].extent = self.camera_extent

        pcd = BasicPointCloud(all_points, ip, normals=normals)
        self.gaussians_assets[0].create_from_pcd(pcd, args.opt.use_normal_init)

        # initialize objects points
        points_num = args.model.obj_pt_num
        for gaussian_model in self.gaussians_assets[1:]:
            points = torch.cat(
                [point for point, _, _ in gaussian_model.tmp_points_intensities_list],
                dim=0,
            )
            intensities = torch.cat(
                [
                    intensitie
                    for _, intensitie, _ in gaussian_model.tmp_points_intensities_list
                ],
                dim=0,
            )
            normals = torch.cat(
                [normal for _, _, normal in gaussian_model.tmp_points_intensities_list],
                dim=0,
            )
            if points.shape[0] < points_num:
                extra_num = points_num - points.shape[0]
                extra_points = torch.zeros((extra_num, 3))
                for i in range(3):
                    extra_points[:, i] = (
                        torch.rand(size=(extra_num,)) * bbox.size[i].cpu()
                        + bbox.min_xyz[i].cpu()
                    )
                points = torch.cat([points, extra_points], dim=0)

                extra_points_intensity = torch.rand(extra_num)
                intensities = torch.cat([intensities, extra_points_intensity], dim=0)

                theta = np.random.uniform(0, 2 * np.pi, extra_num)
                phi = np.random.uniform(0, np.pi, extra_num)
                extra_normals = np.zeros((extra_num, 3))
                extra_normals[:, 0] = np.sin(phi) * np.cos(theta)
                extra_normals[:, 1] = np.sin(phi) * np.sin(theta)
                extra_normals[:, 2] = np.cos(phi)

                normals = torch.cat(
                    [normals, torch.tensor(extra_normals).float()], dim=0
                )

            elif points.shape[0] > points_num:
                mask = torch.randperm(points.shape[0])[:points_num]
                points = points[mask]
                intensities = intensities[mask]
                normals = normals[mask]

            hit_probs = torch.ones(points.shape[0])
            drop_probs = torch.zeros(points.shape[0])
            ip = torch.stack([intensities, hit_probs, drop_probs], dim=1)
            pcd = BasicPointCloud(points, ip, normals=normals)
            gaussian_model.create_from_pcd(pcd, args.opt.use_normal_init)
            del gaussian_model.tmp_points_intensities_list

        print("[Loaded] object guassians")

    @property
    def train_lidar(self):
        """Backward compat: return first sensor."""
        return self.train_lidars[self._sensor_names[0]]

    def training_setup(self, args):
        for gs in self.gaussians_assets:
            gs.training_setup(args)

    def restore(self, model_params, args):
        for i, gs in enumerate(self.gaussians_assets):
            gs.restore(model_params[i], args)

    def update_learning_rate(self, iteration):
        for gs in self.gaussians_assets:
            gs.update_learning_rate(iteration)

    def oneupSHdegree(self):
        for gs in self.gaussians_assets:
            gs.oneupSHdegree()

    def save(self, iteration, model_name):
        model_pth = os.path.join(self.model_save_dir, model_name + ".pth")
        model_params = []
        for i, gs in enumerate(self.gaussians_assets):
            model_params.append(gs.capture())
        torch.save((model_params, iteration), model_pth)

    def optimize(
        self,
        args,
        iteration,
        mean_grads,
        accum_weights,
        visibility_filter_list,
        radii_list,
        sky_weights=None,
        front_weights=None,
        occupancy_grid=None,
    ):

        clone_num, split_num, prune_scale_num, prune_opacity_num, prune_sky_num, prune_aniso_num, prune_front_num, prune_occ_num = 0, 0, 0, 0, 0, 0, 0, 0

        sky_prune_enabled_global = bool(getattr(args.opt, "sky_prune_enabled", False))
        sky_prune_warmup_iter = int(getattr(args.opt, "sky_prune_warmup_iter", 0))
        sky_ratio_threshold = float(getattr(args.opt, "sky_prune_pixel_ratio_threshold", 0.8))
        sky_min_total_contrib = float(getattr(args.opt, "sky_prune_min_total_contrib", 1e-3))
        sky_view_consistency = float(getattr(args.opt, "sky_prune_view_consistency_threshold", 0.8))
        sky_min_views = int(getattr(args.opt, "sky_prune_min_views", 3))

        aniso_prune_enabled_global = bool(getattr(args.opt, "aniso_prune_enabled", False))
        aniso_prune_warmup_iter = int(getattr(args.opt, "aniso_prune_warmup_iter", 0))
        max_aniso_prune = float(getattr(args.opt, "max_aniso_prune", 10.0))
        aniso_prune_max_opacity = float(getattr(args.opt, "aniso_prune_max_opacity", 0.5))

        front_prune_enabled_global = bool(getattr(args.opt, "front_prune_enabled", False))
        front_prune_warmup_iter = int(getattr(args.opt, "front_prune_warmup_iter", 0))
        front_ratio_threshold = float(getattr(args.opt, "front_prune_pixel_ratio_threshold", 0.8))
        front_min_total_contrib = float(getattr(args.opt, "front_prune_min_total_contrib", 1e-3))
        front_view_consistency = float(getattr(args.opt, "front_prune_view_consistency_threshold", 0.8))
        front_min_views = int(getattr(args.opt, "front_prune_min_views", 3))

        occupancy_prune_enabled_global = bool(getattr(args.opt, "occupancy_prune_enabled", False))
        occupancy_prune_warmup_iter = int(getattr(args.opt, "occupancy_prune_warmup_iter", 0))
        occupancy_prune_opacity_threshold = float(getattr(args.opt, "occupancy_prune_opacity_threshold", 0.5))

        begin_index = 0
        for gaussians in self.gaussians_assets:
            points_num = gaussians.get_local_xyz.shape[0]
            instance_mean_grads = mean_grads[begin_index : begin_index + points_num]
            instance_accum_weights = (
                accum_weights[begin_index : begin_index + points_num] > 0
            )
            # Accumulate sky / front-side stats per asset before begin_index
            # advances. Only background Gaussians (no bounding box) get
            # multi-view phantom-pruned — object Gaussians live inside a
            # tracking box and these criteria are not valid for them.
            asset_sky_active = (sky_weights is not None
                                and sky_prune_enabled_global
                                and gaussians.bounding_box is None
                                and iteration >= sky_prune_warmup_iter)
            asset_front_active = (front_weights is not None
                                  and front_prune_enabled_global
                                  and gaussians.bounding_box is None
                                  and iteration >= front_prune_warmup_iter)
            if asset_sky_active or asset_front_active:
                instance_total = accum_weights[begin_index : begin_index + points_num]
                # Shared denominator: bump view_count once per active view.
                # min_total_contrib uses whichever sub-stat is active (they
                # should agree; both default to 1e-3).
                bump_thresh = (sky_min_total_contrib if asset_sky_active
                               else front_min_total_contrib)
                gaussians.add_view_count(instance_total,
                                         min_total_contrib=bump_thresh)
                if asset_sky_active:
                    instance_sky = sky_weights[begin_index : begin_index + points_num]
                    gaussians.add_sky_stats(
                        instance_total, instance_sky,
                        min_total_contrib=sky_min_total_contrib,
                        sky_ratio_threshold=sky_ratio_threshold,
                    )
                if asset_front_active:
                    instance_front = front_weights[begin_index : begin_index + points_num]
                    gaussians.add_front_stats(
                        instance_total, instance_front,
                        min_total_contrib=front_min_total_contrib,
                        front_ratio_threshold=front_ratio_threshold,
                    )
            begin_index += points_num

            # Per-asset cap: each asset (bg, each object) is gated on its
            # own point count, not the sum across assets. Previously bg and
            # all object Gaussians shared one cap; once total crossed it,
            # densify_and_prune stopped firing for bg too, which froze the
            # min_range_prune / low-opacity / bbox-escape cleanup at the
            # same time and left phantoms accumulated near the sensor.
            #
            # Pruning runs every densification step regardless of the cap
            # (see skip_densify below); only clone/split is skipped when an
            # asset is over its own cap.
            max_points = getattr(args.opt, "densify_until_num_points", -1)
            asset_points = gaussians.get_local_xyz.shape[0]
            under_cap = max_points < 0 or asset_points < max_points
            if iteration < args.opt.densify_until_iter:
                gaussians.add_densification_stats(
                    instance_mean_grads, instance_accum_weights
                )

                if (
                    iteration > args.opt.densify_from_iter
                    and iteration % args.opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > args.opt.opacity_reset_interval else None
                    )
                    # Gather all training-frame sensor centers (background-only prune)
                    sensor_centers = None
                    if gaussians.bounding_box is None:
                        center_list = []
                        for lidar in self.train_lidars.values():
                            for c in lidar.sensor_center.values():
                                center_list.append(c)
                        if center_list:
                            sensor_centers = torch.stack(center_list).cuda()
                    asset_sky_prune_enabled = (
                        sky_prune_enabled_global
                        and gaussians.bounding_box is None
                        and iteration >= sky_prune_warmup_iter
                    )
                    asset_aniso_prune_enabled = (
                        aniso_prune_enabled_global
                        and gaussians.bounding_box is None
                        and iteration >= aniso_prune_warmup_iter
                    )
                    asset_front_prune_enabled = (
                        front_prune_enabled_global
                        and gaussians.bounding_box is None
                        and iteration >= front_prune_warmup_iter
                    )
                    asset_occupancy_prune_enabled = (
                        occupancy_prune_enabled_global
                        and gaussians.bounding_box is None
                        and iteration >= occupancy_prune_warmup_iter
                        and occupancy_grid is not None
                    )
                    densify_info = gaussians.densify_and_prune(
                        args.opt, 0.005, size_threshold,
                        sensor_centers=sensor_centers,
                        min_range_prune=getattr(args.opt, "min_range_prune", 0.0),
                        skip_densify=not under_cap,
                        sky_prune_enabled=asset_sky_prune_enabled,
                        sky_view_consistency_threshold=sky_view_consistency,
                        sky_prune_min_views=sky_min_views,
                        aniso_prune_enabled=asset_aniso_prune_enabled,
                        max_aniso_prune=max_aniso_prune,
                        aniso_prune_max_opacity=aniso_prune_max_opacity,
                        front_prune_enabled=asset_front_prune_enabled,
                        front_view_consistency_threshold=front_view_consistency,
                        front_prune_min_views=front_min_views,
                        occupancy_grid=occupancy_grid if asset_occupancy_prune_enabled else None,
                        occupancy_prune_enabled=asset_occupancy_prune_enabled,
                        occupancy_prune_opacity_threshold=occupancy_prune_opacity_threshold,
                    )
                    clone_num += densify_info[0]
                    split_num += densify_info[1]
                    prune_scale_num += densify_info[2]
                    prune_opacity_num += densify_info[3]
                    prune_sky_num += densify_info[4]
                    prune_aniso_num += densify_info[5]
                    prune_front_num += densify_info[6]
                    prune_occ_num += densify_info[7]

                if iteration % args.opt.opacity_reset_interval == 0 or (
                    args.model.white_background
                    and iteration == args.opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # args.optimizer step
            if iteration < args.opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                # Floor _scaling so Adam can't drive σ = exp(_scaling) below the
                # OptiX degenerate-primitive threshold (~1e-8). Without this,
                # post-densify_until_iter the scale parameter drifts negative
                # unchecked — we observed 37% of all Gaussians collapsed to
                # σ < 1e-4 by iter 30000, triggering the BVH builder's
                # EXCESSIVE_DEGENERATE_PRIMITIVES warning. A 1e-6 floor stays
                # well below any structural surfel size while keeping the BVH
                # well-formed. Configurable via opt.min_scale (linear metres).
                min_scale = float(getattr(args.opt, "min_scale", 1e-6))
                if min_scale > 0:
                    gaussians._scaling.data.clamp_(min=math.log(min_scale))

        return (clone_num, split_num, prune_scale_num, prune_opacity_num,
                prune_sky_num, prune_aniso_num, prune_front_num, prune_occ_num)
