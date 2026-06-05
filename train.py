# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import argparse
import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_USE_CUDA_DSA"] = "1"
import torch
import torch.nn.functional as F
import yaml
from lib import dataloader
from lib.arguments import parse
from lib.gaussian_renderer import raytracing
from lib.scene import Scene
from lib.scene.unet import UNet
from lib.utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from lib.utils.console_utils import *
from lib.utils.image_utils import mse, psnr, colorize_depth, colorize_intensity
from lib.utils.loss_utils import (
    BinaryCrossEntropyLoss,
    BinaryFocalLoss,
    l1_loss,
    l2_loss,
    ssim,
)
from lib.utils.record_utils import make_recorder
from ruamel.yaml import YAML
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb

    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False


def set_seed(seed):
    """
    Useless function, result still have a 1e-7 difference.
    Need to test problem in optix.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def training(args):
    first_iter = 0

    color = cv2.COLORMAP_JET

    scene = dataloader.load_scene(args.source_dir, args, test=False)
    gaussians_assets = scene.gaussians_assets
    scene.training_setup(args.opt)
    log = {
        "depth_mse": [],
        "points_num": [],
        "clone_sum": [],
        "split_sum": [],
        "prune_scale_sum": [],
        "prune_opacity_sum": [],
        "prune_sky_sum": [],
        "prune_aniso_sum": [],
    }
    scene_id = str(args.scene_id) if isinstance(args.scene_id, int) else args.scene_id
    output_dir = os.path.join(
        args.model_dir, args.task_name, args.exp_name, "scene_" + scene_id
    )
    record_dir = os.path.join(output_dir, "records")
    recorder = make_recorder(args, record_dir)
    print(
        blue(
            f"Task: {args.task_name}, Experiment: {args.exp_name}, Scene: {args.scene_id}"
        )
    )
    print("Output dir: ", output_dir)

    # Initialize wandb
    if WANDB_FOUND:
        wandb_config = {
            "task_name": args.task_name,
            "exp_name": args.exp_name,
            "scene_id": args.scene_id,
            "data_type": getattr(args, "data_type", None),
            "iterations": args.opt.iterations,
            "lr_xyz": getattr(args.opt, "position_lr_init", None),
            "lr_feature": getattr(args.opt, "feature_lr", None),
            "lr_opacity": getattr(args.opt, "opacity_lr", None),
            "lr_scaling": getattr(args.opt, "scaling_lr", None),
            "lr_rotation": getattr(args.opt, "rotation_lr", None),
            "lambda_depth_l1": getattr(args.opt, "lambda_depth_l1", None),
            "lambda_intensity_l1": getattr(args.opt, "lambda_intensity_l1", None),
            "lambda_intensity_l2": getattr(args.opt, "lambda_intensity_l2", None),
            "lambda_intensity_dssim": getattr(args.opt, "lambda_intensity_dssim", None),
            "lambda_raydrop_bce": getattr(args.opt, "lambda_raydrop_bce", None),
            "lambda_cd": getattr(args.opt, "lambda_cd", None),
            "lambda_reg": getattr(args.opt, "lambda_reg", None),
            "lambda_sky": getattr(args.opt, "lambda_sky", None),
            "lambda_freespace": getattr(args.opt, "lambda_freespace", None),
            "lambda_front_acc": getattr(args.opt, "lambda_front_acc", None),
            "lambda_occupancy": getattr(args.opt, "lambda_occupancy", None),
            "occupancy_voxel_size": getattr(args.opt, "occupancy_voxel_size", None),
            "occupancy_warmup_iter": getattr(args.opt, "occupancy_warmup_iter", None),
            "min_range_prune": getattr(args.opt, "min_range_prune", None),
            "max_depth": getattr(args, "max_depth", None),
            "densify_until_iter": getattr(args.opt, "densify_until_iter", None),
            "densify_from_iter": getattr(args.opt, "densify_from_iter", None),
            "densification_interval": getattr(args.opt, "densification_interval", None),
            "use_refine": getattr(args.refine, "use_refine", False),
        }
        wandb.init(
            entity="advanced-technology-department",
            project="LiDAR-RT-debug",
            name=f"{args.exp_name}_scene{scene_id}",
            config=wandb_config,
        )

    if args.model_path:
        (model_params, first_iter) = torch.load(args.model_path)
        scene.restore(model_params, args.opt)
        with open(os.path.join(output_dir, "logs/log.json"), "r") as json_file:
            log = json.load(json_file)
    print("Continuing from iteration ", first_iter)

    # bg_color = [1, 1, 1] if args.model.white_background else [0, 0, 0]
    background = torch.tensor(
        [0, 0, 1], device="cuda"
    ).float()  # background (intensity, hit prob, drop prob)

    BFLoss = BinaryFocalLoss()
    BCELoss = BinaryCrossEntropyLoss()
    frame_stack = []  # list of (sensor_name, frame_id) tuples

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        initial=first_iter, total=args.opt.iterations, desc="Training progress"
    )
    first_iter += 1

    # Build a one-shot world-frame occupancy grid from training LiDAR points
    # with dynamic-bbox interiors removed. Used downstream to penalise the
    # opacity of bg Gaussians that sit in voxels no LiDAR return ever fell
    # into — a structural anti-phantom signal independent of sky masks.
    lambda_occupancy = float(getattr(args.opt, "lambda_occupancy", 0.0))
    occupancy_warmup_iter = int(getattr(args.opt, "occupancy_warmup_iter", 0))
    occupancy_grid = None
    if lambda_occupancy > 0:
        from lib.scene.occupancy_grid import WorldOccupancyGrid
        occ_voxel_size = float(getattr(args.opt, "occupancy_voxel_size", 0.5))
        occupancy_grid = WorldOccupancyGrid(voxel_size=occ_voxel_size)
        stats = occupancy_grid.build(scene.train_lidars, gaussians_assets[1:])
        print(f"[Occupancy] voxel_size={stats['voxel_size']}  "
              f"frames={stats['n_frames']}  points={stats['n_points']}  "
              f"occupied_voxels={stats['n_voxels']}")

    end = time.time()
    frame_s, frame_e = args.frame_length[0], args.frame_length[1]
    render_cams = []
    best_mix_metric = 0
    for iteration in range(first_iter, args.opt.iterations + 1):
        if args.only_refine:
            break
        iter_start.record()
        recorder.step += 1

        scene.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            scene.oneupSHdegree()

        # Pick a random (sensor, frame) pair
        if not frame_stack:
            frame_stack = [
                (sensor_name, fid)
                for sensor_name, lidar in scene.train_lidars.items()
                for fid in lidar.train_frames
            ]
            random.shuffle(frame_stack)
        sensor_name, frame = frame_stack.pop()
        cur_lidar = scene.train_lidars[sensor_name]
        data_time = time.time() - end

        # Render
        if args.pipe.debug_from and (iteration - 1) == args.pipe.debug_from:
            args.pipe.debug = True

        # Compute sky mask up front so we can pass it as pixel_weight to the
        # rasterizer — the kernel then accumulates per-Gaussian alpha*T*sky for
        # the sky hard-prune path. The same mask is reused below for the
        # sky/free-space losses to avoid double computation.
        gt_mask = cur_lidar.get_mask(frame).cuda()
        sky_kernel = int(getattr(args.opt, "sky_morph_kernel", 3))
        if sky_kernel > 1:
            gt_f = gt_mask.float().unsqueeze(0).unsqueeze(0)
            pad = sky_kernel // 2
            dilated = F.max_pool2d(gt_f, sky_kernel, stride=1, padding=pad)
            closed = -F.max_pool2d(-dilated, sky_kernel, stride=1, padding=pad)
            gt_mask_closed = closed.squeeze(0).squeeze(0) > 0.5
            sky_mask = ~gt_mask_closed
        else:
            sky_mask = ~gt_mask
        sky_prune_enabled = bool(getattr(args.opt, "sky_prune_enabled", False))
        pixel_weight = sky_mask.float() if sky_prune_enabled else None

        # Front-side accumulation target depth: for pixels where GT returned
        # a hit, set target = gt_depth so the kernel snapshots accumulated
        # alpha BEFORE the real surface (phantom-in-front signal). For sky
        # pixels, set target = large sentinel so the snapshot equals the
        # full-ray integral — subsumes the old freespace_loss behaviour
        # when lambda_front_acc is non-zero.
        lambda_front_acc = float(getattr(args.opt, "lambda_front_acc", 0.0))
        if lambda_front_acc > 0:
            gt_depth_pre = cur_lidar.get_depth(frame).cuda()
            target_depth = torch.where(
                gt_mask, gt_depth_pre, torch.full_like(gt_depth_pre, 1e6)
            )
        else:
            target_depth = None

        render_pkg = raytracing(
            frame, gaussians_assets, cur_lidar, background, args,
            pixel_weight=pixel_weight,
            target_depth=target_depth,
        )
        batch_time = time.time() - end
        depth = render_pkg["depth"]
        intensity = render_pkg["intensity"]
        raydrop_prob = render_pkg["raydrop"]
        accum = render_pkg["accum"]
        means3d = render_pkg["means3D"]
        acc_wet = render_pkg["accum_gaussian_weight"]
        acc_sky_wet = render_pkg["accum_gaussian_sky_weight"]
        accum_at_target = render_pkg["accum_at_target"]

        H, W = depth.shape[0], depth.shape[1]

        # === Depth loss ===
        depth = depth.squeeze(-1)
        gt_depth = cur_lidar.get_depth(frame).cuda()
        loss_depth = args.opt.lambda_depth_l1 * l1_loss(
            depth[gt_mask], gt_depth[gt_mask]
        )

        # === Intensity loss ===
        intensity = intensity.squeeze(-1)
        gt_intensity = cur_lidar.get_intensity(frame).cuda()
        loss_intensity = (
            args.opt.lambda_intensity_l1
            * l1_loss(intensity[gt_mask], gt_intensity[gt_mask])
            + args.opt.lambda_intensity_l2
            * l2_loss(intensity[gt_mask], gt_intensity[gt_mask])
            + args.opt.lambda_intensity_dssim
            * (
                1
                - ssim(
                    (intensity * gt_mask).unsqueeze(0),
                    (gt_intensity * gt_mask).unsqueeze(0),
                )
            )
        )

        # === Raydrop loss ===
        raydrop_prob = raydrop_prob.reshape(-1, 1)
        labels_idx = (
            ~gt_mask
        )  # (1, h, w) notice: hit is true (1). apply ~ to make idx 0 represent hit
        labels = labels_idx.reshape(-1, 1)  # (h*w, 1)

        loss_raydrop = args.opt.lambda_raydrop_bce * BCELoss(labels, preds=raydrop_prob)

        # === CD loss ===
        chamLoss = chamfer_3DDist()
        gt_pts = cur_lidar.inverse_projection_with_range(
            frame, gt_depth, gt_mask
        )
        pred_pts = cur_lidar.inverse_projection_with_range(
            frame, depth, gt_mask
        )

        dist1, dist2, _, _ = chamLoss(pred_pts[None, ...], gt_pts[None, ...])
        chamfer_loss = (dist1 + dist2).mean() * 0.5
        loss_cd = args.opt.lambda_cd * chamfer_loss

        # === regularization loss ===
        loss_reg = 0
        for gaussians in gaussians_assets:
            loss_reg += args.opt.lambda_reg * gaussians.box_reg_loss()

        # === sky transparency loss ===
        # For sky-direction rays (no GT return), the rendered depth should be 0.
        # A non-zero rendered depth there means a phantom Gaussian intercepted
        # the ray; pushing depth toward 0 forces those Gaussians to thin out.
        #
        # We morphologically close gt_mask so isolated few-pixel "no return"
        # spots (real ray-drops on glass, wet surfaces, low-reflectance, etc.
        # — surrounded by hits) are treated as hits. Without this the sky
        # loss would also suppress Gaussians at those genuine-drop pixels and
        # rob the SH / intensity supervision of any learning signal there.
        #
        # We average over phantom pixels only (sky ∩ depth>0), NOT over the
        # full sky mask. Averaging over the full sky dilutes the per-phantom
        # gradient by N_sky / N_phantom (often 20x+), so each phantom Gaussian
        # receives much weaker pressure than the nominal lambda suggests. The
        # per-phantom mean keeps the gradient magnitude scene-invariant.
        lambda_sky = getattr(args.opt, "lambda_sky", 0.0)
        if lambda_sky > 0:
            phantom_in_sky = sky_mask & (depth > 0)
            if phantom_in_sky.any():
                loss_sky = lambda_sky * depth[phantom_in_sky].mean()
            else:
                loss_sky = torch.tensor(0.0, device="cuda")
        else:
            loss_sky = torch.tensor(0.0, device="cuda")

        # === free-space loss ===
        # Pushes the accumulated alpha W = sum_i alpha_i * T_i toward 0 on
        # rays that did not return (sky / dropout). Operates directly on the
        # opacity channel via the tracer's ACCUM output, so the per-Gaussian
        # gradient is +λ * T_i / (1-α_i) — always positive regardless of
        # what lies behind, unlike sky_loss whose sign can flip when there
        # is a background hit on the same ray and which actively *grows*
        # foreground phantoms in that case. Averaged over the phantom set
        # only to keep magnitude scene-invariant.
        lambda_fs = getattr(args.opt, "lambda_freespace", 0.0)
        if lambda_fs > 0:
            phantom_in_sky_fs = sky_mask & (accum.squeeze(-1) > 0)
            if phantom_in_sky_fs.any():
                loss_freespace = lambda_fs * accum.squeeze(-1)[phantom_in_sky_fs].mean()
            else:
                loss_freespace = torch.tensor(0.0, device="cuda")
        else:
            loss_freespace = torch.tensor(0.0, device="cuda")

        # === Front-side accumulation loss ===
        # Penalises alpha accumulated strictly in front of the LiDAR's GT hit
        # (or to infinity for sky rays). Directly addresses Tokyo / Hesai
        # OT128 horizon-ring phantoms that survive sky_loss / freespace_loss
        # — those only operate on sky pixels, whereas horizon phantoms sit
        # on rays that DO return a building hit at moderate range.
        if lambda_front_acc > 0:
            front_phantom = accum_at_target > 0
            if front_phantom.any():
                loss_front_acc = lambda_front_acc * accum_at_target[front_phantom].mean()
            else:
                loss_front_acc = torch.tensor(0.0, device="cuda")
        else:
            loss_front_acc = torch.tensor(0.0, device="cuda")

        # === 3D occupancy loss ===
        # Penalises the opacity of background Gaussians whose centres sit in
        # voxels no training LiDAR return ever fell into (dynamic-bbox
        # interiors excluded at grid build time). Operates only on the bg
        # asset — object Gaussians live inside their bboxes whose interiors
        # were stripped from the grid, so every object Gaussian would
        # otherwise be falsely judged free. Per-phantom mean keeps the
        # gradient magnitude scene-invariant, matching the sky/free-space
        # loss pattern.
        if (lambda_occupancy > 0 and occupancy_grid is not None
                and iteration >= occupancy_warmup_iter):
            bg_gs = gaussians_assets[0]
            bg_xyz_world = bg_gs.get_world_xyz()  # bg has bounding_box=None → just _xyz
            is_free = occupancy_grid.free_mask(bg_xyz_world)
            if is_free.any():
                loss_occupancy = lambda_occupancy * bg_gs.get_opacity[is_free].mean()
            else:
                loss_occupancy = torch.tensor(0.0, device="cuda")
        else:
            loss_occupancy = torch.tensor(0.0, device="cuda")

        loss = loss_depth + loss_intensity + loss_raydrop + loss_cd + loss_reg + loss_sky + loss_freespace + loss_occupancy + loss_front_acc

        # Skip iteration if loss is NaN/Inf (numerical instability in tracer)
        if not torch.isfinite(loss):
            print(f"\n[ITER {iteration}] Warning: non-finite loss ({loss.item():.4f}), skipping")
            for gs in gaussians_assets:
                gs.optimizer.zero_grad(set_to_none=True)
            continue

        try:
            loss.backward()
        except RuntimeError as e:
            if "nan" in str(e).lower():
                print(f"\n[ITER {iteration}] Warning: NaN in backward pass, skipping")
                for gs in gaussians_assets:
                    gs.optimizer.zero_grad(set_to_none=True)
                continue
            raise

        # Check for NaN gradients and skip if found
        has_nan_grad = False
        for gs in gaussians_assets:
            if gs._xyz.grad is not None and not torch.isfinite(gs._xyz.grad).all():
                has_nan_grad = True
                break
        if has_nan_grad:
            print(f"\n[ITER {iteration}] Warning: NaN in gradients, skipping")
            for gs in gaussians_assets:
                gs.optimizer.zero_grad(set_to_none=True)
            continue

        with torch.no_grad():
            densify_info = scene.optimize(
                args, iteration, means3d.grad, acc_wet, None, None,
                sky_weights=acc_sky_wet,
            )

            points_num = 0
            for i in gaussians_assets:
                points_num += i.get_local_xyz.shape[0]
            depth_mse = mse(depth[gt_mask], gt_depth[gt_mask]).mean().item()
            depth_mae = torch.abs(depth[gt_mask] - gt_depth[gt_mask]).mean().item()
            depth_rmse = depth_mse ** 0.5
            clone_sum = (
                densify_info[0] + log["clone_sum"][-1]
                if log["clone_sum"]
                else densify_info[0]
            )
            split_sum = (
                densify_info[1] + log["split_sum"][-1]
                if log["split_sum"]
                else densify_info[1]
            )
            prune_scale_sum = (
                densify_info[2] + log["prune_scale_sum"][-1]
                if log["prune_scale_sum"]
                else densify_info[2]
            )
            prune_opacity_sum = (
                densify_info[3] + log["prune_opacity_sum"][-1]
                if log["prune_opacity_sum"]
                else densify_info[3]
            )
            prune_sky_sum = (
                densify_info[4] + log["prune_sky_sum"][-1]
                if log.get("prune_sky_sum")
                else densify_info[4]
            )
            prune_aniso_sum = (
                densify_info[5] + log["prune_aniso_sum"][-1]
                if log.get("prune_aniso_sum")
                else densify_info[5]
            )
            log["depth_mse"].append(depth_mse)
            log["points_num"].append(points_num)
            log["clone_sum"].append(clone_sum)
            log["split_sum"].append(split_sum)
            log["prune_scale_sum"].append(prune_scale_sum)
            log["prune_opacity_sum"].append(prune_opacity_sum)
            log.setdefault("prune_sky_sum", []).append(prune_sky_sum)
            log.setdefault("prune_aniso_sum", []).append(prune_aniso_sum)

            # prepare loss stats for tensorboard record
            loss_stats = {
                "all_loss": loss,
                "depth_loss": loss_depth,
                "intensity_loss": loss_intensity,
                "ema_loss": 0.4 * loss + 0.6 * ema_loss_for_log,
                "points_num": torch.tensor(points_num).float(),
                "depth_mse": torch.tensor(depth_mse).float(),
            }

            reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
            recorder.update_loss_stats(reduced_losses)

            if WANDB_FOUND:
                # Raydrop accuracy metrics
                pred_drop = (render_pkg["raydrop"].reshape(-1) > 0.5)
                gt_drop = labels_idx.reshape(-1)
                rd_tp = (pred_drop & gt_drop).sum().item()
                rd_fp = (pred_drop & ~gt_drop).sum().item()
                rd_fn = (~pred_drop & gt_drop).sum().item()
                rd_precision = rd_tp / max(rd_tp + rd_fp, 1)
                rd_recall = rd_tp / max(rd_tp + rd_fn, 1)
                rd_f1 = 2 * rd_precision * rd_recall / max(rd_precision + rd_recall, 1e-8)
                rd_accuracy = (pred_drop == gt_drop).float().mean().item()

                # Intensity loss components (unweighted)
                int_l1 = l1_loss(intensity[gt_mask], gt_intensity[gt_mask]).item()
                int_l2 = l2_loss(intensity[gt_mask], gt_intensity[gt_mask]).item()
                int_ssim_val = ssim(
                    (intensity * gt_mask).unsqueeze(0),
                    (gt_intensity * gt_mask).unsqueeze(0),
                ).item()

                # Phantom pixel count on the *current training frame* (cheap:
                # reuses the main depth tensor instead of an extra render).
                # Logged every iter rather than gated on visual_interval so we
                # can watch the trend at fine granularity, especially around
                # opacity_reset (3K, 6K, ...) where phantom counts can change
                # sharply. The viz/* version stays as the fixed-frame variant
                # for cross-iter visual comparison.
                viz_min_depth_train = float(getattr(args, "viz_min_depth", 0.0))
                with torch.no_grad():
                    train_phantom_mask = (depth > 0) & (depth <= viz_min_depth_train)
                    train_phantom_pixels = int(train_phantom_mask.sum().item())

                wandb.log(
                    {
                        # Total losses
                        "train/loss": loss.item(),
                        "train/ema_loss": (0.4 * loss + 0.6 * ema_loss_for_log).item(),
                        # Depth
                        "train/depth_loss": loss_depth.item(),
                        "train/depth_mse": depth_mse,
                        "train/depth_rmse": depth_rmse,
                        "train/depth_mae": depth_mae,
                        # Near-range phantom pixel count (per current frame)
                        "train/phantom_pixels": train_phantom_pixels,
                        # Intensity (weighted total + unweighted components)
                        "train/intensity_loss": loss_intensity.item(),
                        "train/intensity_l1": int_l1,
                        "train/intensity_l2": int_l2,
                        "train/intensity_ssim": int_ssim_val,
                        # Raydrop
                        "train/raydrop_loss": loss_raydrop.item(),
                        "train/raydrop_accuracy": rd_accuracy,
                        "train/raydrop_precision": rd_precision,
                        "train/raydrop_recall": rd_recall,
                        "train/raydrop_f1": rd_f1,
                        # Chamfer distance (unweighted + weighted)
                        "train/chamfer_loss": chamfer_loss.item(),
                        "train/cd_loss": loss_cd.item(),
                        # Regularization
                        "train/reg_loss": loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg,
                        # Sky transparency
                        "train/sky_loss": loss_sky.item(),
                        # Free-space supervision
                        "train/freespace_loss": loss_freespace.item(),
                        # Front-side accumulation supervision
                        "train/front_acc_loss": loss_front_acc.item(),
                        # 3D occupancy supervision
                        "train/occupancy_loss": loss_occupancy.item(),
                        # Densification
                        "train/points_num": points_num,
                        "train/clone_sum": clone_sum,
                        "train/split_sum": split_sum,
                        "train/prune_scale_sum": prune_scale_sum,
                        "train/prune_opacity_sum": prune_opacity_sum,
                        "train/prune_sky_sum": prune_sky_sum,
                        "train/prune_aniso_sum": prune_aniso_sum,
                        # Learning rate
                        "train/lr_xyz": gaussians_assets[0].optimizer.param_groups[0]["lr"],
                    },
                    step=iteration,
                )

            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)
            recorder.record("train")

            if iteration % args.visual_interval == 0:
                # Pick a non-dropped frame for visualization. train_lidar.train_frames
                # has already been filtered by skip_dropped_frames in the loader.
                viz_frames = scene.train_lidar.train_frames or [frame_s]
                viz_frame = viz_frames[0]
                render_pkg = raytracing(
                    viz_frame, gaussians_assets, scene.train_lidar, background, args  # first sensor for viz
                )
                rendered_depth = render_pkg["depth"]
                rendered_intensity = render_pkg["intensity"]
                rendered_raydrop = render_pkg["raydrop"]

                # GT for side-by-side comparison
                gt_depth_viz = scene.train_lidar.get_depth(viz_frame).cuda()
                gt_mask_viz = scene.train_lidar.get_mask(viz_frame).cuda()

                # Use 0..max(GT, rendered) so near-range phantom Gaussians (depth
                # below GT's minimum) are still visible in the colormap rather
                # than getting clipped to the darkest color.
                if gt_mask_viz.any():
                    dmax = max(
                        float(gt_depth_viz.max().item()),
                        float(rendered_depth.max().item()),
                    )
                else:
                    dmax = float(rendered_depth.max().item())
                dmin = 0.0

                # Predicted hit: any Gaussian intercepted the ray. Tri-state
                # for the pred panel: BLACK = no Gaussian (depth==0), MAGENTA
                # = phantom hit at 0 < depth <= viz_min_depth (hardware-min
                # violation), JET log-scale = depth > viz_min_depth (normal).
                rendered_depth_2d = rendered_depth.squeeze(-1).detach().cpu().numpy()
                gt_depth_np = gt_depth_viz.detach().cpu().numpy()
                gt_mask_np = gt_mask_viz.detach().cpu().numpy().astype(bool)
                viz_min_depth = float(getattr(args, "viz_min_depth", 0.0))
                pred_hit_any = rendered_depth_2d > 0
                pred_phantom = pred_hit_any & (rendered_depth_2d <= viz_min_depth)

                # log scale so close-range (1-10 m) is not buried at the dark
                # end of a 0-200 m linear ramp.
                gt_depth_img = colorize_depth(gt_depth_np, dmin, dmax, mask=gt_mask_np, log_scale=True)
                pred_depth_img = colorize_depth(rendered_depth_2d, dmin, dmax, mask=pred_hit_any, log_scale=True)
                if pred_phantom.any():
                    pred_depth_img[pred_phantom] = (255, 0, 255)  # BGR magenta

                # Background-only render: same viewpoint, but with object
                # Gaussians stripped from the asset list. Lets us see what the
                # static background alone is reconstructing — useful for
                # diagnosing whether near-range / dynamic-object hits in the
                # full-asset panel are coming from bg phantoms or genuine
                # tracked-object Gaussians.
                if len(gaussians_assets) > 1:
                    # raytracing's default branch assumes >=1 object asset and
                    # crashes on torch.cat(obj_rot[1:]) when given a bg-only
                    # list. Use decomp="background" so the renderer takes its
                    # bg-only code path (gaussian_assets[:1] + rot_in_local[0]
                    # only, no object concat).
                    bg_render_pkg = raytracing(
                        viz_frame, gaussians_assets, scene.train_lidar,
                        background, args, decomp="background",
                    )
                    bg_depth_2d = bg_render_pkg["depth"].squeeze(-1).detach().cpu().numpy()
                else:
                    bg_depth_2d = rendered_depth_2d
                bg_hit_any = bg_depth_2d > 0
                bg_phantom = bg_hit_any & (bg_depth_2d <= viz_min_depth)
                bg_depth_img = colorize_depth(bg_depth_2d, dmin, dmax,
                                              mask=bg_hit_any, log_scale=True)
                if bg_phantom.any():
                    bg_depth_img[bg_phantom] = (255, 0, 255)  # BGR magenta

                # Per-pixel |gt - pred| at pixels where both are valid.
                err_valid = gt_mask_np & pred_hit_any
                err_map = np.zeros_like(gt_depth_np, dtype=np.float32)
                err_map[err_valid] = np.abs(
                    gt_depth_np[err_valid] - rendered_depth_2d[err_valid]
                )
                # Cap colormap at the 99th percentile of valid errors so a few
                # huge outliers (e.g., a phantom Gaussian) don't wash out the rest.
                if err_valid.any():
                    err_cap = float(np.quantile(err_map[err_valid], 0.99))
                else:
                    err_cap = 1.0
                err_cap = max(err_cap, 1e-3)
                error_img = colorize_depth(err_map, 0.0, err_cap, mask=err_valid)

                concat_image = np.concatenate(
                    [gt_depth_img, pred_depth_img, bg_depth_img, error_img], axis=0
                )
                rgb_image = concat_image
                os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
                cv2.imwrite(
                    os.path.join(output_dir, "images", str(iteration) + ".png"),
                    rgb_image,
                )
                render_cams.append(rgb_image)

                if WANDB_FOUND:
                    # cv2 returns BGR; wandb expects RGB
                    log_payload = {
                        "viz/depth_compare": wandb.Image(
                            cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB),
                            caption=(
                                f"iter {iteration} | frame {viz_frame} | top→bot: "
                                f"depth_gt / depth_rendered(all) / depth_rendered(bg only) / "
                                f"error (cap={err_cap:.2f}m)"
                            ),
                        ),
                        # Magenta pixel counts — the per-frame proxy for
                        # how many near-range phantoms (0 < depth <=
                        # viz_min_depth) are still surviving. Logging the
                        # all-asset and bg-only counts separately so an
                        # increase can be attributed to bg drift vs
                        # tracked-object Gaussians that legitimately sit
                        # close to the sensor on this frame.
                        "viz/phantom_pixels_all": int(pred_phantom.sum()),
                        "viz/phantom_pixels_bg": int(bg_phantom.sum()),
                    }
                    # Depth-error histogram over pixels where BOTH GT and the
                    # rendered image have a return (err_valid). Fixed bins so
                    # the distribution is comparable across iterations: 50 bins
                    # spanning 0-50 m (1 m each). Pixels with no-return on
                    # either side are excluded by construction (err_valid).
                    if err_valid.any():
                        err_vals = err_map[err_valid].astype(np.float32)
                        # Clip into the last bin instead of dropping outliers:
                        # a tall right-most bar tells you "X pixels had error
                        # >= 50 m" rather than silently shrinking the
                        # distribution.
                        bin_edges = np.linspace(0.0, 50.0, 51, dtype=np.float32)
                        err_vals = np.minimum(err_vals, bin_edges[-1] - 1e-3)
                        counts, _ = np.histogram(err_vals, bins=bin_edges)
                        log_payload["viz/depth_error_hist"] = wandb.Histogram(
                            np_histogram=(counts, bin_edges)
                        )
                    wandb.log(log_payload, step=iteration)

                # === Intensity comparison (same viz_frame, same sensor) ===
                # Mirrors the depth panel: top=GT, mid=rendered, bot=|err|.
                # T4 LiDAR stores raw uint8 intensity (p50≈12, p99≈64,
                # retroreflectors at 255), so we use the same log-ramp
                # [0, 64] colormap as the rerun 3D point cloud — keeps the
                # wandb 2D panel visually aligned with whatever the user
                # sees in rerun. GT and Render are colorised with the same
                # parameters (no fading) so the only visible difference
                # between the two rows is the actual reconstruction error.
                gt_intensity_viz = scene.train_lidar.get_intensity(viz_frame).cuda()
                gt_intensity_np = gt_intensity_viz.detach().cpu().numpy()
                rendered_intensity_2d = rendered_intensity.squeeze(-1).detach().cpu().numpy()

                int_vmin, int_vmax = 0.0, 64.0
                gt_intensity_img = colorize_intensity(
                    gt_intensity_np, mask=gt_mask_np,
                    vmin=int_vmin, vmax=int_vmax, log_scale=True,
                )
                pred_intensity_img = colorize_intensity(
                    rendered_intensity_2d, mask=pred_hit_any,
                    vmin=int_vmin, vmax=int_vmax, log_scale=True,
                )

                int_err_valid = gt_mask_np & pred_hit_any
                int_err_map = np.zeros_like(gt_intensity_np, dtype=np.float32)
                int_err_map[int_err_valid] = np.abs(
                    gt_intensity_np[int_err_valid] - rendered_intensity_2d[int_err_valid]
                )
                if int_err_valid.any():
                    int_err_cap = float(np.quantile(int_err_map[int_err_valid], 0.99))
                else:
                    int_err_cap = 1.0
                int_err_cap = max(int_err_cap, 1e-3)
                # colorize_depth normalises to [vmin, vmax] linearly which is
                # exactly the behaviour we want for the intensity error map.
                int_error_img = colorize_depth(int_err_map, 0.0, int_err_cap,
                                               mask=int_err_valid)

                intensity_concat = np.concatenate(
                    [gt_intensity_img, pred_intensity_img, int_error_img], axis=0
                )
                cv2.imwrite(
                    os.path.join(output_dir, "images",
                                 str(iteration) + "_intensity.png"),
                    intensity_concat,
                )

                if WANDB_FOUND:
                    wandb.log({
                        "viz/intensity_compare": wandb.Image(
                            cv2.cvtColor(intensity_concat, cv2.COLOR_BGR2RGB),
                            caption=(
                                f"iter {iteration} | frame {viz_frame} | top→bot: "
                                f"intensity_gt / intensity_rendered / |error| "
                                f"(cap={int_err_cap:.3f})"
                            ),
                        ),
                    }, step=iteration)

            # Progress bar
            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log.item():.{5}f}",
                        # "L_all": f"{loss.item():.{5}f}",
                        # "L_depth": f"{loss_depth.item():.{5}f}",
                        # "L_intensity": f"{loss_intensity.item():.{5}f}",
                        # "L_raydrop": f"{loss_raydrop.item():.{5}f}",
                        "points": f"{points_num}",
                        "exp": args.exp_name,
                        "scene": args.scene_id,
                    }
                )
                progress_bar.update(10)
            if iteration == args.opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in args.saving_iterations:
                progress_bar.write("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, "model_it_" + str(iteration))

            if iteration % args.testing_iterations == 0:
                if iteration >= args.saving_iterations[0] - 3000:
                    mix_metric = 0
                    eval_count = 0
                    eval_depth_mae_sum = 0
                    eval_depth_rmse_sum = 0
                    eval_psnr_depth_sum = 0
                    eval_psnr_intensity_sum = 0
                    for eval_sensor_name, eval_lidar in scene.train_lidars.items():
                        for frame in eval_lidar.eval_frames:
                            render_pkg = raytracing(
                                frame, gaussians_assets, eval_lidar, background, args
                            )
                            depth = render_pkg["depth"].detach()
                            intensity = render_pkg["intensity"].detach()
                            raydrop_prob = render_pkg["raydrop"].detach()
                            mask = raydrop_prob < 0.5

                            gt_depth = eval_lidar.get_depth(frame).cuda()
                            gt_intensity = eval_lidar.get_intensity(frame).cuda()
                            gt_mask = eval_lidar.get_mask(frame).cuda()
                            depth_norm = getattr(args, "max_depth", 80)
                            psnr_depth = (
                                psnr(
                                    depth[..., 0] * mask[..., 0] / depth_norm,
                                    gt_depth * gt_mask / depth_norm,
                                )
                                .mean()
                                .item()
                            )
                            intensity = intensity.clamp(0, 1)
                            gt_intensity = gt_intensity.clamp(0, 1)
                            psnr_intensity = (
                                psnr(
                                    intensity[..., 0] * mask[..., 0], gt_intensity * gt_mask
                                )
                                .mean()
                                .item()
                            )
                            # Per-valid-pixel depth metrics
                            valid = gt_mask & mask[..., 0]
                            if valid.sum() > 0:
                                eval_depth_mae_sum += torch.abs(depth[..., 0][valid] - gt_depth[valid]).mean().item()
                                eval_depth_rmse_sum += (mse(depth[..., 0][valid], gt_depth[valid]).mean().item()) ** 0.5
                            eval_psnr_depth_sum += psnr_depth
                            eval_psnr_intensity_sum += psnr_intensity
                            mix_metric += psnr_depth + psnr_intensity
                            eval_count += 1
                    mix_metric /= max(eval_count, 1)
                    if WANDB_FOUND:
                        wandb.log(
                            {
                                "eval/mix_metric": mix_metric,
                                "eval/depth_mae": eval_depth_mae_sum / max(eval_count, 1),
                                "eval/depth_rmse": eval_depth_rmse_sum / max(eval_count, 1),
                                "eval/psnr_depth": eval_psnr_depth_sum / max(eval_count, 1),
                                "eval/psnr_intensity": eval_psnr_intensity_sum / max(eval_count, 1),
                            },
                            step=iteration,
                        )
                    print(mix_metric, best_mix_metric)
                    if mix_metric > best_mix_metric:
                        for file in os.listdir(scene.model_save_dir):
                            if file.endswith(".pth") and "ckpt_it_" in file:
                                os.remove(os.path.join(scene.model_save_dir, file))
                        best_mix_metric = mix_metric
                        scene.save(iteration, "ckpt_it_" + str(iteration) + "_good")
                else:
                    previous_checkpoint_nopfix = os.path.join(
                        scene.model_save_dir,
                        "ckpt_it_" + str(iteration - args.testing_iterations) + ".pth",
                    )
                    if os.path.exists(previous_checkpoint_nopfix):
                        os.remove(previous_checkpoint_nopfix)

                    progress_bar.write(
                        "\n[ITER {}] Saving Checkpoint".format(iteration)
                    )
                    scene.save(iteration, "ckpt_it_" + str(iteration))

                logging(log, output_dir)

        iter_end.record()

    # Final degenerate-Gaussian cleanup. After densify_until_iter the
    # densify-time prune stops running, but Adam keeps updating _scaling and
    # _opacity, so by the end of training the assets accumulate Gaussians
    # whose σ has collapsed below 1e-6 or whose opacity sits below
    # thresh_opa_prune. OptiX flags those as degenerate primitives during the
    # final-sweep BVH build (EXCESSIVE_DEGENERATE_PRIMITIVES warning with
    # percentage: .inf). Strip them here so any downstream raytracing pass
    # (sky-prune sweep, viz, refine) sees a clean asset list.
    if not args.only_refine:
        cleanup_min_scale = float(getattr(args.opt, "min_scale", 1e-6))
        cleanup_opacity = float(getattr(args.opt, "thresh_opa_prune", 0.001))
        total_cleaned = 0
        for i, gs in enumerate(gaussians_assets):
            n_before = gs.get_local_xyz.shape[0]
            if n_before == 0:
                continue
            sigma = torch.exp(gs._scaling.detach())
            bad_scale = (sigma < cleanup_min_scale).any(dim=-1)
            opa = torch.sigmoid(gs._opacity.detach()).view(-1)
            bad_opa = opa < cleanup_opacity
            bad = bad_scale | bad_opa
            n_bad = int(bad.sum().item())
            if 0 < n_bad < n_before:
                gs.prune_points(bad)
                total_cleaned += n_bad
                print(f"  [Final cleanup] asset[{i}]: "
                      f"bad_scale={int(bad_scale.sum().item())} "
                      f"bad_opacity={int(bad_opa.sum().item())} "
                      f"pruned={n_bad}/{n_before}")
        print(f"[Final cleanup] total pruned (degenerate scale / "
              f"opacity<{cleanup_opacity}): {total_cleaned}")
        if WANDB_FOUND and total_cleaned > 0:
            wandb.log({"train/final_cleanup_total": total_cleaned},
                      step=args.opt.iterations)

    # Final sky-mask hard prune: deterministic full-view sweep over every
    # training (sensor, frame), then prune background Gaussians whose
    # sky-pixel contribution concentrates across enough views. Catches any
    # sky phantoms that survived the in-training densify-time prune (which
    # only sees the per-cycle 100-iter window). Gated on sky_prune_enabled
    # — disable via sky_prune_final_sweep=false in config to skip just the
    # final pass while keeping the in-training one.
    sky_prune_enabled_global = bool(getattr(args.opt, "sky_prune_enabled", False))
    final_sweep_enabled = bool(getattr(args.opt, "sky_prune_final_sweep", True))
    if (not args.only_refine
            and sky_prune_enabled_global
            and final_sweep_enabled):
        from lib.scene.sky_prune import sweep_and_sky_prune
        ratio_thr = float(getattr(args.opt, "sky_prune_pixel_ratio_threshold", 0.8))
        view_cons = float(getattr(args.opt, "sky_prune_view_consistency_threshold", 0.8))
        min_v = int(getattr(args.opt, "sky_prune_min_views", 3))
        min_tc = float(getattr(args.opt, "sky_prune_min_total_contrib", 1e-3))
        print(f"\n[Final sky-prune] sweeping with ratio>{ratio_thr}, "
              f"view_consistency>{view_cons}, min_views={min_v}")
        with torch.no_grad():
            stats = sweep_and_sky_prune(
                gaussians_assets,
                scene.train_lidars,
                args,
                ratio_threshold=ratio_thr,
                view_consistency=view_cons,
                min_views=min_v,
                min_total_contrib=min_tc,
                progress_desc="Final sky-prune sweep",
            )
        for i, st in stats["per_asset"].items():
            print(f"  asset[{i}] (bg) Gaussians={st['n_before']}  "
                  f"observed_any={st['observed_any']}  "
                  f"observed>={min_v}={st['observed_min']}  "
                  f"to_prune={st['n_prune']}")
        print(f"[Final sky-prune] total hard-pruned: {stats['pruned']}")
        scene.save(args.opt.iterations,
                   "model_it_" + str(args.opt.iterations) + "_skyprune")
        if WANDB_FOUND:
            wandb.log({"train/final_sky_prune_total": stats["pruned"]},
                      step=args.opt.iterations)

    if args.refine.use_refine:
        print(output_dir)
        in_channels = 9 if args.refine.use_spatial else 3

        # Per-sensor UNet (range image resolution differs across sensors)
        unets = {}
        unet_optimizers = {}
        for sname in scene.train_lidars:
            unet = UNet(in_channels=in_channels, out_channels=1).cuda()
            unets[sname] = unet
            unet_optimizers[sname] = torch.optim.Adam(unet.parameters(), lr=0.001)

        for epoch in tqdm(range(0, args.refine.epochs), desc="Refine raydrop"):
            for iter in range(0, args.refine.batch_size):
                if not frame_stack:
                    frame_stack = [
                        (sname, fid)
                        for sname, lidar in scene.train_lidars.items()
                        for fid in lidar.train_frames
                    ]
                    random.shuffle(frame_stack)
                sensor_name, frame = frame_stack.pop()
                cur_lidar = scene.train_lidars[sensor_name]
                unet = unets[sensor_name]

                render_pkg = raytracing(
                    frame, gaussians_assets, cur_lidar, background, args
                )
                depth = render_pkg["depth"].detach()
                intensity = render_pkg["intensity"].detach()
                raydrop_prob = render_pkg["raydrop"].detach()

                H, W = depth.shape[0], depth.shape[1]
                input_depth = depth.reshape(1, H, W)
                input_intensity = intensity.reshape(1, H, W)
                input_raydrop = raydrop_prob.reshape(1, H, W)
                raydrop_prob = torch.cat(
                    [input_raydrop, input_intensity, input_depth], dim=0
                )
                if args.refine.use_spatial:
                    ray_o, ray_d = cur_lidar.get_range_rays(frame)
                    raydrop_prob = torch.cat(
                        [raydrop_prob, ray_o.permute(2, 0, 1), ray_d.permute(2, 0, 1)],
                        dim=0,
                    )
                raydrop_prob = raydrop_prob.unsqueeze(0)
                if args.refine.use_rot:
                    rot = torch.randint(0, W, (1,))
                    raydrop_prob = torch.cat(
                        [raydrop_prob[:, :, :, rot:], raydrop_prob[:, :, :, :rot]],
                        dim=-1,
                    )
                raydrop_prob = unet(raydrop_prob)

                raydrop_prob = raydrop_prob.reshape(-1, 1)

                gt_mask = cur_lidar.get_mask(frame).cuda()
                labels_idx = (
                    ~gt_mask
                )  # (1, h, w) notice: hit is true (1). apply ~ to make idx 0 represent hit
                if args.refine.use_rot:
                    labels_idx = torch.cat(
                        [labels_idx[:, rot:], labels_idx[:, :rot]], dim=-1
                    )
                labels = labels_idx.reshape(-1, 1)  # (h*w, 1)
                loss_raydrop = args.refine.lambda_raydrop_bce * BCELoss(
                    labels, preds=raydrop_prob
                )

                loss_raydrop.backward()

            for sname in scene.train_lidars:
                unet_optimizers[sname].step()
                unet_optimizers[sname].zero_grad()

        for sname, unet in unets.items():
            torch.save(unet.state_dict(), os.path.join(output_dir, "models", f"unet_{sname}.pth"))


def logging(log, output_dir):
    indices = range(len(log["depth_mse"]))

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color = "tab:blue"
    ax1.set_ylabel("Depth MSE", color=color)
    ax1.plot(indices, log["depth_mse"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color = "tab:red"

    ax2.set_ylabel("Points Num", color=color)
    clone_sum = np.array(log["clone_sum"])
    split_sum = np.array(log["split_sum"])
    prune_scale_sum = np.array(log["prune_scale_sum"])
    prune_opacity_sum = np.array(log["prune_opacity_sum"])

    plt.fill_between(indices, 0, clone_sum, label="clone_sum", color="blue", alpha=0.5)
    plt.fill_between(
        indices,
        clone_sum,
        clone_sum + split_sum,
        label="split_sum",
        color="green",
        alpha=0.5,
    )
    plt.fill_between(
        indices,
        clone_sum + split_sum,
        clone_sum + split_sum + prune_scale_sum,
        label="prune_scale_sum",
        color="red",
        alpha=0.5,
    )
    plt.fill_between(
        indices,
        clone_sum + split_sum + prune_scale_sum,
        clone_sum + split_sum + prune_scale_sum + prune_opacity_sum,
        label="prune_opacity_sum",
        color="yellow",
        alpha=0.5,
    )

    ax2.plot(indices, log["points_num"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, "log.png"))
    plt.close()
    with open(os.path.join(log_dir, "log.json"), "w") as json_file:
        json.dump(log, json_file, indent=4)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="launch args")
    parser.add_argument("-dc", "--data_config_path", type=str, help="config path")
    parser.add_argument("-ec", "--exp_config_path", type=str, help="config path")
    parser.add_argument("-m", "--model", type=str, help="the path to a checkpoint")
    parser.add_argument(
        "-r",
        "--only_refine",
        action="store_true",
        help="skip the training. only refine the model. E.g. load a checkpoint and only refine the unet to fit the checkpoint",
    )
    parser.add_argument(
        "-s",
        "--source_dir",
        type=str,
        help="override source_dir from data config",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=None,
        help="CUDA device ID to use (e.g. 0, 1). Defaults to current device.",
    )
    # Wandb-sweep / quick-experiment overrides. Each one falls back to the
    # config value when not passed, so manual runs are unaffected.
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override args.opt.iterations (e.g. 20000 for short sweeps)")
    parser.add_argument("--disable_refine", action="store_true",
                        help="Skip the post-training refinement stage")
    parser.add_argument("--lambda_occupancy", type=float, default=None)
    parser.add_argument("--occupancy_voxel_size", type=float, default=None)
    parser.add_argument("--occupancy_warmup_iter", type=int, default=None)
    parser.add_argument("--lambda_freespace", type=float, default=None)
    parser.add_argument("--lambda_front_acc", type=float, default=None)
    parser.add_argument("--lambda_sky", type=float, default=None)
    parser.add_argument("--exp_suffix", type=str, default="",
                        help="Append to exp_name (use to keep sweep run dirs distinct)")
    launch_args = parser.parse_args()

    args = parse(launch_args.exp_config_path)
    args = parse(launch_args.data_config_path, args)
    args.model_path = launch_args.model
    args.only_refine = launch_args.only_refine
    if launch_args.source_dir:
        args.source_dir = launch_args.source_dir

    # CLI overrides — apply BEFORE wandb.init so the logged config reflects
    # the actual values used for training (these end up in wandb run metadata).
    opt_overrides = {
        "lambda_occupancy": launch_args.lambda_occupancy,
        "occupancy_voxel_size": launch_args.occupancy_voxel_size,
        "occupancy_warmup_iter": launch_args.occupancy_warmup_iter,
        "lambda_freespace": launch_args.lambda_freespace,
        "lambda_front_acc": launch_args.lambda_front_acc,
        "lambda_sky": launch_args.lambda_sky,
    }
    for k, v in opt_overrides.items():
        if v is not None:
            setattr(args.opt, k, v)
            print(f"[override] args.opt.{k} = {v}")
    if launch_args.iterations is not None:
        args.opt.iterations = launch_args.iterations
        # position_lr_max_steps controls the cosine schedule and is normally
        # tied to iterations; keep them in lockstep so a short sweep doesn't
        # leave the xyz LR stuck at the start of the ramp.
        args.opt.position_lr_max_steps = launch_args.iterations
        print(f"[override] args.opt.iterations = {launch_args.iterations}  "
              f"(position_lr_max_steps tracked)")
    if launch_args.disable_refine:
        args.refine.use_refine = False
        print("[override] args.refine.use_refine = False")
    if launch_args.exp_suffix:
        args.exp_name = f"{args.exp_name}_{launch_args.exp_suffix}"
        print(f"[override] args.exp_name = {args.exp_name}")

    if launch_args.gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; cannot select device.")
        if launch_args.gpu < 0 or launch_args.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid --gpu {launch_args.gpu}: only {torch.cuda.device_count()} CUDA device(s) visible."
            )
        torch.cuda.set_device(launch_args.gpu)
        print(blue(f"Using CUDA device {launch_args.gpu}: {torch.cuda.get_device_name(launch_args.gpu)}"))

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.seed is not None:
        set_seed(args.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args)

    # All done
    if WANDB_FOUND:
        wandb.finish()
    print(blue("\nTraining complete."))
