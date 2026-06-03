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
            "lambda_near": getattr(args.opt, "lambda_near", None),
            "min_range_prune": getattr(args.opt, "min_range_prune", None),
            "min_range_push": getattr(args.opt, "min_range_push", None),
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

        render_pkg = raytracing(
            frame, gaussians_assets, cur_lidar, background, args
        )
        batch_time = time.time() - end
        depth = render_pkg["depth"]
        intensity = render_pkg["intensity"]
        raydrop_prob = render_pkg["raydrop"]
        means3d = render_pkg["means3D"]
        acc_wet = render_pkg["accum_gaussian_weight"]

        H, W = depth.shape[0], depth.shape[1]

        gt_mask = cur_lidar.get_mask(frame).cuda()

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
        lambda_sky = getattr(args.opt, "lambda_sky", 0.0)
        if lambda_sky > 0:
            sky_mask = ~gt_mask
            if sky_mask.any():
                loss_sky = lambda_sky * depth[sky_mask].abs().mean()
            else:
                loss_sky = torch.tensor(0.0, device="cuda")
        else:
            loss_sky = torch.tensor(0.0, device="cuda")

        # === near-range push loss ===
        # Soft hinge: penalize any rendered depth below the LiDAR's effective
        # minimum range, pushing Gaussians away from the sensor's blind zone.
        # depth=0 (no Gaussian hit) is ignored.
        lambda_near = getattr(args.opt, "lambda_near", 0.0)
        min_range_push = getattr(args.opt, "min_range_push", 0.0)
        if lambda_near > 0 and min_range_push > 0:
            valid = depth > 0
            if valid.any():
                violation = torch.clamp(min_range_push - depth, min=0.0)
                loss_near = lambda_near * violation[valid].mean()
            else:
                loss_near = torch.tensor(0.0, device="cuda")
        else:
            loss_near = torch.tensor(0.0, device="cuda")

        loss = loss_depth + loss_intensity + loss_raydrop + loss_cd + loss_reg + loss_sky + loss_near

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
                args, iteration, means3d.grad, acc_wet, None, None
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
            log["depth_mse"].append(depth_mse)
            log["points_num"].append(points_num)
            log["clone_sum"].append(clone_sum)
            log["split_sum"].append(split_sum)
            log["prune_scale_sum"].append(prune_scale_sum)
            log["prune_opacity_sum"].append(prune_opacity_sum)

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
                        # Near-range push
                        "train/near_loss": loss_near.item(),
                        # Densification
                        "train/points_num": points_num,
                        "train/clone_sum": clone_sum,
                        "train/split_sum": split_sum,
                        "train/prune_scale_sum": prune_scale_sum,
                        "train/prune_opacity_sum": prune_opacity_sum,
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

                # Predicted hit mask: any Gaussian intercepted the ray. We
                # deliberately do NOT filter by the raw raydrop classifier here
                # — vis_rerun.py uses the same mask to keep the two viewers in
                # sync (UNet refinement is post-training and unavailable mid-run).
                rendered_depth_2d = rendered_depth.squeeze(-1).detach().cpu().numpy()
                gt_depth_np = gt_depth_viz.detach().cpu().numpy()
                gt_mask_np = gt_mask_viz.detach().cpu().numpy().astype(bool)
                pred_hit_any = rendered_depth_2d > 0

                gt_depth_img = colorize_depth(gt_depth_np, dmin, dmax, mask=gt_mask_np)
                pred_depth_img = colorize_depth(rendered_depth_2d, dmin, dmax, mask=pred_hit_any)

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
                    [gt_depth_img, pred_depth_img, error_img], axis=0
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
                    wandb.log({
                        "viz/depth_compare": wandb.Image(
                            cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB),
                            caption=(
                                f"iter {iteration} | frame {viz_frame} | top→bot: "
                                f"depth_gt / depth_rendered / error (cap={err_cap:.2f}m)"
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
    launch_args = parser.parse_args()

    args = parse(launch_args.exp_config_path)
    args = parse(launch_args.data_config_path, args)
    args.model_path = launch_args.model
    args.only_refine = launch_args.only_refine
    if launch_args.source_dir:
        args.source_dir = launch_args.source_dir

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
